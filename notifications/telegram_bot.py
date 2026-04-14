"""
Telegram 通知模組 — 交易信號、持倉更新、風險警報

Production-grade: 三位專家五輪交叉 debug 修正版
修復: polling 重複、command 並發、periodic report 重複、message debounce

功能:
- 開倉 / 平倉即時通知 (去重)
- 定期持倉報告 (每小時, debounce)
- 每日績效摘要 (每天 UTC+8 08:00)
- 風險警報 (MDD、異常等)
- Telegram 查詢指令 (/status, /positions, /pnl, /checksignal, etc.)
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from collections import deque
from datetime import datetime, timezone, timedelta
from typing import Optional

import aiohttp

logger = logging.getLogger("telegram")

HKT = timezone(timedelta(hours=8))

# ── Debounce / dedup constants ──
_MIN_REPORT_INTERVAL = 55       # position report 最小間隔 (秒)
_MIN_COMMAND_INTERVAL = 3       # 同一 command 最小間隔 (秒)
_POLL_TIMEOUT = 35              # client-side HTTP timeout (> server 30s long-poll)
_DEDUP_MAXLEN = 200             # processed update ID deque 上限
_EQUITY_SYNC_STALE = 300        # exchange equity cache TTL (秒)

# Per-symbol price display precision
PRICE_DECIMALS = {"FETUSDT": 4}


class TelegramNotifier:
    """Telegram Bot 通知器 — 生產級防重複"""

    def __init__(self, token: str = "", chat_id: str = ""):
        self.token = token or os.getenv("TELEGRAM_BOT_TOKEN", "")
        self.chat_id = chat_id or os.getenv("TELEGRAM_CHAT_ID", "")
        self.base_url = f"https://api.telegram.org/bot{self.token}"
        self._session: Optional[aiohttp.ClientSession] = None
        self._session_lock = asyncio.Lock()

        # ── Polling state ──
        self._poll_task: Optional[asyncio.Task] = None
        self._poll_lock = asyncio.Lock()          # 原子啟動 guard
        self._last_update_id = 0
        self._processed_updates: deque = deque(maxlen=_DEDUP_MAXLEN)  # O(1) append, bounded

        # ── Periodic report state ──
        self._periodic_task: Optional[asyncio.Task] = None
        self._periodic_lock = asyncio.Lock()      # 原子啟動 guard

        # ── Command concurrency control ──
        self._cmd_lock = asyncio.Lock()            # 防止同一 command 並行執行
        self._last_cmd_time: dict[str, float] = {} # command throttle

        # ── Message dedup ──
        self._last_report_time: float = 0          # position report debounce
        self._last_summary_time: float = 0         # daily summary debounce
        self._notified_entries: set = set()         # 防止同一筆 entry 通知兩次
        self._notified_exits: set = set()           # 防止同一筆 exit 通知兩次

        # ── State tracking ──
        self.positions: dict = {}       # symbol -> position info
        self.daily_trades: list = []    # today's trades
        self.daily_pnl: float = 0.0
        self.equity: float = 0.0
        self.initial_capital: float = 5000.0
        self.last_prices: dict = {}     # symbol -> latest price

        # Exchange-authoritative equity cache (from MEXC/Binance API)
        self.exchange_equity: Optional[float] = None
        self._last_equity_sync: float = 0.0

        # Orchestrator reference (injected at startup for command routing)
        self.orchestrator = None

        # Command handlers
        self._commands = {
            "/status": self._cmd_status,
            "/positions": self._cmd_positions,
            "/pnl": self._cmd_pnl,
            "/pause": self._cmd_pause,
            "/resume": self._cmd_resume,
            "/unblock_longs": self._cmd_unblock_longs,
            "/optimize": self._cmd_optimize,
            "/signal_on": self._cmd_signal_on,
            "/signal_off": self._cmd_signal_off,
            "/checksignal": self._cmd_checksignal,
            "/validate": self._cmd_validate,
            "/help": self._cmd_help,
        }

    @property
    def enabled(self) -> bool:
        return bool(self.token and self.chat_id)

    def reconfigure(self, token: str, chat_id: str):
        """Switch to a different bot token / chat (e.g. paper vs live)."""
        self.token = token
        self.chat_id = chat_id
        self.base_url = f"https://api.telegram.org/bot{self.token}"
        # Reset session so next request uses new token
        if self._session and not self._session.closed:
            asyncio.get_event_loop().create_task(self._session.close())
        self._session = None
        logger.info(f"Telegram reconfigured: token=...{token[-6:]}, chat={chat_id}")

    def _fmt_price(self, symbol: str, price: float) -> str:
        """Per-symbol price formatter. FET 4 decimals, others 2."""
        d = PRICE_DECIMALS.get(symbol, 2)
        return f"${price:,.{d}f}"

    async def _refresh_exchange_equity(self):
        """Pull authoritative equity from exchange (MEXC/Binance)."""
        if self.orchestrator is None or getattr(self.orchestrator, "execution", None) is None:
            return
        try:
            from config.settings import settings as _settings
            acc = await asyncio.wait_for(
                self.orchestrator.execution.get_account_info(), timeout=10
            )
            eq = None
            if _settings.exchange.is_mexc:
                for asset in (acc.get("data") or []):
                    if asset.get("currency") == "USDT":
                        eq = float(asset.get("equity", 0))
                        break
            else:
                eq = float(acc.get("totalWalletBalance", 0))
            if eq and eq > 0:
                self.exchange_equity = eq
                self._last_equity_sync = time.time()
        except Exception as e:
            logger.debug(f"Exchange equity sync failed: {e}")

    def _display_equity(self) -> float:
        """Prefer exchange-synced equity (< 5 min old); fallback to MTM calc."""
        from notifications.trade_logger import trade_logger
        if self.exchange_equity and (time.time() - self._last_equity_sync < _EQUITY_SYNC_STALE):
            return self.exchange_equity
        return trade_logger.get_mark_to_market_equity(self.last_prices)

    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create aiohttp session (thread-safe with lock)."""
        async with self._session_lock:
            if self._session is None or self._session.closed:
                timeout = aiohttp.ClientTimeout(total=_POLL_TIMEOUT + 10, sock_read=_POLL_TIMEOUT + 5)
                self._session = aiohttp.ClientSession(timeout=timeout)
            return self._session

    async def close(self):
        """Gracefully shut down all tasks and sessions."""
        # Cancel polling
        if self._poll_task and not self._poll_task.done():
            self._poll_task.cancel()
            try:
                await self._poll_task
            except asyncio.CancelledError:
                pass
        self._poll_task = None

        # Cancel periodic reports
        if self._periodic_task and not self._periodic_task.done():
            self._periodic_task.cancel()
            try:
                await self._periodic_task
            except asyncio.CancelledError:
                pass
        self._periodic_task = None

        # Close HTTP session
        async with self._session_lock:
            if self._session and not self._session.closed:
                await self._session.close()
            self._session = None

    # ── Send messages ──

    async def send(self, text: str, parse_mode: str = "HTML"):
        """Send a message to the configured chat."""
        if not self.enabled:
            logger.debug(f"Telegram disabled, skipping: {text[:50]}...")
            return

        try:
            session = await self._get_session()
            # Use shorter timeout for send (not long-poll)
            send_timeout = aiohttp.ClientTimeout(total=15)
            async with session.post(
                f"{self.base_url}/sendMessage",
                json={
                    "chat_id": self.chat_id,
                    "text": text,
                    "parse_mode": parse_mode,
                    "disable_web_page_preview": True,
                },
                timeout=send_timeout,
            ) as resp:
                if resp.status != 200:
                    body = await resp.text()
                    logger.error(f"Telegram send failed: {resp.status} {body}")
        except Exception as e:
            logger.error(f"Telegram send error: {e}")

    # ── Trade notifications (with dedup) ──

    async def notify_entry(self, symbol: str, side: str, price: float,
                           quantity: float, sl: float, reason: str = ""):
        """Notify when a position is opened (deduplicated)."""
        # Dedup key: symbol + side + price (rounded)
        dedup_key = f"{symbol}_{side}_{price:.2f}"
        if dedup_key in self._notified_entries:
            logger.warning(f"Duplicate entry notification suppressed: {dedup_key}")
            return
        self._notified_entries.add(dedup_key)
        # Keep set bounded
        if len(self._notified_entries) > 50:
            self._notified_entries.clear()

        side_emoji = "🟢" if side == "long" else "🔴"
        sl_pct = abs(price - sl) / price * 100 if price > 0 else 0
        notional = price * quantity

        text = (
            f"{side_emoji} <b>OPEN {side.upper()}</b> {symbol}\n"
            f"━━━━━━━━━━━━━━━\n"
            f"💰 Price: <code>{self._fmt_price(symbol, price)}</code>\n"
            f"📏 Size: <code>{quantity:.6f}</code> (${notional:,.2f})\n"
            f"🛑 SL: <code>{self._fmt_price(symbol, sl)}</code> ({sl_pct:.1f}%)\n"
        )
        if reason:
            text += f"📋 Reason: {reason}\n"
        text += f"🕐 {datetime.now(HKT).strftime('%m/%d %H:%M')} HKT"

        self.positions[symbol] = {
            "side": side, "entry_price": price,
            "quantity": quantity, "sl": sl,
            "entry_time": datetime.now(HKT),
        }
        self.last_prices[symbol] = price

        await self.send(text)

    async def notify_exit(self, symbol: str, side: str, entry_price: float,
                          exit_price: float, quantity: float, pnl: float,
                          pnl_pct: float, exit_reason: str = ""):
        """Notify when a position is closed (deduplicated)."""
        # Dedup key: symbol + entry + exit price
        dedup_key = f"{symbol}_{entry_price:.2f}_{exit_price:.2f}"
        if dedup_key in self._notified_exits:
            logger.warning(f"Duplicate exit notification suppressed: {dedup_key}")
            return
        self._notified_exits.add(dedup_key)
        if len(self._notified_exits) > 50:
            self._notified_exits.clear()

        win = pnl > 0
        emoji = "✅" if win else "❌"
        pnl_emoji = "📈" if win else "📉"

        # Use pnl as-is (commission already deducted by trade_logger)
        self.daily_trades.append({"symbol": symbol, "pnl": pnl, "pnl_pct": pnl_pct})
        self.daily_pnl += pnl
        self.positions.pop(symbol, None)
        self.last_prices[symbol] = exit_price

        from notifications.trade_logger import trade_logger
        mtm_equity = self._display_equity()
        total_ret = (mtm_equity - self.initial_capital) / self.initial_capital * 100 if self.initial_capital > 0 else 0

        text = (
            f"{emoji} <b>CLOSE {side.upper()}</b> {symbol}\n"
            f"━━━━━━━━━━━━━━━\n"
            f"💰 Entry: <code>{self._fmt_price(symbol, entry_price)}</code>\n"
            f"💰 Exit: <code>{self._fmt_price(symbol, exit_price)}</code>\n"
            f"{pnl_emoji} P&L: <code>{'+' if pnl > 0 else ''}${pnl:,.2f}</code> ({'+' if pnl_pct > 0 else ''}{pnl_pct:.1f}%)\n"
            f"📋 Reason: {exit_reason}\n"
            f"━━━━━━━━━━━━━━━\n"
            f"💼 Equity (MTM): <code>${mtm_equity:,.2f}</code> ({'+' if total_ret > 0 else ''}{total_ret:.1f}%)\n"
            f"📊 Today: {len(self.daily_trades)} trades, {'+' if self.daily_pnl >= 0 else ''}${self.daily_pnl:,.2f}\n"
            f"🕐 {datetime.now(HKT).strftime('%m/%d %H:%M')} HKT"
        )

        await self.send(text)

    async def notify_risk_alert(self, level: str, message: str):
        """Send risk alert."""
        emoji = "⚠️" if level == "elevated" else "🚨"
        text = (
            f"{emoji} <b>RISK ALERT — {level.upper()}</b>\n"
            f"━━━━━━━━━━━━━━━\n"
            f"{message}\n"
            f"🕐 {datetime.now(HKT).strftime('%m/%d %H:%M')} HKT"
        )
        await self.send(text)

    # ── Periodic reports (with debounce) ──

    async def send_position_report(self):
        """Send current positions summary (debounced: min 55s interval)."""
        now = time.monotonic()
        if now - self._last_report_time < _MIN_REPORT_INTERVAL:
            logger.debug(f"Position report debounced ({now - self._last_report_time:.0f}s < {_MIN_REPORT_INTERVAL}s)")
            return
        self._last_report_time = now

        if not self.positions:
            text = "📊 <b>Positions Report</b>\n━━━━━━━━━━━━━━━\n💤 No open positions"
        else:
            text = f"📊 <b>Positions Report</b> ({len(self.positions)} open)\n━━━━━━━━━━━━━━━\n"
            for sym, pos in self.positions.items():
                side_emoji = "🟢" if pos["side"] == "long" else "🔴"
                entry_t = pos["entry_time"]
                if isinstance(entry_t, str):
                    entry_t = datetime.fromisoformat(entry_t)
                held = datetime.now(HKT) - entry_t
                held_hours = held.total_seconds() / 3600
                qty = pos.get('quantity', 0)
                notional = qty * pos['entry_price'] if qty else 0
                cur_price = self.last_prices.get(sym, pos['entry_price'])
                if pos['side'] == 'long':
                    upnl = (cur_price - pos['entry_price']) * qty
                else:
                    upnl = (pos['entry_price'] - cur_price) * qty
                upnl_pct = (cur_price / pos['entry_price'] - 1) * 100 if pos['entry_price'] > 0 else 0
                if pos['side'] == 'short':
                    upnl_pct = -upnl_pct
                upnl_emoji = "📈" if upnl >= 0 else "📉"
                text += (
                    f"\n{side_emoji} <b>{sym}</b> {pos['side'].upper()}\n"
                    f"   Entry: {self._fmt_price(sym, pos['entry_price'])} | SL: {self._fmt_price(sym, pos['sl'])}\n"
                    f"   Now: {self._fmt_price(sym, cur_price)} {upnl_emoji} {'+' if upnl >= 0 else ''}${upnl:,.2f} ({'+' if upnl_pct >= 0 else ''}{upnl_pct:.1f}%)\n"
                    f"   Qty: {qty:.6f} (${notional:,.2f})\n"
                    f"   Held: {held_hours:.0f}h\n"
                )

        from notifications.trade_logger import trade_logger
        mtm_equity = self._display_equity()
        total_upnl = trade_logger.get_unrealized_pnl(self.last_prices)
        if mtm_equity > 0:
            ret = (mtm_equity - self.initial_capital) / self.initial_capital * 100 if self.initial_capital > 0 else 0
            upnl_emoji = "📈" if total_upnl >= 0 else "📉"
            text += (
                f"\n{upnl_emoji} Total Unrealized: <code>{'+' if total_upnl >= 0 else ''}${total_upnl:,.2f}</code>"
                f"\n💼 Equity (MTM): <code>${mtm_equity:,.2f}</code> ({'+' if ret > 0 else ''}{ret:.1f}%)"
            )

        text += f"\n🕐 {datetime.now(HKT).strftime('%m/%d %H:%M')} HKT"
        await self.send(text)

    async def send_daily_summary(self):
        """Send end-of-day summary (debounced: max once per 10min)."""
        now = time.monotonic()
        if now - self._last_summary_time < 600:
            logger.debug("Daily summary debounced")
            return
        self._last_summary_time = now

        n_trades = len(self.daily_trades)
        wins = sum(1 for t in self.daily_trades if t["pnl"] > 0)
        wr = wins / n_trades * 100 if n_trades > 0 else 0
        emoji = "📈" if self.daily_pnl >= 0 else "📉"

        text = (
            f"📅 <b>Daily Summary</b> — {datetime.now(HKT).strftime('%Y-%m-%d')}\n"
            f"━━━━━━━━━━━━━━━\n"
            f"{emoji} P&L: <code>{'+' if self.daily_pnl >= 0 else ''}${self.daily_pnl:,.2f}</code>\n"
            f"📊 Trades: {n_trades} | Wins: {wins} | WR: {wr:.0f}%\n"
        )

        from notifications.trade_logger import trade_logger
        mtm_equity = self._display_equity()
        if mtm_equity > 0:
            ret = (mtm_equity - self.initial_capital) / self.initial_capital * 100 if self.initial_capital > 0 else 0
            upnl = trade_logger.get_unrealized_pnl(self.last_prices)
            text += f"💼 Equity (MTM): ${mtm_equity:,.2f} ({'+' if ret > 0 else ''}{ret:.1f}%)\n"
            if upnl != 0:
                text += f"📊 Unrealized: {'+' if upnl >= 0 else ''}${upnl:,.2f}\n"

        open_pos = len(self.positions)
        text += f"📋 Open positions: {open_pos}\n"
        text += f"🕐 {datetime.now(HKT).strftime('%H:%M')} HKT"

        await self.send(text)

        # Reset daily counters
        self.daily_trades = []
        self.daily_pnl = 0.0

    # ── Polling system (production-grade) ──

    async def start_polling(self):
        """Start polling for Telegram commands (atomic guard, flush stale updates)."""
        if not self.enabled:
            return

        async with self._poll_lock:
            # Guard: don't start if already running
            if self._poll_task is not None and not self._poll_task.done():
                logger.warning("Telegram polling already running, skipping duplicate start")
                return

            # Flush stale updates from before restart
            await self._flush_stale_updates()

            self._poll_task = asyncio.create_task(self._poll_loop())
            logger.info("Telegram command polling started")

    async def _flush_stale_updates(self):
        """Discard any pending updates from before this boot (prevents processing old commands)."""
        try:
            session = await self._get_session()
            flush_timeout = aiohttp.ClientTimeout(total=10)
            async with session.get(
                f"{self.base_url}/getUpdates",
                params={"offset": -1, "timeout": 0},
                timeout=flush_timeout,
            ) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    results = data.get("result", [])
                    if results:
                        # Set offset past the last stale update
                        self._last_update_id = results[-1]["update_id"]
                        logger.info(f"Flushed {len(results)} stale update(s), offset now {self._last_update_id}")
                    else:
                        logger.info("No stale updates to flush")
        except Exception as e:
            logger.warning(f"Failed to flush stale updates: {e}")

    async def _poll_loop(self):
        """Long-poll for new Telegram messages (single instance, deduped)."""
        while True:
            try:
                session = await self._get_session()
                poll_timeout = aiohttp.ClientTimeout(total=_POLL_TIMEOUT, sock_read=_POLL_TIMEOUT)
                async with session.get(
                    f"{self.base_url}/getUpdates",
                    params={"offset": self._last_update_id + 1, "timeout": 30},
                    timeout=poll_timeout,
                ) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        for update in data.get("result", []):
                            uid = update["update_id"]
                            # Always advance offset (even if deduped)
                            self._last_update_id = uid

                            # Dedup: skip already processed (deque is O(1) append, bounded)
                            if uid in self._processed_updates:
                                continue
                            self._processed_updates.append(uid)

                            msg = update.get("message", {})
                            text = msg.get("text", "").strip()
                            chat_id = str(msg.get("chat", {}).get("id", ""))

                            # Only respond to our chat
                            if chat_id == self.chat_id and text.startswith("/"):
                                await self._dispatch_command(text)
                    elif resp.status == 409:
                        # Conflict: another bot instance is polling
                        logger.error("Telegram 409 Conflict — another polling instance detected!")
                        await asyncio.sleep(10)
                    else:
                        logger.warning(f"Telegram getUpdates: HTTP {resp.status}")
                        await asyncio.sleep(3)

            except asyncio.CancelledError:
                logger.info("Telegram polling cancelled")
                break
            except asyncio.TimeoutError:
                # Long-poll timeout is normal, just retry
                continue
            except aiohttp.ClientError as e:
                logger.warning(f"Telegram network error: {e}")
                await asyncio.sleep(5)
            except Exception as e:
                logger.error(f"Telegram poll error: {e}", exc_info=True)
                await asyncio.sleep(5)

    async def _dispatch_command(self, text: str):
        """Parse, throttle, and execute a command with concurrency lock."""
        # Strip @BotName suffix: "/checksignal@MyBot" → "/checksignal"
        cmd = text.split()[0].lower()
        if "@" in cmd:
            cmd = cmd.split("@")[0]

        handler = self._commands.get(cmd)
        if not handler:
            await self.send(f"Unknown command: {cmd}\nUse /help for available commands.")
            return

        # Throttle: prevent rapid-fire of same command
        now = time.monotonic()
        last_time = self._last_cmd_time.get(cmd, 0)
        if now - last_time < _MIN_COMMAND_INTERVAL:
            logger.info(f"Command {cmd} throttled ({now - last_time:.1f}s < {_MIN_COMMAND_INTERVAL}s)")
            return
        self._last_cmd_time[cmd] = now

        # Execute with lock (prevent concurrent command execution)
        if self._cmd_lock.locked():
            logger.info(f"Command {cmd} skipped — another command is executing")
            return

        async with self._cmd_lock:
            try:
                await handler()
            except Exception as e:
                logger.error(f"Command {cmd} error: {e}", exc_info=True)
                await self.send(f"❌ Command error: {e}")

    # ── Command handlers ──

    async def _cmd_status(self):
        """Handle /status command."""
        from notifications.trade_logger import trade_logger
        open_pos = len(self.positions)
        mtm_equity = self._display_equity()
        upnl = trade_logger.get_unrealized_pnl(self.last_prices)
        ret = (mtm_equity - self.initial_capital) / self.initial_capital * 100 if mtm_equity > 0 and self.initial_capital > 0 else 0
        today_trades = len(self.daily_trades)
        upnl_emoji = "📈" if upnl >= 0 else "📉"

        text = (
            f"🤖 <b>System Status</b>\n"
            f"━━━━━━━━━━━━━━━\n"
            f"💼 Equity: <code>${mtm_equity:,.2f}</code> ({'+' if ret > 0 else ''}{ret:.1f}%)\n"
            f"{upnl_emoji} Unrealized: <code>{'+' if upnl >= 0 else ''}${upnl:,.2f}</code>\n"
            f"📋 Open: {open_pos} positions\n"
            f"📊 Today: {today_trades} trades, {'+' if self.daily_pnl >= 0 else ''}${self.daily_pnl:,.2f}\n"
            f"🕐 {datetime.now(HKT).strftime('%m/%d %H:%M')} HKT"
        )
        await self.send(text)

    async def _cmd_positions(self):
        """Handle /positions command."""
        await self.send_position_report()

    async def _cmd_pnl(self):
        """Handle /pnl command."""
        await self.send_daily_summary()

    async def _cmd_pause(self):
        """Handle /pause — manually pause all trading."""
        if self.orchestrator:
            from models.messages import AgentMessage, MessageType
            await self.orchestrator.bus.publish(AgentMessage(
                sender="telegram",
                receiver="risk_manager",
                msg_type=MessageType.COMMAND,
                payload={"action": "pause_trading"},
            ))
            await self.send(
                "⏸️ <b>交易已暫停</b>\n"
                "━━━━━━━━━━━━━━━\n"
                "所有新交易已停止\n"
                "現有持倉不受影響\n"
                "👉 /resume 恢復交易"
            )
        else:
            await self.send("⚠️ Orchestrator not connected.")

    async def _cmd_resume(self):
        """Handle /resume — resume trading after extreme news pause."""
        if self.orchestrator:
            from models.messages import AgentMessage, MessageType
            await self.orchestrator.bus.publish(AgentMessage(
                sender="telegram",
                receiver="risk_manager",
                msg_type=MessageType.COMMAND,
                payload={"action": "resume_trading"},
            ))
            await self.send("▶️ <b>交易已恢復</b>")
        else:
            await self.send("⚠️ Orchestrator not connected. Cannot send resume command.")

    async def _cmd_unblock_longs(self):
        """Handle /unblock_longs — allow longs again after major negative news."""
        if self.orchestrator:
            from models.messages import AgentMessage, MessageType
            await self.orchestrator.bus.publish(AgentMessage(
                sender="telegram",
                receiver="risk_manager",
                msg_type=MessageType.COMMAND,
                payload={"action": "unblock_longs"},
            ))
            await self.send("🟢 <b>Longs unblocked</b>")
        else:
            await self.send("⚠️ Orchestrator not connected. Cannot send command.")

    async def _cmd_optimize(self):
        """Handle /optimize — manually trigger parameter re-optimization."""
        if self.orchestrator:
            await self.send("🔧 Starting parameter re-optimization...")
            from models.messages import AgentMessage, MessageType
            await self.orchestrator.bus.publish(AgentMessage(
                sender="telegram",
                receiver="signal_engineer",
                msg_type=MessageType.COMMAND,
                payload={"action": "reoptimize"},
            ))
        else:
            await self.send("⚠️ Orchestrator not connected.")

    async def _cmd_signal_on(self):
        """Handle /signal_on — enable debug signal notifications."""
        from config.settings import settings
        settings.trading.debug_signal_notify = True
        await self.send("🔔 <b>Signal debug notifications ON</b>\n入場/出場信號將會發送 TG 通知")

    async def _cmd_signal_off(self):
        """Handle /signal_off — disable debug signal notifications."""
        from config.settings import settings
        settings.trading.debug_signal_notify = False
        await self.send("🔕 <b>Signal debug notifications OFF</b>\n信號通知已關閉")

    async def _cmd_checksignal(self):
        """Handle /checksignal — show last signals for ETH + SOL (DT 5m strategy)."""
        try:
            from data.historical_fetcher import load_data
            from strategies.fvg_trend import FVGTrendStrategy
            from config.settings import settings
            from notifications.trade_logger import trade_logger
            import pandas as pd

            cfg = settings.strategy
            strategy = FVGTrendStrategy(
                daily_ema_fast=cfg.daily_ema_fast,
                daily_ema_slow=cfg.daily_ema_slow,
                fvg_min_size_pct=cfg.fvg_min_size_pct,
                entry_min_score=cfg.entry_min_score,
                sl_atr=cfg.sl_atr,
                fvg_max_age=cfg.fvg_max_age,
                max_active_fvgs=cfg.max_active_fvgs,
                trail_start_atr=cfg.trail_start_atr,
                trail_atr=cfg.trail_atr,
                breakeven_atr=cfg.breakeven_atr,
                cooldown=cfg.cooldown,
                max_hold=cfg.max_hold,
                vol_mult=cfg.vol_mult,
                min_bars_for_trend_exit=cfg.min_bars_for_trend_exit,
            )

            symbols = settings.trading.symbols  # ["ETHUSDT", "SOLUSDT"]
            alloc = {"ETHUSDT": "40%", "SOLUSDT": "30%", "FETUSDT": "30%"}
            lines = ["📡 <b>Signal Check</b> — ETH+SOL+FET DT 5m\n━━━━━━━━━━━━━━━"]

            for sym in symbols:
                df = load_data(sym, "5m")
                if df is None or len(df) < 100:
                    lines.append(f"\n⚠️ <b>{sym}</b> — 5m 數據不足")
                    continue

                sig_df = strategy.generate_signals(df.copy())
                if sig_df is None or len(sig_df) < 2:
                    lines.append(f"\n⚠️ <b>{sym}</b> — 信號生成失敗")
                    continue

                latest = sig_df.iloc[-1]
                price = float(latest["close"])
                trend = int(latest.get("daily_trend", 0))
                atr = float(latest.get("atr", 0))
                rsi = float(latest.get("rsi", 0)) if not pd.isna(latest.get("rsi", float("nan"))) else 0

                trend_str = "🟢 上升" if trend == 1 else ("🔴 下降" if trend == -1 else "⚪ 中性")
                atr_pct = atr / price * 100 if price > 0 else 0

                # ── Find last 3 entry signals (signal != 0) ──
                entry_mask = sig_df["signal"] != 0
                last_entries = []
                if entry_mask.any():
                    recent_entries = sig_df[entry_mask].tail(3)
                    for idx, row in recent_entries.iterrows():
                        entry_sig = int(row["signal"])
                        entry_side = "LONG 🟢" if entry_sig == 1 else "SHORT 🔴"
                        entry_price = float(row["close"])
                        entry_sl = float(row.get("stop_loss", 0))
                        entry_time = idx
                        if hasattr(entry_time, 'strftime'):
                            entry_time_str = entry_time.strftime("%m/%d %H:%M")
                        else:
                            entry_time_str = str(entry_time)[:16]
                        sl_dist = abs(entry_price - entry_sl) / entry_price * 100 if entry_price > 0 and entry_sl > 0 else 0
                        last_entries.append(f"  {entry_side} @ {self._fmt_price(sym, entry_price)} | SL {self._fmt_price(sym, entry_sl)} ({sl_dist:.1f}%) | {entry_time_str}")

                entries_str = "\n".join(last_entries) if last_entries else "  無記錄"

                # ── Find last exit: check trade_logger for closed trades ──
                last_exit_str = "無記錄"
                closed = [t for t in trade_logger.trades if t.get("symbol") == sym]
                if closed:
                    last_trade = closed[-1]
                    exit_reason_map = {
                        "trend_reversal": "趨勢反轉",
                        "stop_loss": "止損",
                        "trail_stop": "追蹤止損",
                        "timeout": "超時",
                        "sl": "止損",
                        "trend_rev": "趨勢反轉",
                    }
                    reason = exit_reason_map.get(last_trade.get("exit_reason", ""), last_trade.get("exit_reason", ""))
                    pnl = last_trade.get("net_pnl", last_trade.get("pnl", 0))
                    pnl_emoji = "✅" if pnl > 0 else "❌"
                    exit_time = last_trade.get("exit_time", "")
                    if isinstance(exit_time, str) and len(exit_time) >= 16:
                        exit_time_str = exit_time[5:16].replace("T", " ")
                    else:
                        exit_time_str = str(exit_time)[:16]
                    last_exit_str = f"{pnl_emoji} {reason} @ {self._fmt_price(sym, last_trade.get('exit_price', 0))} | PnL {'+' if pnl > 0 else ''}${pnl:,.2f} | {exit_time_str}"

                # ── Current position status ──
                pos = trade_logger.positions.get(sym)
                if pos:
                    pos_side = "🟢 LONG" if pos["side"] == "long" else "🔴 SHORT"
                    cur_price = self.last_prices.get(sym, price)
                    if pos["side"] == "long":
                        upnl = (cur_price - pos["entry_price"]) * pos["quantity"]
                    else:
                        upnl = (pos["entry_price"] - cur_price) * pos["quantity"]
                    pos_str = f"{pos_side} @ {self._fmt_price(sym, pos['entry_price'])} | uPnL {'+' if upnl >= 0 else ''}${upnl:,.2f}"
                else:
                    pos_str = "💤 無持倉"

                lines.append(
                    f"\n<b>{'─' * 15}</b>\n"
                    f"📊 <b>{sym}</b> ({alloc.get(sym, '')}) — {self._fmt_price(sym, price)}\n"
                    f"趨勢: {trend_str} | ATR: {atr_pct:.2f}% | RSI: {rsi:.0f}\n"
                    f"🔹 持倉: {pos_str}\n"
                    f"🔸 最近入場信號 (×3):\n{entries_str}\n"
                    f"🔸 上次離場: {last_exit_str}"
                )

            lines.append(f"\n🕐 {datetime.now(HKT).strftime('%m/%d %H:%M')} HKT")
            await self.send("\n".join(lines))

        except Exception as e:
            logger.error(f"checksignal error: {e}", exc_info=True)
            await self.send(f"❌ 信號檢查失敗: {e}")

    async def _cmd_validate(self):
        """Handle /validate — manually trigger monthly validation pipeline."""
        if self.orchestrator:
            await self.send("🔬 Starting monthly validation pipeline...\nThis may take 10-15 minutes.")
            await self.orchestrator.trigger_monthly_validation()
        else:
            await self.send("⚠️ Orchestrator not connected.")

    async def _cmd_help(self):
        """Handle /help command."""
        from config.settings import settings
        sig_status = "ON 🔔" if settings.trading.debug_signal_notify else "OFF 🔕"
        text = (
            "🤖 <b>FVG Trading Bot Commands</b>\n"
            "━━━━━━━━━━━━━━━\n"
            "/status — System status & equity\n"
            "/positions — Current open positions\n"
            "/pnl — Today's P&L summary\n"
            "/checksignal — ETH+SOL+FET 最新 5m 信號\n"
            "/pause — Pause all trading\n"
            "/resume — Resume trading\n"
            "/optimize — Re-optimize parameters\n"
            "/validate — Run monthly validation suite\n"
            "/unblock_longs — Allow longs again\n"
            f"/signal_on — Enable signal alerts ({sig_status})\n"
            "/signal_off — Disable signal alerts\n"
            "/help — Show this message"
        )
        await self.send(text)

    # ── Periodic scheduler (integrated into class) ──

    async def start_periodic_reports(self):
        """Start periodic report task (atomic guard, handles restart)."""
        async with self._periodic_lock:
            if self._periodic_task is not None and not self._periodic_task.done():
                logger.warning("Periodic reports already running, skipping duplicate start")
                return
            self._periodic_task = asyncio.create_task(self._periodic_loop())
            logger.info("Periodic reports started")

    async def _periodic_loop(self):
        """Hourly position reports + daily summary (single instance)."""
        last_report_hour = datetime.now(HKT).hour  # Skip immediate trigger on startup
        last_summary_date = None
        while True:
            try:
                await asyncio.sleep(30)
                now = datetime.now(HKT)

                # Refresh authoritative exchange equity (cached, used in all displays)
                await self._refresh_exchange_equity()

                # Hourly position report (only if positions open, trigger on hour change)
                if now.hour != last_report_hour and self.positions:
                    await self.send_position_report()
                    last_report_hour = now.hour
                elif now.hour != last_report_hour:
                    last_report_hour = now.hour  # Update even if no positions

                # Daily summary at 08:00 HKT (once per day)
                today = now.date()
                if now.hour >= 8 and last_summary_date != today:
                    await self.send_daily_summary()
                    last_summary_date = today

            except asyncio.CancelledError:
                logger.info("Periodic reports cancelled")
                break
            except Exception as e:
                logger.error(f"Periodic report error: {e}", exc_info=True)
                await asyncio.sleep(30)


# ── Legacy wrapper for backward compatibility ──

async def run_periodic_reports(notifier: TelegramNotifier):
    """Legacy wrapper — delegates to notifier.start_periodic_reports()."""
    await notifier.start_periodic_reports()


# Global singleton
notifier = TelegramNotifier()
