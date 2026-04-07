"""
Telegram 通知模組 — 交易信號、持倉更新、風險警報

功能:
- 開倉 / 平倉即時通知
- 定期持倉報告 (每小時)
- 每日績效摘要 (每天 UTC+8 08:00)
- 風險警報 (MDD、異常等)
- 可通過 Telegram 查詢指令 (/status, /positions, /pnl)
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
from datetime import datetime, timezone, timedelta
from typing import Optional

import aiohttp

logger = logging.getLogger("telegram")

HKT = timezone(timedelta(hours=8))


class TelegramNotifier:
    """Telegram Bot 通知器"""

    def __init__(self, token: str = "", chat_id: str = ""):
        self.token = token or os.getenv("TELEGRAM_BOT_TOKEN", "")
        self.chat_id = chat_id or os.getenv("TELEGRAM_CHAT_ID", "")
        self.base_url = f"https://api.telegram.org/bot{self.token}"
        self._session: Optional[aiohttp.ClientSession] = None
        self._poll_task: Optional[asyncio.Task] = None
        self._last_update_id = 0

        # State tracking
        self.positions: dict = {}       # symbol -> position info
        self.daily_trades: list = []    # today's trades
        self.daily_pnl: float = 0.0
        self.equity: float = 0.0
        self.initial_capital: float = 2000.0

        # Command handlers
        self._commands = {
            "/status": self._cmd_status,
            "/positions": self._cmd_positions,
            "/pnl": self._cmd_pnl,
            "/help": self._cmd_help,
        }

    @property
    def enabled(self) -> bool:
        return bool(self.token and self.chat_id)

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession()
        return self._session

    async def close(self):
        if self._poll_task:
            self._poll_task.cancel()
        if self._session and not self._session.closed:
            await self._session.close()

    # ── Send messages ──

    async def send(self, text: str, parse_mode: str = "HTML"):
        """Send a message to the configured chat."""
        if not self.enabled:
            logger.debug(f"Telegram disabled, skipping: {text[:50]}...")
            return

        try:
            session = await self._get_session()
            async with session.post(
                f"{self.base_url}/sendMessage",
                json={
                    "chat_id": self.chat_id,
                    "text": text,
                    "parse_mode": parse_mode,
                    "disable_web_page_preview": True,
                },
            ) as resp:
                if resp.status != 200:
                    body = await resp.text()
                    logger.error(f"Telegram send failed: {resp.status} {body}")
        except Exception as e:
            logger.error(f"Telegram send error: {e}")

    # ── Trade notifications ──

    async def notify_entry(self, symbol: str, side: str, price: float,
                           quantity: float, sl: float, reason: str = ""):
        """Notify when a position is opened."""
        side_emoji = "🟢" if side == "long" else "🔴"
        sl_pct = abs(price - sl) / price * 100
        notional = price * quantity

        text = (
            f"{side_emoji} <b>OPEN {side.upper()}</b> {symbol}\n"
            f"━━━━━━━━━━━━━━━\n"
            f"💰 Price: <code>${price:,.2f}</code>\n"
            f"📏 Size: <code>{quantity:.6f}</code> (${notional:,.2f})\n"
            f"🛑 SL: <code>${sl:,.2f}</code> ({sl_pct:.1f}%)\n"
        )
        if reason:
            text += f"📋 Reason: {reason}\n"
        text += f"🕐 {datetime.now(HKT).strftime('%m/%d %H:%M')} HKT"

        self.positions[symbol] = {
            "side": side, "entry_price": price,
            "quantity": quantity, "sl": sl,
            "entry_time": datetime.now(HKT),
        }

        await self.send(text)

    async def notify_exit(self, symbol: str, side: str, entry_price: float,
                          exit_price: float, quantity: float, pnl: float,
                          pnl_pct: float, exit_reason: str = ""):
        """Notify when a position is closed."""
        win = pnl > 0
        emoji = "✅" if win else "❌"
        pnl_emoji = "📈" if win else "📉"

        self.daily_trades.append({"symbol": symbol, "pnl": pnl, "pnl_pct": pnl_pct})
        self.daily_pnl += pnl
        self.equity += pnl
        self.positions.pop(symbol, None)

        total_ret = (self.equity - self.initial_capital) / self.initial_capital * 100

        text = (
            f"{emoji} <b>CLOSE {side.upper()}</b> {symbol}\n"
            f"━━━━━━━━━━━━━━━\n"
            f"💰 Entry: <code>${entry_price:,.2f}</code>\n"
            f"💰 Exit: <code>${exit_price:,.2f}</code>\n"
            f"{pnl_emoji} P&L: <code>{'+' if pnl > 0 else ''}${pnl:,.2f}</code> ({'+' if pnl_pct > 0 else ''}{pnl_pct:.1f}%)\n"
            f"📋 Reason: {exit_reason}\n"
            f"━━━━━━━━━━━━━━━\n"
            f"💼 Equity: <code>${self.equity:,.2f}</code> ({'+' if total_ret > 0 else ''}{total_ret:.1f}%)\n"
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

    # ── Periodic reports ──

    async def send_position_report(self):
        """Send current positions summary."""
        if not self.positions:
            text = "📊 <b>Positions Report</b>\n━━━━━━━━━━━━━━━\n💤 No open positions"
        else:
            text = f"📊 <b>Positions Report</b> ({len(self.positions)} open)\n━━━━━━━━━━━━━━━\n"
            for sym, pos in self.positions.items():
                side_emoji = "🟢" if pos["side"] == "long" else "🔴"
                held = datetime.now(HKT) - pos["entry_time"]
                held_hours = held.total_seconds() / 3600
                text += (
                    f"\n{side_emoji} <b>{sym}</b> {pos['side'].upper()}\n"
                    f"   Entry: ${pos['entry_price']:,.2f} | SL: ${pos['sl']:,.2f}\n"
                    f"   Held: {held_hours:.0f}h\n"
                )

        if self.equity > 0:
            ret = (self.equity - self.initial_capital) / self.initial_capital * 100
            text += f"\n💼 Equity: ${self.equity:,.2f} ({'+' if ret > 0 else ''}{ret:.1f}%)"

        text += f"\n🕐 {datetime.now(HKT).strftime('%m/%d %H:%M')} HKT"
        await self.send(text)

    async def send_daily_summary(self):
        """Send end-of-day summary."""
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

        if self.equity > 0:
            ret = (self.equity - self.initial_capital) / self.initial_capital * 100
            text += f"💼 Equity: ${self.equity:,.2f} ({'+' if ret > 0 else ''}{ret:.1f}%)\n"

        open_pos = len(self.positions)
        text += f"📋 Open positions: {open_pos}\n"
        text += f"🕐 {datetime.now(HKT).strftime('%H:%M')} HKT"

        await self.send(text)

        # Reset daily counters
        self.daily_trades = []
        self.daily_pnl = 0.0

    # ── Command handlers (respond to Telegram messages) ──

    async def start_polling(self):
        """Start polling for incoming Telegram commands."""
        if not self.enabled:
            return
        self._poll_task = asyncio.create_task(self._poll_loop())
        logger.info("Telegram command polling started")

    async def _poll_loop(self):
        """Poll for new messages from Telegram."""
        while True:
            try:
                session = await self._get_session()
                async with session.get(
                    f"{self.base_url}/getUpdates",
                    params={"offset": self._last_update_id + 1, "timeout": 30},
                ) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        for update in data.get("result", []):
                            self._last_update_id = update["update_id"]
                            msg = update.get("message", {})
                            text = msg.get("text", "").strip()
                            chat_id = str(msg.get("chat", {}).get("id", ""))

                            # Only respond to our chat
                            if chat_id == self.chat_id and text.startswith("/"):
                                cmd = text.split()[0].lower()
                                handler = self._commands.get(cmd)
                                if handler:
                                    await handler()
                                else:
                                    await self.send(f"Unknown command: {cmd}\nUse /help for available commands.")
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Telegram poll error: {e}")
                await asyncio.sleep(5)

    async def _cmd_status(self):
        """Handle /status command."""
        open_pos = len(self.positions)
        ret = (self.equity - self.initial_capital) / self.initial_capital * 100 if self.equity > 0 else 0
        today_trades = len(self.daily_trades)

        text = (
            f"🤖 <b>System Status</b>\n"
            f"━━━━━━━━━━━━━━━\n"
            f"💼 Equity: ${self.equity:,.2f} ({'+' if ret > 0 else ''}{ret:.1f}%)\n"
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

    async def _cmd_help(self):
        """Handle /help command."""
        text = (
            "🤖 <b>FVG Trading Bot Commands</b>\n"
            "━━━━━━━━━━━━━━━\n"
            "/status — System status & equity\n"
            "/positions — Current open positions\n"
            "/pnl — Today's P&L summary\n"
            "/help — Show this message"
        )
        await self.send(text)


# ── Periodic scheduler ──

async def run_periodic_reports(notifier: TelegramNotifier):
    """Run hourly position reports and daily summaries."""
    while True:
        now = datetime.now(HKT)
        # Hourly position report (only if positions open)
        if now.minute == 0 and notifier.positions:
            await notifier.send_position_report()

        # Daily summary at 08:00 HKT
        if now.hour == 8 and now.minute == 0:
            await notifier.send_daily_summary()

        await asyncio.sleep(60)


# Global singleton
notifier = TelegramNotifier()
