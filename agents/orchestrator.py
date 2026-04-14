"""團隊調度器 — 協調 FVG Trend Follow 交易系統

工作流程:
1. 啟動所有 6 agent
2. 載入歷史數據到信號工程師
3. 部署 FVG Trend Follow Combo3 策略
4. 啟動市場監控 (BTC/ETH/SOL)
5. 信號 → 風控 → 執行 的自動化管線
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from typing import Optional, List

import pandas as pd

from agents.base_agent import BaseAgent, MessageBus
from agents.quant_researcher import QuantResearcher
from agents.backtest_engineer import BacktestEngineer
from agents.risk_manager import RiskManager
from agents.signal_engineer import SignalEngineer
from agents.execution_engineer import ExecutionEngineer
from agents.market_analyst import MarketAnalyst
from data.data_manager import DataManager
from data.historical_fetcher import load_data
from models.messages import AgentMessage, MessageType
from models.risk import RiskLimits
from config.settings import settings
from notifications.telegram_bot import notifier as tg
from notifications.trade_logger import trade_logger


logger = logging.getLogger("orchestrator")


class _OrchestratorProxy(BaseAgent):
    """Lightweight agent that receives VALIDATION_RESULT messages for the orchestrator."""

    def __init__(self):
        super().__init__(agent_id="orchestrator", name="調度器")
        self.owner = None  # set to TeamOrchestrator instance

    async def handle_message(self, message: AgentMessage):
        if message.msg_type == MessageType.VALIDATION_RESULT and self.owner:
            self.owner._on_validation_result(message)


class TeamOrchestrator:
    """
    團隊調度器：
    - 初始化並管理所有 agent
    - 透過 MessageBus 路由消息
    - 整合 DataManager 提供數據
    - 提供高層操作接口
    """

    def __init__(
        self,
        risk_limits: Optional[RiskLimits] = None,
        tv_username: Optional[str] = None,
        tv_password: Optional[str] = None,
        webhook_port: int = 8080,
    ):
        self.bus = MessageBus()
        self.data = DataManager(
            tv_username=tv_username,
            tv_password=tv_password,
            webhook_port=webhook_port,
        )

        # 初始化 6 個角色
        self.quant = QuantResearcher()
        self.backtest = BacktestEngineer()
        self.risk = RiskManager(limits=risk_limits)
        self.signal = SignalEngineer()
        self.execution = ExecutionEngineer()
        self.market = MarketAnalyst()

        # 注入 data_manager 到需要數據的 agent
        self.backtest.data_manager = self.data
        self.market.data_manager = self.data

        # Orchestrator proxy (receives VALIDATION_RESULT messages)
        self._proxy = _OrchestratorProxy()
        self._proxy.owner = self

        self._agents = [
            self.quant,
            self.backtest,
            self.risk,
            self.signal,
            self.execution,
            self.market,
            self._proxy,
        ]

        # 註冊到消息匯流排
        for agent in self._agents:
            self.bus.register(agent)

    async def start(self):
        """啟動所有 agent"""
        logger.info("=" * 60)
        logger.info("FVG Trend Follow 交易系統啟動")
        logger.info(f"  資產: {settings.trading.symbols}")
        logger.info(f"  資金: ${settings.trading.initial_capital:,.0f}")
        logger.info(f"  策略: Combo3 (EMA {settings.strategy.daily_ema_fast}/{settings.strategy.daily_ema_slow}, SL {settings.strategy.sl_atr} ATR)")
        logger.info("=" * 60)
        await self.bus.start_all()
        logger.info("所有 agent 已啟動")

    async def stop(self):
        """停止所有 agent"""
        await self.bus.stop_all()
        await self.data.close()
        logger.info("所有 agent 已停止")

    # ── FVG 交易專用接口 ─────────────────────

    async def start_trading_session(self, paper: bool = True):
        """
        啟動完整交易 session:
        1. 啟動所有 agent
        2. 載入歷史數據到信號工程師
        3. 部署 FVG 策略
        4. 啟動市場監控
        """
        await self.start()

        # 更新並載入每個幣種的歷史 5m 數據
        await self._update_and_load_historical()

        # ── 恢復持倉狀態 ──
        await self._restore_state(paper)

        # 部署 FVG Trend Follow 策略
        await self.bus.publish(
            AgentMessage(
                sender="orchestrator",
                receiver="signal_engineer",
                msg_type=MessageType.COMMAND,
                payload={
                    "action": "deploy_strategy",
                    "strategy": {
                        "strategy_name": "FVG Trend Follow Combo3",
                        "type": "fvg_trend",
                        "symbols": settings.trading.symbols,
                    },
                },
            )
        )

        # ── LIVE: 設定 leverage/margin，開啟交易 ──
        if not paper and settings.exchange.api_key:
            self.execution.state["trading_enabled"] = False  # 確保未就緒前不會執行
            await self._setup_exchange_config()
            # 設定 position mode (Binance only — MEXC uses positionMode in order)
            if not settings.exchange.is_mexc:
                try:
                    params_mode = {"dualSidePosition": "false",
                                   "timestamp": self.execution._get_timestamp()}
                    await self.execution._signed_request("POST", "/fapi/v1/positionSide/dual", params_mode)
                    logger.info("Position mode set to One-way")
                except Exception as e:
                    if "-4059" in str(e):
                        logger.info("Position mode already One-way")
                    else:
                        logger.warning(f"Set position mode failed: {e}")
            # Enable trading BEFORE compute_now so startup signals can execute
            await self.enable_trading()
            logger.info("實盤交易已開啟（leverage + margin 就緒）")
        elif paper:
            logger.info("紙上交易模式")
        else:
            logger.warning("未設定 API Key，維持紙上交易模式")

        # 補齊最新已收盤 K 線（REST API 拉取，確保不漏掉啟動前的信號）
        await self._fetch_latest_closed_candles()

        # 一次性取消 cooldown，讓啟動時的信號不被 cooldown 阻擋
        self.signal._skip_cooldown_once = True
        logger.info("已啟用一次性 cooldown 跳過（僅本次啟動生效）")

        # 用最後一根已收盤 K 線立即計算信號（不用等下一個整點）
        await self.bus.publish(
            AgentMessage(
                sender="orchestrator",
                receiver="signal_engineer",
                msg_type=MessageType.COMMAND,
                payload={"action": "compute_now"},
            )
        )
        # 給 signal_engineer 處理完信號的時間（避免與 WebSocket 數據競爭）
        await asyncio.sleep(2)

        # 啟動市場監控
        await self.start_monitoring()

        # Telegram: switch bot token for paper vs live
        if paper:
            paper_token = os.getenv("TELEGRAM_PAPER_BOT_TOKEN", "")
            paper_chat = os.getenv("TELEGRAM_PAPER_CHAT_ID", os.getenv("TELEGRAM_CHAT_ID", ""))
            if paper_token:
                tg.reconfigure(paper_token, paper_chat)
                logger.info("Telegram switched to PAPER bot")

        # Telegram: startup notification + periodic reports + command polling
        if tg.enabled:
            tg.initial_capital = trade_logger.initial_capital
            tg.equity = trade_logger.equity
            tg.positions = {s: p for s, p in trade_logger.positions.items()}
            # Seed last_prices with entry prices so MTM equity is reasonable before tickers arrive
            for sym, pos in trade_logger.positions.items():
                if pos.get("entry_price", 0) > 0:
                    tg.last_prices[sym] = pos["entry_price"]
            tg.orchestrator = self
            mode = "LIVE" if not paper else "PAPER"
            n_pos = len(trade_logger.positions)
            restored_info = ""
            if n_pos > 0 or trade_logger.equity != trade_logger.initial_capital:
                restored_info = (
                    f"\n🔄 Restored: {n_pos} positions, "
                    f"${trade_logger.equity:,.2f} equity"
                )
            from strategies.param_optimizer import get_current_params, load_optimizer_state
            opt_params = get_current_params()
            opt_state = load_optimizer_state()
            last_opt = opt_state.get("last_optimization", "never")
            await tg.send(
                f"🤖 <b>FVG Trading System Started</b>\n"
                f"━━━━━━━━━━━━━━━\n"
                f"📋 Mode: {mode}\n"
                f"💰 Capital: ${trade_logger.equity:,.2f}\n"
                f"📊 Symbols: {', '.join(settings.trading.symbols)}\n"
                f"⚙️ Strategy: FVG Trend Follow (Dynamic)\n"
                f"   score≥{opt_params.get('entry_min_score',2)} | "
                f"FVG≥{opt_params.get('fvg_min_size_pct',0.10)}% | "
                f"SL {opt_params.get('sl_atr',6.0)} ATR\n"
                f"🔧 Last optimized: {last_opt}"
                f"{restored_info}\n\n"
                f"Commands: /status /positions /pnl /pause /help"
            )
            await tg.start_polling()          # Guarded: won't start twice
            await tg.start_periodic_reports()  # Guarded: won't start twice
            logger.info("Telegram notifications enabled")

        # Periodic CSV sync: buffer → CSV every 10 min (keeps CSV fresh for restarts & analysis)
        asyncio.create_task(self._periodic_csv_sync())

        # Periodic equity sync from exchange (keeps trade_logger.equity accurate)
        asyncio.create_task(self._periodic_equity_sync())

        # Monthly validation pipeline (check every 24h, trigger every 30 days)
        self._validation_pending = {}  # {source: result} for collecting async results
        self._validation_request_id = ""
        asyncio.create_task(self._periodic_monthly_validation())

        # MEXC: refresh plan orders every 20h (executeCycle max = 24h)
        if settings.exchange.is_mexc and settings.trading.trading_enabled:
            asyncio.create_task(self._periodic_plan_order_refresh())

    async def _update_and_load_historical(self):
        """Update CSV data from Binance API, then load into signal engineer."""
        import aiohttp
        from data.historical_fetcher import fetch_klines_page, save_data

        for symbol in settings.trading.symbols:
            df = load_data(symbol, "5m")
            if df is None or len(df) < 500:
                logger.warning(f"{symbol} 無歷史數據，請先執行 run_cross_asset.py 下載")
                continue

            # Check gap: fetch missing bars from Binance
            from datetime import datetime
            last_ts = int(df.index[-1].timestamp() * 1000) + 1
            now_ts = int(datetime.utcnow().timestamp() * 1000)
            gap_minutes = (now_ts - last_ts) / 60000

            if gap_minutes > 7.5:
                logger.info(f"{symbol} 數據落後 {gap_minutes:.0f}min，正在補齊...")
                try:
                    all_new = []
                    current_ts = last_ts
                    async with aiohttp.ClientSession() as session:
                        while current_ts < now_ts:
                            data = await fetch_klines_page(session, symbol, "5m", current_ts, now_ts)
                            if not data:
                                break
                            all_new.extend(data)
                            current_ts = data[-1][6] + 1
                            await asyncio.sleep(0.05)

                    if all_new:
                        import pandas as pd
                        new_df = pd.DataFrame(all_new, columns=[
                            "open_time", "open", "high", "low", "close", "volume",
                            "close_time", "quote_volume", "trades",
                            "taker_buy_volume", "taker_buy_quote_volume", "ignore",
                        ])
                        new_df["datetime"] = pd.to_datetime(new_df["open_time"], unit="ms")
                        for col in ["open", "high", "low", "close", "volume", "quote_volume",
                                    "taker_buy_volume", "taker_buy_quote_volume"]:
                            new_df[col] = new_df[col].astype(float)
                        new_df["trades"] = new_df["trades"].astype(int)
                        new_df = new_df.set_index("datetime").drop(columns=["open_time", "close_time", "ignore"])
                        df = pd.concat([df, new_df])
                        df = df[~df.index.duplicated(keep="first")].sort_index()
                        save_data(df, symbol, "5m")
                        logger.info(f"{symbol} 已補齊 +{len(all_new)} bars -> {len(df)} total")
                except Exception as e:
                    logger.error(f"{symbol} 數據更新失敗: {e}")

            # Load into signal engineer
            await self.bus.publish(
                AgentMessage(
                    sender="orchestrator",
                    receiver="signal_engineer",
                    msg_type=MessageType.COMMAND,
                    payload={
                        "action": "load_historical",
                        "symbol": symbol,
                        "data": df,
                    },
                )
            )
            logger.info(f"已載入 {symbol} 歷史數據: {len(df)} bars")

    async def _fetch_latest_closed_candles(self):
        """Fetch the latest closed 5m candles from exchange REST API for each symbol.
        This ensures compute_now uses the absolute latest data even if CSV is stale."""
        import aiohttp
        import pandas as pd
        import time as _time

        now_s = int(_time.time())  # UTC epoch seconds (always correct)
        start_s = now_s - 25 * 60  # Last 25 minutes (5 candles of 5m)
        current_5m_start = (now_s // 300) * 300  # Start of current (unclosed) 5m candle

        try:
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=10)) as session:
                for symbol in settings.trading.symbols:
                    try:
                        if settings.exchange.is_mexc:
                            mexc_sym = settings.exchange.to_mexc_symbol(symbol)
                            url = f"{settings.exchange.base_url}/api/v1/contract/kline/{mexc_sym}"
                            params = {"interval": "Min5", "start": start_s, "end": now_s}
                            async with session.get(url, params=params) as resp:
                                if resp.status != 200:
                                    continue
                                result = await resp.json()
                                data = result.get("data", {})
                                if not data or not data.get("time"):
                                    continue
                                times = data["time"]
                                added = 0
                                for i in range(len(times)):
                                    if times[i] >= current_5m_start:
                                        continue  # Skip current (unclosed) candle
                                    candle_ts = pd.Timestamp(times[i], unit="s", tz="UTC").tz_localize(None)
                                    buf = self.signal.ohlcv_buffers.get(symbol)
                                    if buf is not None and candle_ts not in buf.index:
                                        new_row = pd.DataFrame({
                                            "open": [float(data["open"][i])],
                                            "high": [float(data["high"][i])],
                                            "low": [float(data["low"][i])],
                                            "close": [float(data["close"][i])],
                                            "volume": [float(data["vol"][i])],
                                        }, index=pd.DatetimeIndex([candle_ts], name="datetime"))
                                        self.signal.ohlcv_buffers[symbol] = pd.concat([buf, new_row]).tail(
                                            self.signal.max_buffer_bars
                                        )
                                        added += 1
                                if added:
                                    logger.info(f"{symbol} 補入 {added} 根 REST 5m K 線")
                        else:
                            from data.historical_fetcher import fetch_klines_page
                            rows = await fetch_klines_page(session, symbol, "5m", start_s * 1000, now_s * 1000, 5)
                            added = 0
                            for row in (rows or []):
                                if row[0] >= current_5m_start * 1000:
                                    continue
                                candle_ts = pd.Timestamp(row[0], unit="ms").tz_localize(None)
                                buf = self.signal.ohlcv_buffers.get(symbol)
                                if buf is not None and candle_ts not in buf.index:
                                    new_row = pd.DataFrame({
                                        "open": [float(row[1])], "high": [float(row[2])],
                                        "low": [float(row[3])], "close": [float(row[4])],
                                        "volume": [float(row[5])],
                                    }, index=pd.DatetimeIndex([candle_ts], name="datetime"))
                                    self.signal.ohlcv_buffers[symbol] = pd.concat([buf, new_row]).tail(
                                        self.signal.max_buffer_bars
                                    )
                                    added += 1
                            if added:
                                logger.info(f"{symbol} 補入 {added} 根 REST 5m K 線")
                    except Exception as e:
                        logger.warning(f"{symbol} REST K 線補齊失敗: {e}")
        except Exception as e:
            logger.warning(f"REST K 線補齊整體失敗: {e}")

    async def _restore_state(self, paper: bool):
        """重啟後恢復持倉狀態到所有 agent"""
        positions = {}
        equity = settings.trading.initial_capital

        if paper:
            # Paper trading: 從 live_portfolio.json 恢復
            if trade_logger.positions:
                positions = trade_logger.positions
                equity = trade_logger.equity
                logger.info(
                    f"從 JSON 恢復 {len(positions)} 個持倉 | "
                    f"權益 ${equity:,.2f}"
                )
            else:
                equity = trade_logger.equity  # 可能無持倉但有歷史權益
                if equity != trade_logger.initial_capital:
                    logger.info(f"從 JSON 恢復權益 ${equity:,.2f}（無持倉）")
                return
        else:
            # Live trading: 從交易所 API 查詢實際持倉
            exchange_name = "MEXC" if settings.exchange.is_mexc else "Binance"
            try:
                account = await asyncio.wait_for(
                    self.execution.get_account_info(), timeout=15
                )
                if account:
                    if settings.exchange.is_mexc:
                        # MEXC: account returns {"success":true, "data":[{currency, equity, ...}]}
                        assets = account.get("data", []) if isinstance(account.get("data"), list) else []
                        for asset in assets:
                            if asset.get("currency") == "USDT":
                                equity = float(asset.get("equity", equity))
                                break
                        # Fetch positions separately
                        open_pos = await asyncio.wait_for(
                            self.execution.get_open_positions(), timeout=15
                        )
                        for pos_data in open_pos:
                            mexc_sym = pos_data.get("symbol", "")
                            sym = settings.exchange.from_mexc_symbol(mexc_sym)
                            amt = float(pos_data.get("holdVol", 0))
                            if sym in settings.trading.symbols and amt != 0:
                                entry = float(pos_data.get("openAvgPrice", 0))
                                pos_type = pos_data.get("positionType", 1)  # 1=long, 2=short
                                side = "long" if pos_type == 1 else "short"
                                local = trade_logger.positions.get(sym, {})
                                local_sl = local.get("sl") or 0
                                if local_sl <= 0 and entry > 0:
                                    from data.historical_fetcher import load_data
                                    df_hist = load_data(sym, "5m")
                                    if df_hist is not None and len(df_hist) > 14:
                                        atr_vals = (df_hist["high"].tail(14) - df_hist["low"].tail(14))
                                        atr_est = float(atr_vals.mean())
                                        local_sl = entry - atr_est * settings.strategy.sl_atr if side == "long" else entry + atr_est * settings.strategy.sl_atr
                                        logger.warning(f"{sym} SL 從 ATR 估算: {local_sl:.4f}")
                                # Convert contract vol to coin qty for internal tracking
                                contract_size = self.execution.MEXC_CONTRACT_SIZE.get(sym, 0.0001)
                                coin_qty = amt * contract_size
                                positions[sym] = {
                                    "side": side, "entry_price": entry,
                                    "quantity": coin_qty, "notional": coin_qty * entry,
                                    "sl": local_sl, "best_price": local.get("best_price", entry),
                                    "trail_sl": local.get("trail_sl"), "entry_time": local.get("entry_time", ""),
                                }
                    else:
                        # Binance
                        equity = float(account.get("totalWalletBalance", equity))
                        for pos_data in account.get("positions", []):
                            sym = pos_data.get("symbol", "")
                            amt = float(pos_data.get("positionAmt", 0))
                            if sym in settings.trading.symbols and amt != 0:
                                entry = float(pos_data.get("entryPrice", 0))
                                local = trade_logger.positions.get(sym, {})
                                local_sl = local.get("sl") or 0
                                if local_sl <= 0 and entry > 0:
                                    from data.historical_fetcher import load_data
                                    df_hist = load_data(sym, "5m")
                                    if df_hist is not None and len(df_hist) > 14:
                                        atr_vals = (df_hist["high"].tail(14) - df_hist["low"].tail(14))
                                        atr_est = float(atr_vals.mean())
                                        if amt > 0:
                                            local_sl = entry - atr_est * settings.strategy.sl_atr
                                        else:
                                            local_sl = entry + atr_est * settings.strategy.sl_atr
                                        logger.warning(f"{sym} SL 從 ATR 估算: {local_sl:.4f}")
                                positions[sym] = {
                                    "side": "long" if amt > 0 else "short",
                                    "entry_price": entry, "quantity": abs(amt),
                                    "notional": abs(amt) * entry, "sl": local_sl,
                                    "best_price": local.get("best_price", entry),
                                    "trail_sl": local.get("trail_sl"), "entry_time": local.get("entry_time", ""),
                                }
                    # Sync trade_logger with exchange truth
                    trade_logger.equity = equity
                    trade_logger.positions = positions
                    trade_logger._save()
                    logger.info(
                        f"從 {exchange_name} 恢復 {len(positions)} 個持倉 | "
                        f"權益 ${equity:,.2f}"
                    )
            except Exception as e:
                logger.warning(f"Binance 持倉查詢失敗: {e}，使用本地數據")
                positions = trade_logger.positions
                equity = trade_logger.equity

        if not positions:
            return

        # Filter: only restore positions for currently active symbols
        active_symbols = set(settings.trading.symbols)
        removed = {s for s in positions if s not in active_symbols}
        if removed:
            logger.warning(f"Skipping restore for removed symbols: {removed}")
            positions = {s: p for s, p in positions.items() if s in active_symbols}
        if not positions:
            return

        # 通知 risk_manager 恢復持倉
        await self.bus.publish(
            AgentMessage(
                sender="orchestrator",
                receiver="risk_manager",
                msg_type=MessageType.COMMAND,
                payload={
                    "action": "restore_positions",
                    "positions": positions,
                    "equity": equity,
                },
            )
        )

        # 通知 signal_engineer 恢復持倉（避免重複開倉）
        await self.bus.publish(
            AgentMessage(
                sender="orchestrator",
                receiver="signal_engineer",
                msg_type=MessageType.COMMAND,
                payload={
                    "action": "restore_positions",
                    "positions": positions,
                },
            )
        )

    async def run_backtest(self, symbols: Optional[List[str]] = None):
        """在指定資產上執行 FVG Combo3 回測"""
        from strategies.fvg_trend import FVGTrendStrategy

        symbols = symbols or settings.trading.symbols
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
            breakeven_atr=cfg.breakeven_atr,
            cooldown=cfg.cooldown,
            max_hold=cfg.max_hold,
        )

        results = {}
        for symbol in symbols:
            df = load_data(symbol, "5m")
            if df is None or len(df) < 1000:
                logger.warning(f"{symbol}: 數據不足")
                continue

            r = strategy.backtest(df, settings.trading.initial_capital, cfg.max_hold)
            m = r["metrics"]
            results[symbol] = m

            if "error" not in m:
                logger.info(
                    f"{symbol}: Ret {m['total_return_pct']:+.1f}% | "
                    f"WR {m['win_rate']:.1f}% | PF {m['profit_factor']:.2f} | "
                    f"Sharpe {m['sharpe_ratio']:.2f} | MDD {m['max_drawdown_pct']:.1f}% | "
                    f"Trades {m['total_trades']}"
                )

        return results

    async def screen_asset(self, symbol: str) -> dict:
        """篩選新資產是否適合 FVG 策略"""
        from strategies.fvg_trend import FVGTrendStrategy
        from data.historical_fetcher import ensure_data

        logger.info(f"篩選新資產: {symbol}")

        # 下載數據
        df = await ensure_data(symbol, "5m", "2021-04-01", "2026-04-01")
        if df is None or len(df) < 5000:
            return {"symbol": symbol, "viable": False, "reason": "數據不足"}

        # 回測
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
            breakeven_atr=cfg.breakeven_atr,
            cooldown=cfg.cooldown,
            max_hold=cfg.max_hold,
        )

        r = strategy.backtest(df, settings.trading.initial_capital, cfg.max_hold)
        m = r["metrics"]

        if "error" in m:
            return {"symbol": symbol, "viable": False, "reason": m["error"]}

        # 判斷是否可行
        viable = (
            m["total_return_pct"] > 100
            and m["profit_factor"] > 1.3
            and m["total_trades"] >= 50
            and m["sharpe_ratio"] > 0.8
        )

        result = {
            "symbol": symbol,
            "viable": viable,
            "return_pct": m["total_return_pct"],
            "win_rate": m["win_rate"],
            "profit_factor": m["profit_factor"],
            "sharpe": m["sharpe_ratio"],
            "max_dd": m["max_drawdown_pct"],
            "trades": m["total_trades"],
            "recommendation": "建議加入" if viable else "不建議",
        }

        logger.info(
            f"篩選結果 {symbol}: {'PASS' if viable else 'FAIL'} | "
            f"Ret {m['total_return_pct']:+.1f}% | WR {m['win_rate']:.1f}% | "
            f"PF {m['profit_factor']:.2f} | Sharpe {m['sharpe_ratio']:.2f}"
        )

        return result

    # ── 原有接口（保留）─────────────────────

    async def design_and_backtest(
        self,
        symbols: Optional[List[str]] = None,
        timeframe: str = "5m",
        use_tv_data: bool = True,
    ):
        """完整流程：拉取數據 → 設計策略 → 回測 → 風控評估"""
        symbols = symbols or settings.trading.symbols
        logger.info(f"啟動策略設計流程: symbols={symbols}, timeframe={timeframe}")

        if use_tv_data:
            for symbol in symbols:
                df = self.data.get_historical_klines(symbol, timeframe, n_bars=5000)
                if df is not None:
                    logger.info(f"已載入 {symbol} 歷史數據: {len(df)} 根 K 線")

        await self.bus.publish(
            AgentMessage(
                sender="orchestrator",
                receiver="quant_researcher",
                msg_type=MessageType.COMMAND,
                payload={
                    "action": "design_strategy",
                    "symbols": symbols,
                    "timeframe": timeframe,
                },
            )
        )

    async def _periodic_csv_sync(self):
        """Sync in-memory OHLCV buffers to CSV files every 10 minutes.
        Keeps CSV fresh for: (1) fast restarts, (2) manual analysis, (3) walk-forward optimization.
        """
        await asyncio.sleep(120)  # wait 2 min for initial data load
        while True:
            try:
                from data.historical_fetcher import load_data, save_data
                synced = []
                for symbol in settings.trading.symbols:
                    buf = self.signal.ohlcv_buffers.get(symbol)
                    if buf is None or len(buf) < 100:
                        continue
                    csv_df = load_data(symbol, "5m")
                    if csv_df is None:
                        continue
                    # Find new rows in buffer that aren't in CSV
                    new_rows = buf[buf.index > csv_df.index[-1]]
                    if len(new_rows) > 0:
                        merged = pd.concat([csv_df, new_rows])
                        merged = merged[~merged.index.duplicated(keep="last")]
                        merged = merged.sort_index()
                        save_data(merged, symbol, "5m")
                        synced.append(f"{symbol}(+{len(new_rows)})")
                if synced:
                    logger.info(f"CSV 同步完成: {', '.join(synced)}")
            except Exception as e:
                logger.warning(f"CSV 同步失敗: {e}")
            await asyncio.sleep(600)  # every 10 minutes

    async def _periodic_equity_sync(self):
        """Refresh trade_logger.equity from exchange every 60s.
        Exchange total equity already includes unrealized PnL of open positions,
        so this value is the source of truth for sizing and display.
        """
        await asyncio.sleep(60)  # let initial state settle
        while True:
            try:
                account = await asyncio.wait_for(
                    self.execution.get_account_info(), timeout=10
                )
                if account:
                    eq = None
                    if settings.exchange.is_mexc:
                        for asset in (account.get("data") or []):
                            if asset.get("currency") == "USDT":
                                eq = float(asset.get("equity", 0))
                                break
                    else:
                        eq = float(account.get("totalWalletBalance", 0))
                    if eq and eq > 0:
                        old = trade_logger.equity
                        trade_logger.equity = eq
                        trade_logger._save()
                        if abs(old - eq) > 0.5:
                            logger.info(f"Equity synced from exchange: ${old:,.2f} → ${eq:,.2f}")
            except Exception as e:
                logger.debug(f"Equity sync failed: {e}")
            await asyncio.sleep(60)

    async def _periodic_monthly_validation(self):
        """Check every 24h if monthly validation is due (30-day cycle)."""
        await asyncio.sleep(120)  # wait 2 min for data to load
        while True:
            try:
                state = self._load_validation_state()
                last_run = state.get("last_validation")
                days_since = 999
                if last_run:
                    from datetime import datetime
                    last_dt = datetime.fromisoformat(last_run)
                    days_since = (datetime.utcnow() - last_dt).days

                if days_since >= 30:
                    logger.info(f"Monthly validation due (last: {days_since}d ago), starting pipeline...")
                    await self._run_monthly_validation()
                else:
                    logger.info(f"Monthly validation not due ({days_since}/30 days)")
            except Exception as e:
                logger.error(f"Monthly validation check failed: {e}", exc_info=True)
            await asyncio.sleep(86400)  # check every 24h

    async def trigger_monthly_validation(self):
        """Manual trigger for /validate command."""
        asyncio.create_task(self._run_monthly_validation())

    async def _run_monthly_validation(self):
        """
        Full monthly validation pipeline:
        Phase 1: Signal Engineer per-asset WFO → candidate params
        Phase 2+3: Backtest Engineer (MC, sensitivity, random, fee) + Quant Researcher (regime, correlation)
        Phase 4: Aggregate → adopt/reject per asset → Telegram report
        """
        import uuid
        from datetime import datetime
        from config.settings import PER_ASSET_STRATEGY_PARAMS
        from strategies.param_optimizer import save_optimizer_state, get_current_params

        start_time = time.time()
        request_id = uuid.uuid4().hex[:8]
        self._validation_request_id = request_id
        self._validation_pending = {}

        if tg.enabled:
            await tg.send("🔬 <b>Monthly Validation Started</b>\nPhase 1: Per-asset WFO...")

        # ── Phase 1: Per-asset WFO (Signal Engineer) ──
        logger.info(f"[Validation {request_id}] Phase 1: Per-asset WFO")
        wfo_future = asyncio.get_event_loop().create_future()
        self._validation_pending["wfo_future"] = wfo_future

        await self.bus.publish(AgentMessage(
            sender="orchestrator",
            receiver="signal_engineer",
            msg_type=MessageType.VALIDATION_REQUEST,
            payload={"request_id": request_id},
        ))

        # Wait for WFO results (timeout 30 min)
        try:
            wfo_result = await asyncio.wait_for(wfo_future, timeout=1800)
        except asyncio.TimeoutError:
            logger.error(f"[Validation {request_id}] WFO timeout")
            if tg.enabled:
                await tg.send("❌ Monthly validation failed: WFO timeout")
            return

        wfo_assets = wfo_result.get("assets", {})
        logger.info(f"[Validation {request_id}] WFO done: {list(wfo_assets.keys())}")

        # ── Phase 2+3: Backtest tests + Research tests (parallel) ──
        if tg.enabled:
            await tg.send("Phase 2+3: Running backtest tests + regime filter...")

        # Prepare per-asset data for backtest engineer
        bt_futures = {}
        for symbol, wfo in wfo_assets.items():
            if not wfo.get("wfo_passed"):
                logger.info(f"[Validation] {symbol} WFO failed, skipping Phase 2")
                continue

            details = wfo.get("details", {})
            candidate_params = wfo.get("candidate_params")
            current_params = PER_ASSET_STRATEGY_PARAMS.get(symbol, {})
            df_oos = details.get("df_oos")

            if candidate_params is None or df_oos is None:
                continue

            fut = asyncio.get_event_loop().create_future()
            bt_futures[symbol] = fut
            self._validation_pending[f"bt_{symbol}"] = fut

            await self.bus.publish(AgentMessage(
                sender="orchestrator",
                receiver="backtest_engineer",
                msg_type=MessageType.VALIDATION_REQUEST,
                payload={
                    "request_id": f"{request_id}_{symbol}",
                    "symbol": symbol,
                    "candidate_params": candidate_params,
                    "current_params": current_params,
                    "df": df_oos,
                    "initial_capital": settings.trading.initial_capital,
                },
            ))

        # Quant researcher: regime + correlation (needs trade lists from WFO)
        research_future = asyncio.get_event_loop().create_future()
        self._validation_pending["research"] = research_future

        # Collect trade lists for all assets that passed WFO
        asset_trades = {}
        for symbol, wfo in wfo_assets.items():
            if wfo.get("wfo_passed"):
                details = wfo.get("details", {})
                new_trades = details.get("new_trades", [])
                if new_trades:
                    asset_trades[symbol] = new_trades

        await self.bus.publish(AgentMessage(
            sender="orchestrator",
            receiver="quant_researcher",
            msg_type=MessageType.VALIDATION_REQUEST,
            payload={
                "request_id": request_id,
                "asset_trades": asset_trades,
            },
        ))

        # Wait for all Phase 2+3 results (timeout 20 min)
        all_tasks = list(bt_futures.values()) + [research_future]
        if all_tasks:
            try:
                await asyncio.wait_for(
                    asyncio.gather(*all_tasks, return_exceptions=True),
                    timeout=1200,
                )
            except asyncio.TimeoutError:
                logger.error(f"[Validation {request_id}] Phase 2+3 timeout")

        bt_results = {}
        for symbol, fut in bt_futures.items():
            if fut.done() and not fut.cancelled():
                bt_results[symbol] = fut.result()
            else:
                bt_results[symbol] = {"all_passed": False, "error": "timeout"}

        research_result = {}
        if research_future.done() and not research_future.cancelled():
            research_result = research_future.result()

        # ── Phase 4: Aggregate + decide + report ──
        logger.info(f"[Validation {request_id}] Phase 4: Aggregating results")
        report_lines = ["📊 <b>Monthly Validation Report</b>", "━━━━━━━━━━━━━━━━━━━━━━━━"]
        adopted_symbols = []

        for symbol in settings.trading.symbols:
            wfo = wfo_assets.get(symbol, {})
            bt = bt_results.get(symbol, {})
            regime = research_result.get("regime", {}).get(symbol, {})

            line = f"\n┌─ {symbol} ─────────────────────"

            # WFO
            if wfo.get("wfo_passed"):
                cp = wfo.get("candidate_params", {})
                line += f"\n│ WFO:          ✅ sl={cp.get('sl_atr')} fvg={cp.get('fvg_min_size_pct')} cd={cp.get('cooldown')}"
            else:
                reason = wfo.get("reason", "failed")
                line += f"\n│ WFO:          ❌ {reason[:40]}"
                line += f"\n│ → ❌ KEEP CURRENT (WFO failed)"
                line += "\n└────────────────────────────"
                report_lines.append(line)
                continue

            # Backtest tests
            mc = bt.get("mc_bootstrap", {})
            sens = bt.get("param_sensitivity", {})
            rand = bt.get("random_entry", {})
            fee = bt.get("fee_stress", {})

            mc_icon = "✅" if mc.get("passed") else "❌"
            line += f"\n│ MC Bootstrap: {mc_icon} ruin {mc.get('ruin_pct', '?')}%"

            if not mc.get("passed"):
                line += f"\n│ → ❌ KEEP CURRENT (MC failed)"
                line += "\n└────────────────────────────"
                report_lines.append(line)
                continue

            sens_icon = "✅" if sens.get("passed") else "❌"
            line += f"\n│ Sensitivity:  {sens_icon} min PF {sens.get('worst_pf', '?')}"

            rand_icon = "✅" if rand.get("passed") else "❌"
            edge = rand.get("edge_pct", 0)
            line += f"\n│ Random Edge:  {rand_icon} +{edge:.0f}% vs random"

            fee_icon = "✅" if fee.get("passed") else "❌"
            line += f"\n│ Fee Stress:   {fee_icon} PF {fee.get('pf_at_worst', '?')} @ 0.05%"

            # Regime
            regime_icon = "✅" if regime.get("passed", True) else "❌"
            worst_pf = regime.get("worst_pf", "?")
            line += f"\n│ Regime:       {regime_icon} worst PF {worst_pf}"

            # Check all tests passed
            bt_all = bt.get("all_passed", False)
            regime_pass = regime.get("passed", True)

            if not bt_all or not regime_pass:
                failed = []
                if not sens.get("passed"): failed.append("sensitivity")
                if not rand.get("passed"): failed.append("random")
                if not fee.get("passed"): failed.append("fee")
                if not regime_pass: failed.append("regime")
                line += f"\n│ → ❌ KEEP CURRENT ({', '.join(failed)} failed)"
                line += "\n└────────────────────────────"
                report_lines.append(line)
                continue

            # New vs Old comparison
            comparison = bt.get("comparison", {})
            wins = comparison.get("wins", 0)
            comps = comparison.get("comparisons", {})
            comp_str = " ".join([
                f"PF{'✅' if comps.get('pf')=='new' else '❌'}",
                f"Ret{'✅' if comps.get('return')=='new' else '❌'}",
                f"DD{'✅' if comps.get('dd')=='new' else '❌'}",
                f"Sharpe{'✅' if comps.get('sharpe')=='new' else '❌'}",
            ])
            comp_icon = "✅" if comparison.get("passed") else "❌"
            line += f"\n│ New vs Old:   {comp_icon} {wins}/4 ({comp_str})"

            new_m = comparison.get("new_metrics", bt.get("candidate_metrics", {}))
            old_m = comparison.get("old_metrics", bt.get("current_metrics", {}))
            line += (
                f"\n│   New: PF {new_m.get('pf', 0)}, "
                f"Ret {new_m.get('return_pct', 0):+.1f}%, "
                f"DD {new_m.get('max_dd', 0):.1f}%, "
                f"Sharpe {new_m.get('sharpe', 0):.1f}"
            )
            line += (
                f"\n│   Old: PF {old_m.get('pf', 0)}, "
                f"Ret {old_m.get('return_pct', 0):+.1f}%, "
                f"DD {old_m.get('max_dd', 0):.1f}%, "
                f"Sharpe {old_m.get('sharpe', 0):.1f}"
            )

            if comparison.get("passed"):
                # ADOPT new params
                candidate = wfo.get("candidate_params", {})
                PER_ASSET_STRATEGY_PARAMS[symbol] = {
                    **PER_ASSET_STRATEGY_PARAMS.get(symbol, {}),
                    **candidate,
                }
                self.signal._rebuild_strategy(candidate, symbol=symbol)
                adopted_symbols.append(symbol)
                line += "\n│ → ✅ ADOPTED"
            else:
                line += "\n│ → ❌ KEEP CURRENT (new not better)"

            line += "\n└────────────────────────────"
            report_lines.append(line)

        # Correlation info
        corr = research_result.get("correlation", {})
        corr_matrix = corr.get("matrix", {})
        if corr_matrix:
            corr_str = " | ".join(f"{k} {v:.2f}" for k, v in corr_matrix.items())
            report_lines.append(f"\n📈 Correlation: {corr_str}")

        elapsed = time.time() - start_time
        report_lines.append(f"⏱️ Runtime: {elapsed/60:.0f}m {elapsed%60:.0f}s")

        report = "\n".join(report_lines)
        logger.info(f"[Validation {request_id}] Complete. Adopted: {adopted_symbols}")

        if tg.enabled:
            await tg.send(report)

        # Save state
        self._save_validation_state({
            "last_validation": datetime.utcnow().isoformat(),
            "request_id": request_id,
            "adopted": adopted_symbols,
            "runtime_s": round(elapsed, 1),
        })

        # Save updated per-asset params if any adopted
        if adopted_symbols:
            save_optimizer_state({
                "last_optimization": datetime.utcnow().isoformat(),
                "current_params": get_current_params(),
                "monthly_validation": request_id,
            })

        self._validation_pending = {}

    def _on_validation_result(self, message: AgentMessage):
        """Route VALIDATION_RESULT to the correct future."""
        payload = message.payload
        req_id = payload.get("request_id", "")
        sender = message.sender

        logger.info(f"[Validation] Result from {sender}, request_id={req_id}")

        if sender == "signal_engineer":
            fut = self._validation_pending.get("wfo_future")
            if fut and not fut.done():
                fut.set_result(payload)
        elif sender == "backtest_engineer":
            symbol = payload.get("symbol", "")
            fut = self._validation_pending.get(f"bt_{symbol}")
            if fut and not fut.done():
                fut.set_result(payload)
        elif sender == "quant_researcher":
            fut = self._validation_pending.get("research")
            if fut and not fut.done():
                fut.set_result(payload)

    def _load_validation_state(self) -> dict:
        """Load validation state from JSON."""
        import json
        path = os.path.join("data", "validation_state.json")
        if os.path.exists(path):
            with open(path) as f:
                return json.load(f)
        return {}

    def _save_validation_state(self, state: dict):
        """Save validation state to JSON."""
        import json
        path = os.path.join("data", "validation_state.json")
        existing = self._load_validation_state()
        existing.update(state)
        with open(path, "w") as f:
            json.dump(existing, f, indent=2, default=str)

    async def _periodic_plan_order_refresh(self):
        """MEXC plan orders expire after 24h. Refresh every 20h to maintain SL protection."""
        await asyncio.sleep(300)  # wait 5 min for startup
        while True:
            try:
                await self.execution.refresh_mexc_plan_orders()
            except Exception as e:
                logger.error(f"Plan order refresh failed: {e}")
            await asyncio.sleep(72000)  # every 20 hours

    async def get_market_snapshot(self, symbol: str = "BTCUSDT") -> dict:
        return await self.data.get_full_market_snapshot(symbol)

    async def start_monitoring(self):
        logger.info("啟動市場監控")
        await self.bus.publish(
            AgentMessage(
                sender="orchestrator",
                receiver="market_analyst",
                msg_type=MessageType.COMMAND,
                payload={"action": "start_monitoring"},
            )
        )

    async def start_webhook(self):
        logger.info("啟動 TradingView Webhook 伺服器")

        async def on_tv_alert(payload: dict):
            await self.bus.publish(
                AgentMessage(
                    sender="webhook",
                    receiver="broadcast",
                    msg_type=MessageType.MARKET_UPDATE,
                    payload=payload,
                    priority=5,
                )
            )

        self.data.webhook.on_alert(on_tv_alert)
        await self.data.webhook.start()

    async def enable_trading(self):
        logger.info("開啟實盤交易")
        await self.bus.publish(
            AgentMessage(
                sender="orchestrator",
                receiver="execution_engineer",
                msg_type=MessageType.COMMAND,
                payload={"action": "enable_trading"},
            )
        )

    async def disable_trading(self):
        logger.info("關閉實盤交易")
        await self.bus.publish(
            AgentMessage(
                sender="orchestrator",
                receiver="execution_engineer",
                msg_type=MessageType.COMMAND,
                payload={"action": "disable_trading"},
            )
        )

    async def emergency_stop(self):
        """緊急停止：取消掛單 + 平掉所有持倉 + 停止交易"""
        logger.warning("緊急停止 — 平掉所有持倉")
        # 1. Close all positions FIRST (before disabling trading!)
        for symbol in settings.trading.symbols:
            await self.bus.publish(
                AgentMessage(
                    sender="orchestrator",
                    receiver="execution_engineer",
                    msg_type=MessageType.RISK_ALERT,
                    payload={"action": "close_position", "symbol": symbol, "reason": "emergency_stop"},
                    priority=10,
                )
            )
        # Give time for close orders to process
        await asyncio.sleep(3)
        # 2. Then disable trading and cancel remaining orders
        await self.bus.publish(
            AgentMessage(
                sender="orchestrator",
                receiver="execution_engineer",
                msg_type=MessageType.COMMAND,
                payload={"action": "disable_trading"},
                priority=10,
            )
        )
        await self.bus.publish(
            AgentMessage(
                sender="orchestrator",
                receiver="execution_engineer",
                msg_type=MessageType.COMMAND,
                payload={"action": "cancel_all"},
                priority=10,
            )
        )

    async def get_risk_report(self) -> dict:
        report = self.risk.generate_risk_report()
        return report.model_dump()

    def get_team_status(self) -> dict:
        status = {}
        for agent in self._agents:
            s = agent.get_status()
            # 信號工程師額外詳情
            if hasattr(agent, "get_status_detail"):
                s["detail"] = agent.get_status_detail()
            status[agent.agent_id] = s
        return status

    async def _setup_exchange_config(self):
        """Set leverage=1x and CROSSED margin for each trading symbol."""
        is_mexc = settings.exchange.is_mexc
        if is_mexc:
            # MEXC: leverage is set per-order (leverage=1 in order body)
            # Cannot pre-set leverage without existing position
            logger.info("MEXC: leverage=1x will be set per-order (cross margin)")
            return

        # Binance: pre-set leverage and margin type
        for symbol in settings.trading.symbols:
            for attempt in range(3):
                try:
                    params_lev = {
                        "symbol": symbol,
                        "leverage": "1",
                        "timestamp": self.execution._get_timestamp(),
                    }
                    await self.execution._signed_request("POST", "/fapi/v1/leverage", params_lev)
                    logger.info(f"{symbol} leverage set to 1x")
                    break
                except Exception as e:
                    if "-4028" in str(e) or "already" in str(e).lower():
                        logger.info(f"{symbol} leverage already 1x")
                        break
                    elif attempt < 2:
                        logger.warning(f"{symbol} leverage retry {attempt+1}/3: {e}")
                        await asyncio.sleep(2 * (attempt + 1))
                    else:
                        logger.critical(f"FATAL: 無法設定 {symbol} leverage: {e}")
                        raise RuntimeError(f"Cannot start live trading: leverage setting failed for {symbol}: {e}")

            try:
                params_margin = {
                    "symbol": symbol,
                    "marginType": "CROSSED",
                    "timestamp": self.execution._get_timestamp(),
                }
                await self.execution._signed_request("POST", "/fapi/v1/marginType", params_margin)
                logger.info(f"{symbol} margin type set to CROSSED")
            except Exception as e:
                if "-4046" in str(e):
                    logger.info(f"{symbol} margin type already CROSSED")
                else:
                    logger.warning(f"{symbol} set margin type failed: {e}")

    def get_message_log(self, last_n: int = 20) -> list:
        messages = self.bus.message_log[-last_n:]
        return [m.model_dump(mode="json") for m in messages]
