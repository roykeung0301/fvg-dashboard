"""信號工程師 — FVG Trend Follow 信號管線

整合 FVG Trend Follow Combo3 策略:
- 維護每個交易對的 5m OHLCV 滾動窗口
- 每根新 K 線收盤時重新計算 FVG 信號
- 使用風險管理師計算的倉位大小
- 檢測趨勢反轉發出平倉信號
"""

from __future__ import annotations

from typing import Optional, Dict
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from agents.base_agent import BaseAgent
from models.messages import AgentMessage, MessageType
from models.signals import Signal, SignalType, SignalStrength
from strategies.fvg_trend import FVGTrendStrategy
from strategies.param_optimizer import (
    optimize_params, should_reoptimize, get_current_params,
    load_optimizer_state, save_optimizer_state, BASE_PARAMS,
)
from config.settings import settings
from notifications.telegram_bot import notifier as tg
from notifications.trade_logger import trade_logger


class SignalEngineer(BaseAgent):
    """
    信號工程師：
    - 維護每幣種 5m OHLCV 滾動數據
    - 使用 FVG Trend Follow 策略生成信號
    - 追蹤持倉狀態，發出入場/出場信號
    - 風險管理師審核後才執行
    """

    def __init__(self):
        super().__init__(agent_id="signal_engineer", name="信號工程師")
        self.active_strategies: list[dict] = []
        self.signal_history: list[Signal] = []  # capped at 500
        self._max_signal_history = 500

        # ── FVG 策略實例 (每資產獨立參數) ──
        from config.settings import PER_ASSET_STRATEGY_PARAMS
        cfg = settings.strategy
        self._base_strategy_params = dict(
            daily_ema_fast=cfg.daily_ema_fast,
            daily_ema_slow=cfg.daily_ema_slow,
            max_active_fvgs=cfg.max_active_fvgs,
            trail_start_atr=cfg.trail_start_atr,
            trail_atr=cfg.trail_atr,
            breakeven_atr=cfg.breakeven_atr,
            atr_period=cfg.atr_period,
            vol_mult=cfg.vol_mult,
            min_bars_for_trend_exit=cfg.min_bars_for_trend_exit,
        )
        self.fvg_strategies: Dict[str, FVGTrendStrategy] = {}
        for sym in settings.trading.symbols:
            asset_params = PER_ASSET_STRATEGY_PARAMS.get(sym, {})
            merged = {**self._base_strategy_params, **asset_params}
            self.fvg_strategies[sym] = FVGTrendStrategy(**merged)
        # Backward compat: default strategy for unknown symbols
        self.fvg_strategy = self.fvg_strategies.get(
            settings.trading.symbols[0],
            FVGTrendStrategy(**self._base_strategy_params)
        )

        # ── 每幣種滾動 OHLCV 數據 ──
        self.ohlcv_buffers: Dict[str, pd.DataFrame] = {}
        self.max_buffer_bars = 2000  # 保留最多 2000 根 5m K 線

        # ── 持倉追蹤 ──
        self.positions: Dict[str, dict] = {}  # {symbol: {side, entry_price, entry_time, sl}}
        self.last_signals: Dict[str, int] = {}  # {symbol: last_signal_value}
        self.last_candle_time: Dict[str, datetime] = {}
        self._close_cooldown: Dict[str, datetime] = {}  # {symbol: close_time} for post-close cooldown
        self._skip_cooldown_once: bool = False  # One-time flag: skip all cooldown on first compute

    def _rebuild_strategy(self, params: dict, symbol: str = None):
        """Rebuild FVG strategy with new optimized parameters.
        If symbol is given, rebuild only that asset's strategy.
        Otherwise rebuild all strategies with the same params.
        """
        merged = {**self._base_strategy_params}
        merged.update(params)
        merged["atr_period"] = settings.strategy.atr_period
        if symbol:
            self.fvg_strategies[symbol] = FVGTrendStrategy(**merged)
            self.logger.info(
                f"Strategy rebuilt for {symbol}: score={params.get('entry_min_score')}, "
                f"fvg={params.get('fvg_min_size_pct')}, sl={params.get('sl_atr')}"
            )
        else:
            for sym in self.fvg_strategies:
                self.fvg_strategies[sym] = FVGTrendStrategy(**merged)
            self.logger.info(
                f"Strategy rebuilt (all): score={params.get('entry_min_score')}, "
                f"fvg={params.get('fvg_min_size_pct')}, sl={params.get('sl_atr')}"
            )
        # Keep backward compat
        self.fvg_strategy = next(iter(self.fvg_strategies.values()), self.fvg_strategy)

    async def on_start(self):
        # Per-asset params from settings; fallback to optimizer_state for symbols without override
        from config.settings import PER_ASSET_STRATEGY_PARAMS
        opt_params = get_current_params()
        for sym in settings.trading.symbols:
            params = PER_ASSET_STRATEGY_PARAMS.get(sym, opt_params)
            self._rebuild_strategy(params, symbol=sym)
        state = load_optimizer_state()
        last_opt = state.get("last_optimization", "never")
        self.logger.info(
            f"FVG Trend Follow 已載入 (last optimized: {last_opt}) | "
            f"交易對: {settings.trading.symbols}"
        )

    async def handle_message(self, message: AgentMessage):
        handlers = {
            MessageType.MARKET_UPDATE: self._on_market_update,
            MessageType.RISK_ALERT: self._on_risk_alert,
            MessageType.COMMAND: self._on_command,
            MessageType.EXECUTION_REPORT: self._on_execution_report,
            MessageType.VALIDATION_REQUEST: self._on_validation_request,
        }
        handler = handlers.get(message.msg_type)
        if handler:
            await handler(message)

    # ── 消息處理 ─────────────────────────────

    async def _on_market_update(self, msg: AgentMessage):
        """收到新 K 線數據，更新 buffer 並計算信號"""
        market = msg.payload
        symbol = market.get("symbol", "")
        if not symbol or symbol not in settings.trading.symbols:
            return

        # 更新 OHLCV buffer
        candle = market.get("candle")
        if candle:
            self._append_candle(symbol, candle)

        # 檢查是否為新的 5m K 線收盤
        candle_time = market.get("candle_close_time")
        if candle_time:
            if isinstance(candle_time, str):
                candle_time = pd.Timestamp(candle_time)
            last = self.last_candle_time.get(symbol)
            if last and candle_time <= last:
                return  # 不是新 K 線
            self.last_candle_time[symbol] = candle_time
            buf_len = len(self.ohlcv_buffers.get(symbol, []))
            self.logger.info(f"新 K 線收盤 {symbol} @ {candle_time} | buffer={buf_len}")

            # Sync buffer → CSV on every candle close
            try:
                from data.historical_fetcher import load_data, save_data
                buf = self.ohlcv_buffers.get(symbol)
                csv_df = load_data(symbol, "5m")
                if buf is not None and csv_df is not None and len(buf) > 100:
                    new_rows = buf[buf.index > csv_df.index[-1]]
                    if len(new_rows) > 0:
                        merged = pd.concat([csv_df, new_rows])
                        merged = merged[~merged.index.duplicated(keep="last")].sort_index()
                        save_data(merged, symbol, "5m")
            except Exception as e:
                self.logger.debug(f"CSV sync {symbol}: {e}")
        else:
            # No candle_close_time means this is a ticker update, skip signal calc
            return

        # 有足夠數據才計算信號
        df = self.ohlcv_buffers.get(symbol)
        if df is None or len(df) < 500:
            self.logger.warning(f"{symbol} buffer 不足: {len(df) if df is not None else 0}/500")
            return

        if self.state.get("paused"):
            # Auto-resume after 1 hour (safety net)
            paused_time = self.state.get("paused_time")
            if paused_time and (datetime.utcnow() - paused_time).total_seconds() > 3600:
                self.state["paused"] = False
                self.state.pop("paused_time", None)
                self.logger.info("信號生成自動恢復（暫停超過 1 小時）")
            else:
                # CRITICAL: Even when paused, MUST still run exit logic for open positions
                # Otherwise SL/trend reversal never fires → death spiral
                if symbol in self.positions:
                    self.logger.info(f"[Paused] 但 {symbol} 有持倉，仍執行出場檢查")
                    await self._compute_fvg_signal(symbol, df, exit_only=True)
                return

        await self._compute_fvg_signal(symbol, df)

    async def _on_risk_alert(self, msg: AgentMessage):
        """風險警報 → 只有嚴重警報才暫停信號生成"""
        alert = msg.payload.get("alert", "")
        # market_anomaly (ticker Z-score) is too sensitive for 1h strategy — log only
        if alert == "market_anomaly":
            self.logger.info(f"收到市場異常警報（不影響信號計算）: {msg.payload.get('details', {}).get('symbol', '')}")
            return
        # Only truly critical alerts pause signal generation (with auto-resume)
        if alert in ("extreme_news", "daily_trade_limit", "max_drawdown"):
            self.state["paused"] = True
            self.state["paused_time"] = datetime.utcnow()
            self.logger.warning(f"因風險警報 [{alert}] 暫停信號生成（1h 後自動恢復）")

    async def _on_command(self, msg: AgentMessage):
        action = msg.payload.get("action")
        if action == "deploy_strategy":
            strategy = msg.payload.get("strategy", {})
            self.active_strategies.append(strategy)
            self.logger.info(f"部署策略: {strategy.get('strategy_name', 'FVG Trend Follow')}")
        elif action == "resume":
            self.state["paused"] = False
            self.logger.info("信號生成已恢復")
        elif action == "load_historical":
            # 載入歷史數據到 buffer (只保留 OHLCV 列，截斷至 buffer 上限)
            symbol = msg.payload.get("symbol", "")
            data = msg.payload.get("data")
            if symbol and data is not None:
                cols = [c for c in ["open", "high", "low", "close", "volume"] if c in data.columns]
                trimmed = data[cols].tail(self.max_buffer_bars).copy()
                self.ohlcv_buffers[symbol] = trimmed
                self.logger.info(f"已載入 {symbol} 歷史數據: {len(trimmed)} bars (from {len(data)})")
        elif action == "compute_now":
            # 重啟後立即用 buffer 中最後一根已收盤 K 線計算信號（不用等下一根收盤）
            computed = []
            for symbol in list(self.ohlcv_buffers.keys()):
                df = self.ohlcv_buffers.get(symbol)
                if df is not None and len(df) >= 500:
                    await self._compute_fvg_signal(symbol, df)
                    computed.append(symbol)
            # Reset one-time cooldown skip after first compute cycle
            if self._skip_cooldown_once:
                self._skip_cooldown_once = False
                self.logger.info("一次性 cooldown 跳過已重置")
            if computed:
                self.logger.info(f"啟動即時信號計算完成: {computed}")
            else:
                self.logger.info("啟動即時信號計算: 無足夠數據的幣種")
        elif action == "reoptimize":
            # Trigger parameter re-optimization
            await self._run_reoptimization()
        elif action == "restore_positions":
            # 重啟後恢復持倉狀態
            restored = msg.payload.get("positions", {})
            for sym, pos in restored.items():
                entry_p = pos.get("entry_price", 0)
                # Parse entry_time: could be ISO string (HKT) from trade_logger or datetime
                raw_et = pos.get("entry_time")
                if isinstance(raw_et, str):
                    try:
                        parsed = datetime.fromisoformat(raw_et.replace("Z", "+00:00"))
                        # Convert to naive UTC for consistent comparison with datetime.utcnow()
                        if parsed.tzinfo is not None:
                            from datetime import timezone as _tz
                            parsed = parsed.astimezone(_tz.utc).replace(tzinfo=None)
                        entry_time = parsed
                    except (ValueError, AttributeError):
                        entry_time = datetime.utcnow()
                elif isinstance(raw_et, datetime):
                    entry_time = raw_et
                else:
                    entry_time = datetime.utcnow()
                self.positions[sym] = {
                    "side": pos.get("side", ""),
                    "entry_price": entry_p,
                    "entry_time": entry_time,
                    "sl": pos.get("sl", 0),
                    "quantity": pos.get("quantity", pos.get("notional", 0) / entry_p if entry_p > 0 else 0),
                    "best_price": pos.get("best_price", entry_p),
                    "trail_sl": pos.get("trail_sl", None),
                }
                # 設置 last_signal 避免重複開倉
                self.last_signals[sym] = 1 if pos.get("side") == "long" else -1
            if restored:
                self.logger.info(f"已恢復 {len(restored)} 個持倉: {list(restored.keys())}")

    async def _on_execution_report(self, msg: AgentMessage):
        """更新持倉狀態"""
        report = msg.payload
        symbol = report.get("symbol", "")
        status = report.get("status", "")
        side = report.get("side", "")

        if report.get("action") == "close_failed":
            # Close order failed — clear pending_close so exit logic can retry next candle
            pos = self.positions.get(symbol)
            if pos:
                pos.pop("pending_close", None)
                pos.pop("pending_close_time", None)
                pos.pop("close_reason", None)
                self.logger.warning(f"Close failed for {symbol}, will retry next candle")
            return

        if report.get("action") == "close":
            pos = self.positions.get(symbol)
            if pos:
                exit_price = report.get("avg_fill_price") or 0
                entry_price = pos.get("entry_price", 0)
                qty = pos.get("quantity", 0)
                side_val = pos.get("side", "long")
                close_reason = pos.get("close_reason", report.get("reason", "risk_alert"))
                # Fallback: use entry_price as exit if no market price available
                if exit_price <= 0:
                    exit_price = entry_price
                    self.logger.warning(f"No market price for {symbol} close, using entry price")
                if qty > 0:
                    pnl = (exit_price - entry_price) * qty if side_val == "long" else (entry_price - exit_price) * qty
                    pnl_pct = (pnl / (entry_price * qty) * 100) if entry_price * qty > 0 else 0
                    trade_logger.log_exit(
                        symbol=symbol, side=side_val,
                        entry_price=entry_price, exit_price=exit_price,
                        quantity=qty, pnl=pnl, pnl_pct=pnl_pct,
                        exit_reason=close_reason,
                    )
                    # Use net_pnl (after commission) for TG display
                    last_trade = trade_logger.trades[-1] if trade_logger.trades else None
                    net_pnl = last_trade["net_pnl"] if last_trade else pnl
                    net_pnl_pct = last_trade["pnl_pct"] if last_trade else pnl_pct
                    if tg.enabled:
                        tg.equity = trade_logger.equity
                        await tg.notify_exit(
                            symbol=symbol, side=side_val,
                            entry_price=entry_price, exit_price=exit_price,
                            quantity=qty, pnl=net_pnl, pnl_pct=net_pnl_pct,
                            exit_reason=close_reason,
                        )
                    self.logger.info(
                        f"平倉 {symbol} {side_val} | reason={close_reason} | "
                        f"entry=${entry_price:,.2f} exit=${exit_price:,.2f} PnL=${net_pnl:+,.2f} ({pnl_pct:+.1f}%)"
                    )
            self.positions.pop(symbol, None)
            self.last_signals[symbol] = 0  # Reset so same direction can re-enter after cooldown
            self._close_cooldown[symbol] = datetime.utcnow()
            self.logger.info(f"已平倉 {symbol} (via execution report)")
        elif status == "filled":
            fill_price = report.get("avg_fill_price", 0)
            fill_qty = report.get("filled_quantity", 0)
            sl = report.get("stop_loss", 0)
            self.positions[symbol] = {
                "side": side,
                "entry_price": fill_price,
                "quantity": fill_qty,
                "entry_time": datetime.utcnow(),
                "sl": sl,
                "best_price": fill_price,
                "trail_sl": None,
            }
            self.logger.info(f"已開倉 {symbol} {side} @ ${fill_price:,.2f} x {fill_qty:.6f}")

    # ── FVG 信號計算 ─────────────────────────

    async def _emergency_sl_check(self, symbol: str, df: pd.DataFrame):
        """Fallback SL check using raw candle data when generate_signals fails."""
        pos = self.positions.get(symbol)
        if not pos:
            return
        try:
            last_row = df.iloc[-1]
            candle_high = float(last_row.get("high", 0))
            candle_low = float(last_row.get("low", 0))
            price = float(last_row.get("close", 0))
            pos_sl = pos.get("sl", 0)
            should_close = False
            if pos_sl > 0:
                if pos["side"] == "long" and candle_low <= pos_sl:
                    should_close = True
                elif pos["side"] == "short" and candle_high >= pos_sl:
                    should_close = True
            if should_close:
                self.logger.warning(f"緊急 SL 觸發 {symbol} (策略計算失敗後備用)")
                qty = pos.get("quantity", 0)
                if qty <= 0:
                    return
                self.positions[symbol]["pending_close"] = True
                self.positions[symbol]["pending_close_time"] = datetime.utcnow()
                signal = Signal(
                    symbol=symbol, signal_type=SignalType.CLOSE, confidence=1.0,
                    suggested_size=0,
                    metadata={"side": pos["side"], "current_price": price,
                              "quantity": qty, "reason": "emergency_sl"},
                )
                await self.send("risk_manager", MessageType.TRADE_SIGNAL, signal.model_dump(mode="json"))
        except Exception as e2:
            self.logger.error(f"Emergency SL check also failed for {symbol}: {e2}")

    async def _compute_fvg_signal(self, symbol: str, df: pd.DataFrame, exit_only: bool = False):
        """使用 FVG Trend Follow 策略計算信號。exit_only=True 時只檢查出場，不開新倉。"""
        try:
            # Use per-asset strategy instance
            strategy = self.fvg_strategies.get(symbol, self.fvg_strategy)
            sig_df = strategy.generate_signals(
                df.copy(),
                skip_cooldown_last_bar=self._skip_cooldown_once,
            )
        except Exception as e:
            self.logger.error(f"信號計算失敗 {symbol}: {e}")
            # CRITICAL: Even if strategy fails, check raw SL for open positions
            if symbol in self.positions:
                await self._emergency_sl_check(symbol, df)
            return

        # 取最後一根 K 線的信號
        if sig_df is None or len(sig_df) == 0:
            return

        last_row = sig_df.iloc[-1]
        current_signal = int(last_row.get("signal", 0))
        prev_signal = self.last_signals.get(symbol, 0)
        self.last_signals[symbol] = current_signal

        price = float(last_row["close"])
        raw_trend = last_row.get("daily_trend", 0)
        _dt = int(raw_trend) if not (isinstance(raw_trend, float) and np.isnan(raw_trend)) else 0
        in_pos = symbol in self.positions
        self.logger.info(
            f"信號計算 {symbol} | signal={current_signal} prev={prev_signal} "
            f"trend={_dt} price=${price:,.2f} in_pos={in_pos}"
        )
        atr = float(last_row.get("atr", price * 0.02))
        raw_trend = last_row.get("daily_trend", 0)
        daily_trend = int(raw_trend) if not (isinstance(raw_trend, float) and np.isnan(raw_trend)) else 0
        sl_price = float(last_row.get("stop_loss", 0))

        in_position = symbol in self.positions
        pos = self.positions.get(symbol)

        # ── 出場信號 ──
        if in_position and pos and pos.get("pending_close"):
            # Timeout: if pending_close has been stuck for > 3 candles (15min for 5m bars), force clear it
            pc_time = pos.get("pending_close_time")
            if pc_time and (datetime.utcnow() - pc_time).total_seconds() > 3 * 300:
                self.logger.warning(f"pending_close timeout for {symbol}, clearing to retry")
                pos.pop("pending_close", None)
                pos.pop("pending_close_time", None)
                pos.pop("close_reason", None)
            else:
                return  # Close already in flight, wait for execution report
        if in_position and pos:
            should_close = False
            close_reason = ""

            # ── Update trailing stop (use high/low to match backtest) ──
            entry_price = pos.get("entry_price", 0)
            pos_atr = atr if atr > 0 else (entry_price * 0.02)
            trail_start = settings.strategy.trail_start_atr
            trail_dist = settings.strategy.trail_atr
            candle_high = float(last_row.get("high", price))
            candle_low = float(last_row.get("low", price))

            if pos["side"] == "long":
                pos["best_price"] = max(pos.get("best_price", entry_price), candle_high)
                profit_atr = (pos["best_price"] - entry_price) / pos_atr if pos_atr > 0 else 0
                if profit_atr >= trail_start:
                    new_trail = pos["best_price"] - trail_dist * pos_atr
                    if pos.get("trail_sl") is None or new_trail > pos["trail_sl"]:
                        pos["trail_sl"] = new_trail
            else:  # short
                pos["best_price"] = min(pos.get("best_price", entry_price), candle_low)
                profit_atr = (entry_price - pos["best_price"]) / pos_atr if pos_atr > 0 else 0
                if profit_atr >= trail_start:
                    new_trail = pos["best_price"] + trail_dist * pos_atr
                    if pos.get("trail_sl") is None or new_trail < pos["trail_sl"]:
                        pos["trail_sl"] = new_trail

            # Persist best_price/trail_sl so they survive restarts
            trade_logger.update_position(symbol, {
                "best_price": pos.get("best_price"),
                "trail_sl": pos.get("trail_sl"),
            })

            # 趨勢反轉出場 (only after holding >= 6 bars of 5m = 30 min)
            # Long exits when daily_trend != 1 (neutral or bearish)
            # Short exits when daily_trend != -1 (neutral or bullish)
            entry_time = pos.get("entry_time", "")
            bars_held = 0
            if entry_time:
                from datetime import datetime as _dt
                try:
                    if isinstance(entry_time, str):
                        et = pd.Timestamp(entry_time)
                    else:
                        et = entry_time
                    bars_held = int((pd.Timestamp.utcnow() - et).total_seconds() / 300)
                except Exception:
                    bars_held = 999  # assume enough time if can't parse
            if bars_held >= 6:
                if pos["side"] == "long" and daily_trend != 1:
                    should_close = True
                    close_reason = "trend_reversal"
                elif pos["side"] == "short" and daily_trend != -1:
                    should_close = True
                    close_reason = "trend_reversal"

            # SL 觸發 (use candle low/high for intra-bar detection, matching backtest)
            pos_sl = pos.get("sl") or 0
            if pos_sl > 0:
                if pos["side"] == "long" and candle_low <= pos_sl:
                    should_close = True
                    close_reason = "stop_loss"
                elif pos["side"] == "short" and candle_high >= pos_sl:
                    should_close = True
                    close_reason = "stop_loss"

            # Trail SL 觸發 (use candle low/high)
            trail_sl = pos.get("trail_sl")
            if trail_sl is not None:
                if pos["side"] == "long" and candle_low <= trail_sl:
                    should_close = True
                    close_reason = "trail_stop"
                elif pos["side"] == "short" and candle_high >= trail_sl:
                    should_close = True
                    close_reason = "trail_stop"

            # 超時出場 (max_hold bars of 5m; max_hold * 5 minutes)
            if pos.get("entry_time"):
                minutes_held = (datetime.utcnow() - pos["entry_time"]).total_seconds() / 60
                if minutes_held >= settings.strategy.max_hold * 5:
                    should_close = True
                    close_reason = "timeout"

            if should_close:
                side = pos["side"]
                entry_price = pos.get("entry_price", 0)
                qty = pos.get("quantity", 0)

                # Guard: qty must be > 0, otherwise close order will fail silently
                if qty <= 0:
                    self.logger.error(
                        f"持倉 {symbol} qty={qty} 異常，強制清除（避免無限循環）"
                    )
                    closed_side = pos.get("side", "")
                    self.positions.pop(symbol, None)
                    self.last_signals[symbol] = 0
                    trade_logger.force_close_position(symbol)
                    # Notify risk_manager to clear ghost position
                    await self.broadcast(
                        MessageType.EXECUTION_REPORT,
                        {"symbol": symbol, "action": "close", "side": closed_side,
                         "avg_fill_price": 0, "filled_quantity": 0,
                         "reason": "force_close_corrupt_qty"},
                    )
                    return

                self.logger.info(
                    f"平倉信號 {symbol} {side} | reason={close_reason} | price=${price:,.2f}"
                )

                # ── Debug TG 出場信號通知 (signal only, PnL logged on execution confirm) ──
                if settings.trading.debug_signal_notify and tg.enabled:
                    try:
                        notional = price * qty
                        held_h = 0
                        if pos.get("entry_time"):
                            held_h = (datetime.utcnow() - pos["entry_time"]).total_seconds() / 3600
                        raw_pnl = (price - entry_price) * qty if side == "long" else (entry_price - price) * qty
                        pnl_emoji = "📈" if raw_pnl >= 0 else "📉"
                        reason_map = {
                            "trend_reversal": "趨勢反轉",
                            "stop_loss": "觸發止損",
                            "trail_stop": "追蹤止損",
                            "timeout": f"超時 ({held_h:.0f}h)",
                        }
                        reason_str = reason_map.get(close_reason, close_reason)
                        await tg.send(
                            f"🔔 <b>EXIT SIGNAL: {side.upper()} {symbol}</b>\n"
                            f"━━━━━━━━━━━━━━━\n"
                            f"📌 Reason: {reason_str}\n"
                            f"💲 Entry: <code>${entry_price:,.2f}</code>\n"
                            f"💲 Exit (est): <code>${price:,.2f}</code>\n"
                            f"{pnl_emoji} PnL (est): <code>${raw_pnl:+,.2f}</code>\n"
                            f"📊 Qty: <code>{qty:.6f}</code> (${notional:,.2f})\n"
                            f"⏱ Held: {held_h:.0f}h\n"
                            f"\n<i>🔧 Debug mode — /signal_off to disable</i>"
                        )
                    except Exception as e:
                        self.logger.warning(f"Debug TG exit notify failed: {e}")

                # Mark pending close — actual logging happens in _on_execution_report
                self.positions[symbol]["pending_close"] = True
                self.positions[symbol]["pending_close_time"] = datetime.utcnow()
                self.positions[symbol]["close_reason"] = close_reason

                signal = Signal(
                    symbol=symbol,
                    signal_type=SignalType.CLOSE,
                    strength=SignalStrength.STRONG,
                    confidence=0.95,
                    suggested_size=0.0,
                    stop_loss=None,
                    take_profit=None,
                    source_indicators=[f"FVG_close_{close_reason}"],
                    metadata={
                        "reason": close_reason,
                        "current_price": price,
                        "side": side,
                        "quantity": qty,
                        "entry_price": entry_price,
                    },
                )
                self.signal_history.append(signal)
                if len(self.signal_history) > self._max_signal_history:
                    self.signal_history = self.signal_history[-self._max_signal_history:]
                await self.send("risk_manager", MessageType.TRADE_SIGNAL, signal.model_dump(mode="json"))
                return

        # ── 入場信號 (只在無持倉時) ──
        if exit_only:
            return  # Paused mode: only exit checks above, skip all entry logic

        # Post-close cooldown: wait at least `cooldown` bars before re-entry (per-asset)
        if not self._skip_cooldown_once:
            cd_time = self._close_cooldown.get(symbol)
            if cd_time:
                from config.settings import PER_ASSET_STRATEGY_PARAMS
                asset_cd = PER_ASSET_STRATEGY_PARAMS.get(symbol, {}).get("cooldown", settings.strategy.cooldown)
                minutes_since_close = (datetime.utcnow() - cd_time).total_seconds() / 60
                if minutes_since_close < asset_cd * 5:
                    return  # Still in cooldown period

        if not in_position and current_signal != 0 and current_signal != prev_signal:
            side = "long" if current_signal == 1 else "short"
            signal_type = SignalType.BUY if current_signal == 1 else SignalType.SELL

            # Per-asset 倉位上限 (ETH 40%, SOL 30%, FET 30%)
            asset_cfg = settings.risk.asset_configs.get(symbol)
            asset_limit = asset_cfg.max_position_pct if asset_cfg else 0.40
            equity = trade_logger.equity or settings.trading.initial_capital
            if equity <= 0:
                self.logger.error(f"Equity <= 0 (${equity}), skipping entry")
                return
            risk_pct = asset_limit
            position_pct = asset_limit
            # NaN ATR 防護: fallback to 2% of price
            safe_atr = atr if (atr and atr == atr and atr > 0) else price * 0.02
            # SL 方向驗證：long SL 必須低於入場，short SL 必須高於入場
            if sl_price > 0:
                if side == "long" and sl_price >= price:
                    self.logger.warning(f"Long SL ${sl_price} >= entry ${price}, using ATR fallback")
                    sl_price = 0  # Force ATR fallback
                elif side == "short" and sl_price <= price:
                    self.logger.warning(f"Short SL ${sl_price} <= entry ${price}, using ATR fallback")
                    sl_price = 0  # Force ATR fallback

            # ── 波動率資訊 (僅用於日誌，不調整倉位) ──
            vol_pct = (atr / price * 100) if price > 0 else 0.0
            if vol_pct > settings.risk.vol_very_high_pct:
                vol_regime = "VERY_HIGH"
            elif vol_pct > settings.risk.vol_high_pct:
                vol_regime = "HIGH"
            elif vol_pct < settings.risk.vol_low_pct:
                vol_regime = "LOW"
            else:
                vol_regime = "NORMAL"
            vol_adj = 1.0
            pre_vol_pct = position_pct

            signal = Signal(
                symbol=symbol,
                signal_type=signal_type,
                strength=SignalStrength.STRONG,
                confidence=0.85,
                suggested_size=round(position_pct, 4),
                stop_loss=sl_price if sl_price > 0 else None,
                take_profit=None,  # 趨勢跟蹤不設 TP
                source_indicators=[
                    f"FVG_{side}",
                    f"daily_trend={'up' if daily_trend == 1 else 'down'}",
                    f"atr={atr:.2f}",
                    f"vol_regime={vol_regime}",
                ],
                metadata={
                    "side": side,
                    "entry_price": price,
                    "sl_price": sl_price,
                    "atr": atr,
                    "risk_pct": risk_pct,
                    "position_pct": position_pct,
                    "pre_vol_position_pct": pre_vol_pct,
                    "vol_pct": round(vol_pct, 2),
                    "vol_regime": vol_regime,
                    "vol_adj": vol_adj,
                    "equity": equity,
                },
            )
            self.signal_history.append(signal)
            if len(self.signal_history) > self._max_signal_history:
                self.signal_history = self.signal_history[-self._max_signal_history:]
            self.logger.info(
                f"FVG 信號: {symbol} {side.upper()} @ ${price:,.2f} | "
                f"SL ${sl_price:,.2f} | Size {position_pct:.1%} | "
                f"Vol {vol_pct:.2f}% ({vol_regime}, adj={vol_adj}x)"
            )

            # ── Debug TG 信號通知 (可在 .env 設 DEBUG_SIGNAL_NOTIFY=false 關閉) ──
            if settings.trading.debug_signal_notify and tg.enabled:
                try:
                    qty = (equity * position_pct) / price if price > 0 else 0
                    notional = qty * price
                    risk_amt = abs(price - sl_price) * qty if sl_price > 0 else 0
                    sl_dist = abs(price - sl_price)
                    emoji = "🟢" if side == "long" else "🔴"
                    await tg.send(
                        f"{emoji} <b>SIGNAL: {side.upper()} {symbol}</b>\n"
                        f"━━━━━━━━━━━━━━━\n"
                        f"💲 Entry: <code>${price:,.2f}</code>\n"
                        f"🛑 SL: <code>${sl_price:,.2f}</code> ({sl_dist/price*100:.1f}%)\n"
                        f"📏 SL Distance: <code>${sl_dist:,.2f}</code> ({sl_dist/atr:.1f} ATR)\n"
                        f"📊 Qty: <code>{qty:.6f}</code> (${notional:,.2f})\n"
                        f"💰 Risk: <code>${risk_amt:,.2f}</code> ({risk_pct*100:.1f}% equity)\n"
                        f"⚖️ Position: {position_pct:.1%} of ${equity:,.0f}\n"
                        f"🌊 Vol: {vol_pct:.2f}% ({vol_regime}, {vol_adj}x)\n"
                        f"📈 Trend: {'UP' if daily_trend == 1 else 'DOWN'}\n"
                        f"\n<i>🔧 Debug mode — /signal_off to disable</i>"
                    )
                except Exception as e:
                    self.logger.warning(f"Debug TG notify failed: {e}")

            await self.send("risk_manager", MessageType.TRADE_SIGNAL, signal.model_dump(mode="json"))

    # ── 月度驗證 WFO ────────────────────────

    async def _on_validation_request(self, msg: AgentMessage):
        """Per-asset WFO for monthly validation pipeline. Returns candidate params without applying."""
        import asyncio
        request_id = msg.payload.get("request_id", "")

        self.logger.info("[Validation] Starting per-asset WFO...")

        try:
            results = await asyncio.to_thread(self._run_per_asset_wfo)
        except Exception as e:
            self.logger.error(f"[Validation] WFO failed: {e}", exc_info=True)
            results = {"error": str(e), "assets": {}}

        results["request_id"] = request_id
        await self.send("orchestrator", MessageType.VALIDATION_RESULT, results)

    def _run_per_asset_wfo(self) -> dict:
        """CPU-bound: run per-asset WFO using optimize_single_asset."""
        from strategies.param_optimizer import optimize_single_asset
        from strategies.validation_suite import compute_indicators
        from config.settings import PER_ASSET_STRATEGY_PARAMS
        from data.historical_fetcher import load_data

        # Determine WFO windows
        end_ts = pd.Timestamp.utcnow()
        from strategies.validation_thresholds import WFO_IS_MONTHS, WFO_OOS_MONTHS
        oos_start = end_ts - pd.DateOffset(months=WFO_OOS_MONTHS)
        train_start = oos_start - pd.DateOffset(months=WFO_IS_MONTHS)

        train_start_str = train_start.strftime("%Y-%m-%d")
        train_end_str = oos_start.strftime("%Y-%m-%d")
        oos_start_str = train_end_str
        oos_end_str = end_ts.strftime("%Y-%m-%d")

        self.logger.info(
            f"WFO windows: IS {train_start_str}→{train_end_str} | OOS {oos_start_str}→{oos_end_str}"
        )

        assets = {}
        for symbol in settings.trading.symbols:
            # Prefer live buffer, fallback to CSV
            df = self.ohlcv_buffers.get(symbol)
            if df is None or len(df) < 500:
                df = load_data(symbol, "5m")
            if df is None or len(df) < 1000:
                self.logger.warning(f"[WFO] {symbol}: insufficient data ({len(df) if df is not None else 0})")
                assets[symbol] = {"passed": False, "reason": "insufficient data"}
                continue

            current_params = PER_ASSET_STRATEGY_PARAMS.get(symbol, {})
            initial_capital = settings.trading.initial_capital

            try:
                result = optimize_single_asset(
                    symbol, df, current_params, initial_capital,
                    train_start_str, train_end_str, oos_start_str, oos_end_str,
                )
                assets[symbol] = result
                status = "PASS" if result.get("adopted") else "FAIL"
                self.logger.info(
                    f"[WFO] {symbol}: {status} | candidate={result.get('candidate_params')}"
                )
            except Exception as e:
                self.logger.error(f"[WFO] {symbol} failed: {e}", exc_info=True)
                assets[symbol] = {"passed": False, "reason": str(e)}

        return {"assets": assets}

    # ── 動態參數優化 ─────────────────────────

    async def _run_reoptimization(self):
        """Walk-forward re-optimization: 5mo train + 1mo OOS, dual-layer validation."""
        self.logger.info("Starting walk-forward re-optimization...")
        try:
            from strategies.param_optimizer import (
                optimize_params, save_optimizer_state,
                TRAIN_MONTHS, OOS_MONTHS,
            )

            # Use OHLCV buffers as data source
            dfs = {}
            for sym in settings.trading.symbols:
                if sym in self.ohlcv_buffers and len(self.ohlcv_buffers[sym]) >= 500:
                    dfs[sym] = self.ohlcv_buffers[sym].copy()

            if len(dfs) < 2:
                self.logger.warning("Not enough data for re-optimization, need at least 2 symbols with 500+ bars")
                return

            # Walk-forward windows: train 5mo | OOS 1mo
            end_ts = max(df.index[-1] for df in dfs.values())
            oos_start_ts = end_ts - pd.DateOffset(months=OOS_MONTHS)
            train_start_ts = oos_start_ts - pd.DateOffset(months=TRAIN_MONTHS)

            train_start = train_start_ts.strftime("%Y-%m-%d")
            train_end = oos_start_ts.strftime("%Y-%m-%d")
            oos_start = train_end
            oos_end = end_ts.strftime("%Y-%m-%d")

            self.logger.info(
                f"Walk-forward: train {train_start}→{train_end} | "
                f"OOS {oos_start}→{oos_end}"
            )

            result = optimize_params(dfs, train_start, train_end, oos_start, oos_end)

            adopted = result["adopted"]
            new_params = result["new_params"]
            reason = result["reason"]
            details = result["details"]

            if adopted and new_params:
                old_params = get_current_params()
                self._rebuild_strategy(new_params)
                save_optimizer_state({
                    "last_optimization": datetime.utcnow().isoformat(),
                    "current_params": new_params,
                    "previous_params": old_params,
                    "train_window": {"start": train_start, "end": train_end},
                    "oos_window": {"start": oos_start, "end": oos_end},
                    "oos_new": details.get("new_oos"),
                    "oos_old": details.get("old_oos"),
                })
                msg = (
                    f"✅ Walk-Forward 通過，參數已更新\n"
                    f"score={new_params.get('entry_min_score')}, "
                    f"fvg={new_params.get('fvg_min_size_pct')}, "
                    f"sl={new_params.get('sl_atr')}\n"
                    f"{reason}"
                )
            else:
                # Not adopted — update timestamp but keep current params
                save_optimizer_state({
                    "last_optimization": datetime.utcnow().isoformat(),
                    "current_params": get_current_params(),
                    "previous_params": load_optimizer_state().get("previous_params"),
                    "train_window": {"start": train_start, "end": train_end},
                    "oos_window": {"start": oos_start, "end": oos_end},
                    "last_rejected": {
                        "params": new_params,
                        "reason": reason,
                        "details": details,
                    },
                })
                msg = f"❌ Walk-Forward 不通過，保留現有參數\n{reason}"

            self.logger.info(msg)
            if tg.enabled:
                await tg.send(f"🔧 {msg}")

        except Exception as e:
            self.logger.error(f"Re-optimization failed: {e}")

    async def check_reoptimization(self):
        """Check if it's time to re-optimize (called periodically by orchestrator)."""
        if should_reoptimize():
            await self._run_reoptimization()

    # ── 數據管理 ─────────────────────────────

    def _append_candle(self, symbol: str, candle: dict):
        """添加新 K 線到滾動 buffer"""
        # Parse timestamp: could be int (ms epoch) or string/datetime
        raw_ts = candle.get("timestamp", candle.get("time", datetime.utcnow()))
        if isinstance(raw_ts, (int, float)):
            ts = pd.Timestamp(raw_ts, unit="ms")
        else:
            ts = pd.Timestamp(raw_ts)

        new_row = pd.DataFrame([{
            "open": float(candle.get("open", 0)),
            "high": float(candle.get("high", 0)),
            "low": float(candle.get("low", 0)),
            "close": float(candle.get("close", 0)),
            "volume": float(candle.get("volume", 0)),
        }], index=[ts])

        if symbol in self.ohlcv_buffers:
            buf = self.ohlcv_buffers[symbol]
            if len(buf) > 0:
                if buf.index[-1] == ts:
                    # Same timestamp: update in-place (partial → final candle)
                    buf.iloc[-1] = new_row.iloc[0]
                    return
                elif buf.index[-1] > ts:
                    return  # Truly old data, skip
            self.ohlcv_buffers[symbol] = pd.concat([buf, new_row]).tail(self.max_buffer_bars)
        else:
            self.ohlcv_buffers[symbol] = new_row

    def get_status_detail(self) -> dict:
        """詳細狀態報告"""
        return {
            "positions": {s: p["side"] for s, p in self.positions.items()},
            "buffer_sizes": {s: len(df) for s, df in self.ohlcv_buffers.items()},
            "signals_generated": len(self.signal_history),
            "paused": self.state.get("paused", False),
            "strategy": "FVG Trend Follow Combo3",
        }

    # ── 保留原有技術指標（供其他用途）──────────

    @staticmethod
    def calc_ema(series: pd.Series, span: int) -> pd.Series:
        return series.ewm(span=span, adjust=False).mean()

    @staticmethod
    def calc_rsi(series: pd.Series, period: int = 14) -> pd.Series:
        delta = series.diff()
        gain = delta.where(delta > 0, 0.0).rolling(period).mean()
        loss = (-delta.where(delta < 0, 0.0)).rolling(period).mean()
        rs = gain / loss.replace(0, np.nan)
        return 100 - (100 / (1 + rs))
