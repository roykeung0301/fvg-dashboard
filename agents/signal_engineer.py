"""信號工程師 — FVG Trend Follow 信號管線

整合 FVG Trend Follow Combo3 策略:
- 維護每個交易對的 1h OHLCV 滾動窗口
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
    - 維護每幣種 1h OHLCV 滾動數據
    - 使用 FVG Trend Follow 策略生成信號
    - 追蹤持倉狀態，發出入場/出場信號
    - 風險管理師審核後才執行
    """

    def __init__(self):
        super().__init__(agent_id="signal_engineer", name="信號工程師")
        self.active_strategies: list[dict] = []
        self.signal_history: list[Signal] = []

        # ── FVG 策略實例 ──
        cfg = settings.strategy
        self.fvg_strategy = FVGTrendStrategy(
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
            atr_period=cfg.atr_period,
        )

        # ── 每幣種滾動 OHLCV 數據 ──
        self.ohlcv_buffers: Dict[str, pd.DataFrame] = {}
        self.max_buffer_bars = 2000  # 保留最多 2000 根 1h K 線

        # ── 持倉追蹤 ──
        self.positions: Dict[str, dict] = {}  # {symbol: {side, entry_price, entry_time, sl}}
        self.last_signals: Dict[str, int] = {}  # {symbol: last_signal_value}
        self.last_candle_time: Dict[str, datetime] = {}

    def _rebuild_strategy(self, params: dict):
        """Rebuild FVG strategy with new optimized parameters."""
        merged = {**BASE_PARAMS}
        merged.update(params)
        # Also include non-optimized settings from config
        merged["atr_period"] = settings.strategy.atr_period
        self.fvg_strategy = FVGTrendStrategy(**merged)
        self.logger.info(
            f"Strategy rebuilt with params: score={params.get('entry_min_score')}, "
            f"fvg={params.get('fvg_min_size_pct')}, sl={params.get('sl_atr')}"
        )

    async def on_start(self):
        # Load optimized params if available
        opt_params = get_current_params()
        self._rebuild_strategy(opt_params)
        state = load_optimizer_state()
        last_opt = state.get("last_optimization", "never")
        self.logger.info(
            f"FVG Trend Follow 已載入 (last optimized: {last_opt}) | "
            f"Params: score={opt_params.get('entry_min_score')}, "
            f"fvg={opt_params.get('fvg_min_size_pct')}, "
            f"sl={opt_params.get('sl_atr')} | "
            f"交易對: {settings.trading.symbols}"
        )

    async def handle_message(self, message: AgentMessage):
        handlers = {
            MessageType.MARKET_UPDATE: self._on_market_update,
            MessageType.RISK_ALERT: self._on_risk_alert,
            MessageType.COMMAND: self._on_command,
            MessageType.EXECUTION_REPORT: self._on_execution_report,
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

        # 檢查是否為新的 1h K 線收盤
        candle_time = market.get("candle_close_time")
        if candle_time:
            if isinstance(candle_time, str):
                candle_time = pd.Timestamp(candle_time)
            last = self.last_candle_time.get(symbol)
            if last and candle_time <= last:
                return  # 不是新 K 線
            self.last_candle_time[symbol] = candle_time

        # 有足夠數據才計算信號
        df = self.ohlcv_buffers.get(symbol)
        if df is None or len(df) < 500:
            return

        if self.state.get("paused"):
            return

        await self._compute_fvg_signal(symbol, df)

    async def _on_risk_alert(self, msg: AgentMessage):
        """風險警報 → 暫停信號生成"""
        alert = msg.payload.get("alert", "")
        if alert in ("market_anomaly", "daily_trade_limit"):
            self.state["paused"] = True
            self.logger.warning(f"因風險警報 [{alert}] 暫停信號生成")

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
            # 載入歷史數據到 buffer
            symbol = msg.payload.get("symbol", "")
            data = msg.payload.get("data")
            if symbol and data is not None:
                self.ohlcv_buffers[symbol] = data.copy()
                self.logger.info(f"已載入 {symbol} 歷史數據: {len(data)} bars")
        elif action == "reoptimize":
            # Trigger parameter re-optimization
            await self._run_reoptimization()
        elif action == "restore_positions":
            # 重啟後恢復持倉狀態
            restored = msg.payload.get("positions", {})
            for sym, pos in restored.items():
                self.positions[sym] = {
                    "side": pos.get("side", ""),
                    "entry_price": pos.get("entry_price", 0),
                    "entry_time": pos.get("entry_time", datetime.utcnow()),
                    "sl": pos.get("sl", 0),
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

        if status == "filled":
            if report.get("action") == "close":
                self.positions.pop(symbol, None)
                self.logger.info(f"已平倉 {symbol}")
            else:
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

    async def _compute_fvg_signal(self, symbol: str, df: pd.DataFrame):
        """使用 FVG Trend Follow 策略計算信號"""
        try:
            sig_df = self.fvg_strategy.generate_signals(df.copy())
        except Exception as e:
            self.logger.error(f"信號計算失敗 {symbol}: {e}")
            return

        # 取最後一根 K 線的信號
        if sig_df is None or len(sig_df) == 0:
            return

        last_row = sig_df.iloc[-1]
        current_signal = int(last_row.get("signal", 0))
        prev_signal = self.last_signals.get(symbol, 0)
        self.last_signals[symbol] = current_signal

        price = float(last_row["close"])
        atr = float(last_row.get("atr", price * 0.02))
        raw_trend = last_row.get("daily_trend", 0)
        daily_trend = int(raw_trend) if not (isinstance(raw_trend, float) and np.isnan(raw_trend)) else 0
        sl_price = float(last_row.get("stop_loss", 0))

        in_position = symbol in self.positions
        pos = self.positions.get(symbol)

        # ── 出場信號 ──
        if in_position and pos:
            should_close = False
            close_reason = ""

            # ── Update trailing stop ──
            entry_price = pos.get("entry_price", 0)
            pos_atr = atr if atr > 0 else (entry_price * 0.02)
            trail_start = settings.strategy.trail_start_atr
            trail_dist = settings.strategy.trail_atr

            if pos["side"] == "long":
                pos["best_price"] = max(pos.get("best_price", entry_price), price)
                profit_atr = (pos["best_price"] - entry_price) / pos_atr if pos_atr > 0 else 0
                if profit_atr >= trail_start:
                    new_trail = pos["best_price"] - trail_dist * pos_atr
                    if pos.get("trail_sl") is None or new_trail > pos["trail_sl"]:
                        pos["trail_sl"] = new_trail
            else:  # short
                pos["best_price"] = min(pos.get("best_price", entry_price), price)
                profit_atr = (entry_price - pos["best_price"]) / pos_atr if pos_atr > 0 else 0
                if profit_atr >= trail_start:
                    new_trail = pos["best_price"] + trail_dist * pos_atr
                    if pos.get("trail_sl") is None or new_trail < pos["trail_sl"]:
                        pos["trail_sl"] = new_trail

            # 趨勢反轉出場
            if pos["side"] == "long" and daily_trend != 1:
                should_close = True
                close_reason = "trend_reversal"
            elif pos["side"] == "short" and daily_trend != -1:
                should_close = True
                close_reason = "trend_reversal"

            # SL 觸發 (fixed SL)
            if pos["side"] == "long" and price <= pos.get("sl", 0):
                should_close = True
                close_reason = "stop_loss"
            elif pos["side"] == "short" and price >= pos.get("sl", 0):
                should_close = True
                close_reason = "stop_loss"

            # Trail SL 觸發 (overrides if tighter)
            trail_sl = pos.get("trail_sl")
            if trail_sl is not None:
                if pos["side"] == "long" and price <= trail_sl:
                    should_close = True
                    close_reason = "trail_stop"
                elif pos["side"] == "short" and price >= trail_sl:
                    should_close = True
                    close_reason = "trail_stop"

            # 超時出場 (max_hold bars)
            if pos.get("entry_time"):
                hours_held = (datetime.utcnow() - pos["entry_time"]).total_seconds() / 3600
                if hours_held >= settings.strategy.max_hold:
                    should_close = True
                    close_reason = "timeout"

            if should_close:
                # Log exit + Telegram notification
                if pos:
                    entry_price = pos.get("entry_price", 0)
                    qty = pos.get("quantity", 0)
                    side = pos["side"]
                    if side == "long":
                        pnl = (price - entry_price) * qty
                    else:
                        pnl = (entry_price - price) * qty
                    pnl_pct = (pnl / (entry_price * qty) * 100) if entry_price * qty > 0 else 0

                    trade_logger.log_exit(
                        symbol=symbol, side=side,
                        entry_price=entry_price, exit_price=price,
                        quantity=qty, pnl=pnl, pnl_pct=pnl_pct,
                        exit_reason=close_reason,
                    )

                    if tg.enabled:
                        await tg.notify_exit(
                            symbol=symbol, side=side,
                            entry_price=entry_price, exit_price=price,
                            quantity=qty, pnl=pnl, pnl_pct=pnl_pct,
                            exit_reason=close_reason,
                        )

                signal = Signal(
                    symbol=symbol,
                    signal_type=SignalType.CLOSE,
                    strength=SignalStrength.STRONG,
                    confidence=0.95,
                    suggested_size=0.0,
                    stop_loss=None,
                    take_profit=None,
                    source_indicators=[f"FVG_close_{close_reason}"],
                    metadata={"reason": close_reason, "current_price": price},
                )
                self.signal_history.append(signal)
                await self.send("risk_manager", MessageType.TRADE_SIGNAL, signal.model_dump(mode="json"))
                return

        # ── 入場信號 (只在無持倉時) ──
        if not in_position and current_signal != 0 and current_signal != prev_signal:
            side = "long" if current_signal == 1 else "short"
            signal_type = SignalType.BUY if current_signal == 1 else SignalType.SELL

            # 風險基礎倉位計算
            risk_pct = settings.risk.risk_per_trade_pct / 100
            equity = trade_logger.equity or settings.trading.initial_capital
            risk_amount = equity * risk_pct
            sl_distance = abs(price - sl_price) if sl_price > 0 else atr * settings.strategy.sl_atr
            if sl_distance > 0:
                position_units = risk_amount / sl_distance
                position_value = position_units * price
                position_pct = position_value / equity
            else:
                position_pct = 0.05  # fallback 5%

            # 限制單資產上限
            asset_cfg = settings.risk.asset_configs.get(symbol)
            max_position_pct = asset_cfg.max_position_pct if asset_cfg else 0.30
            position_pct = min(position_pct, max_position_pct)

            # ── 波動率倉位調整 ──
            vol_pct = (atr / price * 100) if price > 0 else 0.0
            vol_adj = 1.0
            if vol_pct > settings.risk.vol_very_high_pct:
                vol_adj = 0.50
                vol_regime = "VERY_HIGH"
            elif vol_pct > settings.risk.vol_high_pct:
                vol_adj = 0.75
                vol_regime = "HIGH"
            elif vol_pct < settings.risk.vol_low_pct:
                vol_adj = 1.25
                vol_regime = "LOW"
            else:
                vol_regime = "NORMAL"

            pre_vol_pct = position_pct
            position_pct = position_pct * vol_adj
            # 調整後仍受單資產上限約束
            position_pct = min(position_pct, max_position_pct)

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
            self.logger.info(
                f"FVG 信號: {symbol} {side.upper()} @ ${price:,.2f} | "
                f"SL ${sl_price:,.2f} | Size {position_pct:.1%} | "
                f"Vol {vol_pct:.2f}% ({vol_regime}, adj={vol_adj}x)"
            )
            await self.send("risk_manager", MessageType.TRADE_SIGNAL, signal.model_dump(mode="json"))

    # ── 動態參數優化 ─────────────────────────

    async def _run_reoptimization(self):
        """Re-optimize parameters using last 6 months of data from buffers."""
        self.logger.info("Starting parameter re-optimization...")
        try:
            # Use OHLCV buffers as data source
            dfs = {}
            for sym in settings.trading.symbols:
                if sym in self.ohlcv_buffers and len(self.ohlcv_buffers[sym]) >= 500:
                    dfs[sym] = self.ohlcv_buffers[sym].copy()

            if len(dfs) < 2:
                self.logger.warning("Not enough data for re-optimization, need at least 2 symbols with 500+ bars")
                return

            # Use last 6 months as training window
            end_ts = max(df.index[-1] for df in dfs.values())
            start_ts = end_ts - pd.DateOffset(months=6)
            train_start = start_ts.strftime("%Y-%m-%d")
            train_end = end_ts.strftime("%Y-%m-%d")

            self.logger.info(f"Optimization window: {train_start} → {train_end}")

            best_params = optimize_params(dfs, train_start, train_end)

            if best_params:
                old_params = get_current_params()
                self._rebuild_strategy(best_params)
                save_optimizer_state({
                    "last_optimization": datetime.utcnow().isoformat(),
                    "current_params": best_params,
                    "previous_params": old_params,
                    "train_window": {"start": train_start, "end": train_end},
                })
                msg = (
                    f"Parameters re-optimized: "
                    f"score={best_params.get('entry_min_score')}, "
                    f"fvg={best_params.get('fvg_min_size_pct')}, "
                    f"sl={best_params.get('sl_atr')}"
                )
                self.logger.info(msg)
                if tg.enabled:
                    await tg.send_message(f"🔧 {msg}")
            else:
                self.logger.info("No better params found, keeping current")

        except Exception as e:
            self.logger.error(f"Re-optimization failed: {e}")

    async def check_reoptimization(self):
        """Check if it's time to re-optimize (called periodically by orchestrator)."""
        if should_reoptimize():
            await self._run_reoptimization()

    # ── 數據管理 ─────────────────────────────

    def _append_candle(self, symbol: str, candle: dict):
        """添加新 K 線到滾動 buffer"""
        new_row = pd.DataFrame([{
            "open": float(candle.get("open", 0)),
            "high": float(candle.get("high", 0)),
            "low": float(candle.get("low", 0)),
            "close": float(candle.get("close", 0)),
            "volume": float(candle.get("volume", 0)),
        }], index=[pd.Timestamp(
            candle.get("time", candle.get("timestamp", datetime.utcnow())),
            unit="ms" if isinstance(candle.get("timestamp"), (int, float)) else None,
        )])

        if symbol in self.ohlcv_buffers:
            self.ohlcv_buffers[symbol] = pd.concat([
                self.ohlcv_buffers[symbol], new_row
            ]).tail(self.max_buffer_bars)
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
