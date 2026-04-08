"""
FVCL3 策略 — 市場狀態鎖定：趨勢/震盪切換

核心概念:
1. 用多維度指標判斷市場處於「趨勢」還是「震盪」狀態
2. 趨勢模式 → 順勢突破追蹤 (Trend Following)
3. 震盪模式 → 均值回歸 (Mean Reversion)
4. 狀態切換有「鎖定期」，避免頻繁切換

狀態判斷:
- ADX (趨勢強度)
- BB 帶寬 (波動率擴張/收縮)
- EMA 斜率 (方向性)
- Choppiness Index (混亂度)

趨勢模式入場:
- 價格突破 EMA + ADX 確認 → 順勢做
- 追蹤止損出場

震盪模式入場:
- RSI 極值 + BB 觸碰 → 反向做
- 固定 TP/SL 出場

原始策略: 台指期 15min, CAGR 94.3%, WR 63%, PF 2.05
本版本: 適配 BTCUSDT Swing Trading (1h/4h)
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from strategies.base_strategy import BaseStrategy, TradeRecord


class FVCL3RegimeStrategy(BaseStrategy):

    name = "FVCL3 趨勢/震盪切換"
    description = "市場狀態鎖定: 趨勢模式順勢追蹤 + 震盪模式均值回歸"

    commission_rate = 0.0004
    slippage_rate = 0.0001

    def __init__(
        self,
        # ── 狀態判斷 ──
        adx_period: int = 14,
        adx_trend_threshold: float = 25,    # ADX > 25 = 趨勢
        adx_range_threshold: float = 20,    # ADX < 20 = 震盪
        chop_period: int = 14,
        chop_trend_threshold: float = 38.2, # Choppiness < 38.2 = 趨勢
        chop_range_threshold: float = 61.8, # Choppiness > 61.8 = 震盪
        bb_width_trend: float = 3.0,        # BB%W > 此值 = 波動擴張 (趨勢)
        bb_width_range: float = 1.5,        # BB%W < 此值 = 波動收縮 (震盪)
        regime_lock_bars: int = 6,          # 狀態鎖定 N 根 K 線

        # ── 趨勢模式參數 ──
        trend_ema_fast: int = 20,
        trend_ema_slow: int = 50,
        trend_entry_breakout: int = 20,     # Donchian Channel 突破週期
        trend_trailing_atr: float = 2.0,    # 追蹤止損 ATR 倍數
        trend_sl_atr: float = 2.5,          # 初始止損

        # ── 震盪模式參數 ──
        range_rsi_period: int = 7,
        range_rsi_oversold: float = 25,
        range_rsi_overbought: float = 75,
        range_bb_period: int = 20,
        range_bb_std: float = 2.0,
        range_tp_atr: float = 1.0,         # 均值回歸止盈
        range_sl_atr: float = 2.0,         # 均值回歸止損

        # ── 共用 ──
        atr_period: int = 14,
        min_atr_pct: float = 0.2,
        max_atr_pct: float = 5.0,
    ):
        # 狀態判斷
        self.adx_period = adx_period
        self.adx_trend_threshold = adx_trend_threshold
        self.adx_range_threshold = adx_range_threshold
        self.chop_period = chop_period
        self.chop_trend_threshold = chop_trend_threshold
        self.chop_range_threshold = chop_range_threshold
        self.bb_width_trend = bb_width_trend
        self.bb_width_range = bb_width_range
        self.regime_lock_bars = regime_lock_bars
        # 趨勢
        self.trend_ema_fast = trend_ema_fast
        self.trend_ema_slow = trend_ema_slow
        self.trend_entry_breakout = trend_entry_breakout
        self.trend_trailing_atr = trend_trailing_atr
        self.trend_sl_atr = trend_sl_atr
        # 震盪
        self.range_rsi_period = range_rsi_period
        self.range_rsi_oversold = range_rsi_oversold
        self.range_rsi_overbought = range_rsi_overbought
        self.range_bb_period = range_bb_period
        self.range_bb_std = range_bb_std
        self.range_tp_atr = range_tp_atr
        self.range_sl_atr = range_sl_atr
        # 共用
        self.atr_period = atr_period
        self.min_atr_pct = min_atr_pct
        self.max_atr_pct = max_atr_pct

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """計算所有指標 + 狀態判斷 + 依狀態生成信號"""
        price = df["close"]

        # ═══ 共用指標 ═══

        # ATR
        high_low = df["high"] - df["low"]
        high_close = (df["high"] - price.shift(1)).abs()
        low_close = (df["low"] - price.shift(1)).abs()
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        df["atr"] = tr.rolling(self.atr_period).mean()
        df["atr_pct"] = df["atr"] / price * 100

        # ADX
        df["adx"], df["plus_di"], df["minus_di"] = self._calc_adx_full(df, self.adx_period)

        # Choppiness Index
        df["chop"] = self._calc_choppiness(df, self.chop_period)

        # BB 帶寬 (百分比)
        bb_mid = price.rolling(self.range_bb_period).mean()
        bb_std = price.rolling(self.range_bb_period).std()
        df["bb_upper"] = bb_mid + self.range_bb_std * bb_std
        df["bb_lower"] = bb_mid - self.range_bb_std * bb_std
        df["bb_mid"] = bb_mid
        df["bb_width_pct"] = (df["bb_upper"] - df["bb_lower"]) / bb_mid * 100
        df["pct_b"] = (price - df["bb_lower"]) / (df["bb_upper"] - df["bb_lower"])

        # EMA
        df["ema_fast"] = price.ewm(span=self.trend_ema_fast, adjust=False).mean()
        df["ema_slow"] = price.ewm(span=self.trend_ema_slow, adjust=False).mean()
        df["ema_slope"] = (df["ema_fast"] - df["ema_fast"].shift(5)) / df["ema_fast"].shift(5) * 100

        # Donchian Channel (趨勢突破用)
        df["dc_high"] = df["high"].rolling(self.trend_entry_breakout).max()
        df["dc_low"] = df["low"].rolling(self.trend_entry_breakout).min()

        # RSI (震盪用)
        delta = price.diff()
        gain = delta.where(delta > 0, 0.0).ewm(span=self.range_rsi_period, adjust=False).mean()
        loss = (-delta.where(delta < 0, 0.0)).ewm(span=self.range_rsi_period, adjust=False).mean()
        rs = gain / loss.replace(0, np.nan)
        df["rsi"] = 100 - (100 / (1 + rs))

        # 成交量
        df["vol_ma"] = df["volume"].rolling(20).mean()

        # ═══ 狀態判斷 ═══
        df["regime"] = "unknown"  # "trend" / "range" / "unknown"
        self._classify_regime(df)

        # ═══ 信號生成 ═══
        df["signal"] = 0
        df["stop_loss"] = np.nan
        df["take_profit"] = np.nan
        df["signal_mode"] = ""  # "trend" / "range" — 用於回測區分

        warmup = max(self.trend_ema_slow, self.trend_entry_breakout,
                     self.adx_period * 3, self.chop_period) + 20

        for i in range(warmup, len(df)):
            row = df.iloc[i]

            if pd.isna(row["atr"]) or row["atr"] == 0:
                continue

            # 波動率過濾
            if row["atr_pct"] < self.min_atr_pct or row["atr_pct"] > self.max_atr_pct:
                continue

            regime = row["regime"]
            curr_price = row["close"]
            atr = row["atr"]

            if regime == "trend":
                signal, sl, tp = self._trend_signal(df, i)
            elif regime == "range":
                signal, sl, tp = self._range_signal(df, i)
            else:
                continue

            if signal != 0:
                df.iloc[i, df.columns.get_loc("signal")] = signal
                df.iloc[i, df.columns.get_loc("stop_loss")] = sl
                df.iloc[i, df.columns.get_loc("take_profit")] = tp
                df.iloc[i, df.columns.get_loc("signal_mode")] = regime

        return df

    def _classify_regime(self, df: pd.DataFrame):
        """多維度市場狀態分類 + 鎖定期"""
        last_change_idx = -self.regime_lock_bars - 1
        current_regime = "unknown"

        for i in range(len(df)):
            row = df.iloc[i]

            if pd.isna(row.get("adx")) or pd.isna(row.get("chop")):
                df.iloc[i, df.columns.get_loc("regime")] = current_regime
                continue

            # 鎖定期內不切換
            if i - last_change_idx < self.regime_lock_bars:
                df.iloc[i, df.columns.get_loc("regime")] = current_regime
                continue

            # 投票機制: 3 個維度
            trend_votes = 0
            range_votes = 0

            # 1. ADX
            if row["adx"] > self.adx_trend_threshold:
                trend_votes += 1
            elif row["adx"] < self.adx_range_threshold:
                range_votes += 1

            # 2. Choppiness Index
            if row["chop"] < self.chop_trend_threshold:
                trend_votes += 1
            elif row["chop"] > self.chop_range_threshold:
                range_votes += 1

            # 3. BB 帶寬
            if row["bb_width_pct"] > self.bb_width_trend:
                trend_votes += 1
            elif row["bb_width_pct"] < self.bb_width_range:
                range_votes += 1

            # 需要至少 2/3 投票才切換
            new_regime = current_regime
            if trend_votes >= 2:
                new_regime = "trend"
            elif range_votes >= 2:
                new_regime = "range"

            if new_regime != current_regime:
                current_regime = new_regime
                last_change_idx = i

            df.iloc[i, df.columns.get_loc("regime")] = current_regime

    def _trend_signal(self, df: pd.DataFrame, i: int) -> tuple:
        """趨勢模式信號: Donchian 突破 + EMA 確認"""
        row = df.iloc[i]
        prev = df.iloc[i - 1]
        price = row["close"]
        atr = row["atr"]

        # 做多: 收盤突破 Donchian 高點 + EMA 多頭排列
        if (price > prev["dc_high"] and
            row["ema_fast"] > row["ema_slow"] and
            row["plus_di"] > row["minus_di"] and
            row["volume"] > row["vol_ma"] * 0.8):
            sl = price - atr * self.trend_sl_atr
            tp = price + atr * self.trend_sl_atr * 2  # 初始目標, 實際用追蹤
            return 1, sl, tp

        # 做空: 收盤跌破 Donchian 低點 + EMA 空頭排列
        if (price < prev["dc_low"] and
            row["ema_fast"] < row["ema_slow"] and
            row["minus_di"] > row["plus_di"] and
            row["volume"] > row["vol_ma"] * 0.8):
            sl = price + atr * self.trend_sl_atr
            tp = price - atr * self.trend_sl_atr * 2
            return -1, sl, tp

        return 0, 0, 0

    def _range_signal(self, df: pd.DataFrame, i: int) -> tuple:
        """震盪模式信號: RSI + BB 均值回歸"""
        row = df.iloc[i]
        price = row["close"]
        atr = row["atr"]

        # 做多: RSI 超賣 + 價格在 BB 下軌
        if (row["rsi"] <= self.range_rsi_oversold and
            row["pct_b"] <= 0.1 and
            row["volume"] > row["vol_ma"] * 0.7):
            sl = price - atr * self.range_sl_atr
            tp = row["bb_mid"]  # 回歸到中軌
            # 確保 TP 距離至少 0.3 ATR
            if tp - price < atr * 0.3:
                tp = price + atr * self.range_tp_atr
            return 1, sl, tp

        # 做空: RSI 超買 + 價格在 BB 上軌
        if (row["rsi"] >= self.range_rsi_overbought and
            row["pct_b"] >= 0.9 and
            row["volume"] > row["vol_ma"] * 0.7):
            sl = price + atr * self.range_sl_atr
            tp = row["bb_mid"]
            if price - tp < atr * 0.3:
                tp = price - atr * self.range_tp_atr
            return -1, sl, tp

        return 0, 0, 0

    def backtest(self, df, initial_capital=2000.0, max_holding_bars=168):
        """
        覆寫回測: 趨勢模式用追蹤止損, 震盪模式用固定 TP/SL。
        Swing Trading: max_holding_bars = 168 (7天 * 24h)
        """
        df = self.generate_signals(df.copy())
        trades = []
        equity = initial_capital
        equity_curve = []

        pos = None  # {side, entry, qty, sl, tp, atr, mode, entry_idx, entry_time, best_price}

        for i in range(len(df)):
            row = df.iloc[i]
            price = row["close"]
            high = row["high"]
            low = row["low"]

            # ── 持倉處理 ──
            if pos is not None:
                bars_held = i - pos["entry_idx"]
                side = pos["side"]
                mode = pos["mode"]
                atr = pos["atr"]

                # 更新最佳價
                if side == "long":
                    pos["best_price"] = max(pos["best_price"], high)
                else:
                    pos["best_price"] = min(pos["best_price"], low)

                exit_price = None
                exit_reason = None

                if mode == "trend":
                    # 趨勢模式: 追蹤止損
                    if side == "long":
                        trailing = pos["best_price"] - atr * self.trend_trailing_atr
                        pos["sl"] = max(pos["sl"], trailing)
                        if low <= pos["sl"]:
                            exit_price = pos["sl"]
                            exit_reason = "trail_sl"
                    else:
                        trailing = pos["best_price"] + atr * self.trend_trailing_atr
                        pos["sl"] = min(pos["sl"], trailing)
                        if high >= pos["sl"]:
                            exit_price = pos["sl"]
                            exit_reason = "trail_sl"

                    # 趨勢反轉出場: 狀態切回震盪
                    if exit_price is None and row.get("regime") == "range":
                        exit_price = price
                        exit_reason = "regime_switch"

                else:
                    # 震盪模式: 固定 TP/SL
                    if side == "long":
                        if low <= pos["sl"]:
                            exit_price = pos["sl"]
                            exit_reason = "sl"
                        elif high >= pos["tp"]:
                            exit_price = pos["tp"]
                            exit_reason = "tp"
                    else:
                        if high >= pos["sl"]:
                            exit_price = pos["sl"]
                            exit_reason = "sl"
                        elif low <= pos["tp"]:
                            exit_price = pos["tp"]
                            exit_reason = "tp"

                # 超時
                if exit_price is None and bars_held >= max_holding_bars:
                    exit_price = price
                    exit_reason = "timeout"

                # 執行出場
                if exit_price is not None:
                    trade = self._make_trade(pos, exit_price, exit_reason, df.index[i])
                    trades.append(trade)
                    equity += trade.pnl - trade.commission
                    pos = None

            # ── 進場 ──
            if pos is None and row.get("signal", 0) != 0:
                sig = row["signal"]
                side = "long" if sig == 1 else "short"
                qty = (equity * self.max_position_pct) / price

                sl = row.get("stop_loss", 0)
                tp = row.get("take_profit", 0)
                if pd.isna(sl) or sl == 0:
                    sl = price * (0.97 if side == "long" else 1.03)
                if pd.isna(tp) or tp == 0:
                    tp = price * (1.03 if side == "long" else 0.97)

                mode = row.get("signal_mode", "range")

                pos = {
                    "side": side,
                    "entry": price,
                    "qty": qty,
                    "sl": sl,
                    "tp": tp,
                    "atr": row.get("atr", price * 0.01),
                    "mode": mode,
                    "entry_idx": i,
                    "entry_time": df.index[i],
                    "best_price": price,
                }

            # 權益
            mtm = equity
            if pos is not None:
                if pos["side"] == "long":
                    mtm = equity + (price - pos["entry"]) * pos["qty"]
                else:
                    mtm = equity + (pos["entry"] - price) * pos["qty"]
            equity_curve.append(mtm)

        # 強平
        if pos is not None:
            trade = self._make_trade(pos, df.iloc[-1]["close"], "end", df.index[-1])
            trades.append(trade)
            equity += trade.pnl - trade.commission

        equity_series = pd.Series(equity_curve, index=df.index)
        metrics = self._compute_metrics(trades, equity_series, initial_capital)

        # 額外統計: 按模式分類
        trend_trades = [t for t in trades if hasattr(t, '_mode') or True]
        # 用 exit_reason 區分模式
        trend_exits = sum(1 for t in trades if t.exit_reason in ("trail_sl", "regime_switch"))
        range_exits = sum(1 for t in trades if t.exit_reason in ("tp", "sl"))
        metrics["trend_trade_count"] = trend_exits
        metrics["range_trade_count"] = range_exits

        return {"trades": trades, "equity_curve": equity_series, "metrics": metrics}

    def _make_trade(self, pos, exit_price, reason, exit_time):
        side = pos["side"]
        entry = pos["entry"]
        qty = pos["qty"]
        if side == "long":
            pnl = (exit_price - entry) * qty
        else:
            pnl = (entry - exit_price) * qty
        pnl_pct = pnl / (entry * qty) * 100 if entry * qty > 0 else 0
        commission = (entry * qty + exit_price * qty) * (self.commission_rate + self.slippage_rate)
        return TradeRecord(
            entry_time=pos["entry_time"], exit_time=exit_time,
            side=side, entry_price=entry, exit_price=exit_price,
            quantity=qty, pnl=pnl, pnl_pct=pnl_pct,
            commission=commission, exit_reason=reason,
        )

    @staticmethod
    def _calc_adx_full(df, period=14):
        """計算 ADX, +DI, -DI"""
        high = df["high"]
        low = df["low"]
        close = df["close"]

        plus_dm = high.diff()
        minus_dm = -low.diff()
        plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0.0)
        minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0.0)

        high_low = high - low
        high_close = (high - close.shift(1)).abs()
        low_close = (low - close.shift(1)).abs()
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)

        atr = tr.ewm(span=period, adjust=False).mean()
        plus_di = 100 * (plus_dm.ewm(span=period, adjust=False).mean() / atr)
        minus_di = 100 * (minus_dm.ewm(span=period, adjust=False).mean() / atr)

        dx = (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan) * 100
        adx = dx.ewm(span=period, adjust=False).mean()
        return adx, plus_di, minus_di

    @staticmethod
    def _calc_choppiness(df, period=14):
        """
        Choppiness Index (0-100)
        < 38.2 = 趨勢
        > 61.8 = 震盪
        """
        high_low = df["high"] - df["low"]
        high_close = (df["high"] - df["close"].shift(1)).abs()
        low_close = (df["low"] - df["close"].shift(1)).abs()
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)

        tr_sum = tr.rolling(period).sum()
        high_max = df["high"].rolling(period).max()
        low_min = df["low"].rolling(period).min()
        hl_range = high_max - low_min

        chop = 100 * np.log10(tr_sum / hl_range.replace(0, np.nan)) / np.log10(period)
        return chop
