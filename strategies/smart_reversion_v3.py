"""
策略 V3: 智能均值回歸 — 解決高勝率陷阱

V2 的問題: 小 TP / 大 SL → 一次虧損吃掉多次獲利
V3 解決方案:
1. 分段止盈 (50% 先走, 50% 追蹤)
2. 保本止損 (達到 1R 後移動止損到保本)
3. 追蹤止損 (用 ATR 動態追蹤)
4. 市場狀態自適應 (震盪/趨勢區分)
5. 加入均值回歸強度評分 (不是所有極值都值得做)
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from strategies.base_strategy import BaseStrategy, TradeRecord


class SmartReversionV3(BaseStrategy):

    name = "智能均值回歸 V3"
    description = "分段止盈 + 保本止損 + 市場自適應"

    commission_rate = 0.0003  # 掛單 + Taker 混合
    slippage_rate = 0.0001

    def __init__(
        self,
        # RSI
        rsi_period: int = 7,
        rsi_extreme_low: float = 15,
        rsi_extreme_high: float = 85,
        # BB
        bb_period: int = 20,
        bb_std: float = 2.5,
        # TP/SL (ATR 倍數)
        tp1_atr: float = 0.5,      # 第一段止盈
        tp2_atr: float = 1.5,      # 第二段止盈目標
        sl_atr: float = 2.5,       # 初始止損
        breakeven_at: float = 0.4,  # 到達此 ATR 倍數時移到保本
        trailing_atr: float = 1.0,  # 追蹤止損距離
        # 倉位分配
        tp1_pct: float = 0.6,      # 第一段出 60%
        # 過濾
        min_confirms: int = 3,
        adx_period: int = 14,
        adx_max: float = 30,       # ADX > 30 = 強趨勢，不做均值回歸
        min_atr_pct: float = 0.25,
        max_atr_pct: float = 2.5,
    ):
        self.rsi_period = rsi_period
        self.rsi_extreme_low = rsi_extreme_low
        self.rsi_extreme_high = rsi_extreme_high
        self.bb_period = bb_period
        self.bb_std = bb_std
        self.tp1_atr = tp1_atr
        self.tp2_atr = tp2_atr
        self.sl_atr = sl_atr
        self.breakeven_at = breakeven_at
        self.trailing_atr = trailing_atr
        self.tp1_pct = tp1_pct
        self.min_confirms = min_confirms
        self.adx_period = adx_period
        self.adx_max = adx_max
        self.min_atr_pct = min_atr_pct
        self.max_atr_pct = max_atr_pct

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """計算指標並生成信號"""
        price = df["close"]

        # RSI
        delta = price.diff()
        gain = delta.where(delta > 0, 0.0).ewm(span=self.rsi_period, adjust=False).mean()
        loss = (-delta.where(delta < 0, 0.0)).ewm(span=self.rsi_period, adjust=False).mean()
        rs = gain / loss.replace(0, np.nan)
        df["rsi"] = 100 - (100 / (1 + rs))

        # Stochastic RSI
        rsi = df["rsi"]
        rsi_min = rsi.rolling(14).min()
        rsi_max = rsi.rolling(14).max()
        rsi_range = rsi_max - rsi_min
        df["stoch_rsi"] = np.where(rsi_range > 0, (rsi - rsi_min) / rsi_range * 100, 50)

        # BB
        df["bb_mid"] = price.rolling(self.bb_period).mean()
        std = price.rolling(self.bb_period).std()
        df["bb_upper"] = df["bb_mid"] + self.bb_std * std
        df["bb_lower"] = df["bb_mid"] - self.bb_std * std
        df["pct_b"] = (price - df["bb_lower"]) / (df["bb_upper"] - df["bb_lower"])

        # ATR
        high_low = df["high"] - df["low"]
        high_close = (df["high"] - price.shift(1)).abs()
        low_close = (df["low"] - price.shift(1)).abs()
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        df["atr"] = tr.rolling(14).mean()
        df["atr_pct"] = df["atr"] / price * 100

        # ADX (趨勢強度)
        df["adx"] = self._calc_adx(df, self.adx_period)

        # VWAP
        typical = (df["high"] + df["low"] + price) / 3
        tp_vol = typical * df["volume"]
        df["vwap"] = tp_vol.rolling(24).sum() / df["volume"].rolling(24).sum()
        df["vwap_dist"] = (price - df["vwap"]) / df["vwap"] * 100

        # 成交量
        df["vol_ma"] = df["volume"].rolling(20).mean()

        # 前K線
        df["prev_bearish"] = (df["close"].shift(1) < df["open"].shift(1)).astype(int)
        df["prev_bullish"] = (df["close"].shift(1) > df["open"].shift(1)).astype(int)

        # 信號
        df["signal"] = 0
        df["stop_loss"] = np.nan
        df["take_profit"] = np.nan

        warmup = max(200, self.bb_period, self.adx_period * 3) + 20

        for i in range(warmup, len(df)):
            row = df.iloc[i]

            if pd.isna(row["atr"]) or row["atr"] == 0 or pd.isna(row["adx"]):
                continue

            # 波動率過濾
            if row["atr_pct"] < self.min_atr_pct or row["atr_pct"] > self.max_atr_pct:
                continue

            # 趨勢過濾: ADX 太高 = 強趨勢，均值回歸不適用
            if row["adx"] > self.adx_max:
                continue

            # 成交量
            if row["volume"] < row["vol_ma"] * 0.7:
                continue

            curr_price = row["close"]
            atr = row["atr"]

            # 做多確認
            long_conf = 0
            if row["rsi"] <= self.rsi_extreme_low:
                long_conf += 1
            if row["stoch_rsi"] <= 10:
                long_conf += 1
            if row["pct_b"] <= 0.05:
                long_conf += 1
            if row["vwap_dist"] < -0.3:
                long_conf += 1
            if row["prev_bearish"]:
                long_conf += 1

            # 做空確認
            short_conf = 0
            if row["rsi"] >= self.rsi_extreme_high:
                short_conf += 1
            if row["stoch_rsi"] >= 90:
                short_conf += 1
            if row["pct_b"] >= 0.95:
                short_conf += 1
            if row["vwap_dist"] > 0.3:
                short_conf += 1
            if row["prev_bullish"]:
                short_conf += 1

            if long_conf >= self.min_confirms:
                df.iloc[i, df.columns.get_loc("signal")] = 1
                df.iloc[i, df.columns.get_loc("stop_loss")] = curr_price - atr * self.sl_atr
                df.iloc[i, df.columns.get_loc("take_profit")] = curr_price + atr * self.tp1_atr
            elif short_conf >= self.min_confirms:
                df.iloc[i, df.columns.get_loc("signal")] = -1
                df.iloc[i, df.columns.get_loc("stop_loss")] = curr_price + atr * self.sl_atr
                df.iloc[i, df.columns.get_loc("take_profit")] = curr_price - atr * self.tp1_atr

        return df

    def backtest(self, df, initial_capital=2000.0, max_holding_bars=24):
        """覆寫回測: 實現分段止盈 + 保本止損 + 追蹤止損"""
        df = self.generate_signals(df.copy())
        trades = []
        equity = initial_capital
        equity_curve = []

        pos = None  # {side, entry, qty, sl, tp1, tp2, atr, entry_idx, entry_time, phase, best_price}
        # phase: 0=全倉, 1=已出第一段(剩餘倉位追蹤)

        for i in range(len(df)):
            row = df.iloc[i]
            price = row["close"]
            high = row["high"]
            low = row["low"]

            if pos is not None:
                bars_held = i - pos["entry_idx"]
                side = pos["side"]
                atr = pos["atr"]

                # 更新最佳價格 (追蹤用)
                if side == "long":
                    pos["best_price"] = max(pos["best_price"], high)
                else:
                    pos["best_price"] = min(pos["best_price"], low)

                # ── Phase 0: 全倉，等第一段止盈或止損 ──
                if pos["phase"] == 0:
                    hit_tp1 = (side == "long" and high >= pos["tp1"]) or \
                              (side == "short" and low <= pos["tp1"])
                    hit_sl = (side == "long" and low <= pos["sl"]) or \
                             (side == "short" and high >= pos["sl"])
                    timeout = bars_held >= max_holding_bars

                    if hit_sl:
                        # 全倉止損
                        exit_p = pos["sl"]
                        trade = self._make_trade(pos, exit_p, "sl", df.index[i], pos["qty"])
                        trades.append(trade)
                        equity += trade.pnl - trade.commission
                        pos = None

                    elif hit_tp1:
                        # 出第一段 (tp1_pct)
                        exit_qty = pos["qty"] * self.tp1_pct
                        trade = self._make_trade(pos, pos["tp1"], "tp1", df.index[i], exit_qty)
                        trades.append(trade)
                        equity += trade.pnl - trade.commission

                        # 剩餘倉位: 移到保本止損 + 追蹤
                        pos["qty"] -= exit_qty
                        pos["sl"] = pos["entry"]  # 保本
                        pos["phase"] = 1
                        pos["best_price"] = pos["tp1"]

                    elif timeout:
                        trade = self._make_trade(pos, price, "timeout", df.index[i], pos["qty"])
                        trades.append(trade)
                        equity += trade.pnl - trade.commission
                        pos = None

                # ── Phase 1: 剩餘倉位追蹤 ──
                elif pos["phase"] == 1:
                    # 追蹤止損
                    if side == "long":
                        trailing_sl = pos["best_price"] - atr * self.trailing_atr
                        pos["sl"] = max(pos["sl"], trailing_sl)
                        hit_sl = low <= pos["sl"]
                        hit_tp2 = high >= pos["entry"] + atr * self.tp2_atr
                    else:
                        trailing_sl = pos["best_price"] + atr * self.trailing_atr
                        pos["sl"] = min(pos["sl"], trailing_sl)
                        hit_sl = high >= pos["sl"]
                        hit_tp2 = low <= pos["entry"] - atr * self.tp2_atr

                    timeout2 = bars_held >= max_holding_bars * 2

                    if hit_tp2:
                        exit_p = pos["entry"] + atr * self.tp2_atr if side == "long" \
                                 else pos["entry"] - atr * self.tp2_atr
                        trade = self._make_trade(pos, exit_p, "tp2", df.index[i], pos["qty"])
                        trades.append(trade)
                        equity += trade.pnl - trade.commission
                        pos = None
                    elif hit_sl:
                        trade = self._make_trade(pos, pos["sl"], "trail_sl", df.index[i], pos["qty"])
                        trades.append(trade)
                        equity += trade.pnl - trade.commission
                        pos = None
                    elif timeout2:
                        trade = self._make_trade(pos, price, "timeout", df.index[i], pos["qty"])
                        trades.append(trade)
                        equity += trade.pnl - trade.commission
                        pos = None

            # ── 進場 ──
            if pos is None and row.get("signal", 0) != 0:
                sig = row["signal"]
                side = "long" if sig == 1 else "short"
                qty = (equity * self.max_position_pct) / price
                atr_val = row.get("atr", price * 0.01)

                sl = row.get("stop_loss", 0)
                tp1 = row.get("take_profit", 0)

                if pd.isna(sl) or sl == 0:
                    sl = price - atr_val * self.sl_atr if side == "long" else price + atr_val * self.sl_atr
                if pd.isna(tp1) or tp1 == 0:
                    tp1 = price + atr_val * self.tp1_atr if side == "long" else price - atr_val * self.tp1_atr

                pos = {
                    "side": side,
                    "entry": price,
                    "qty": qty,
                    "sl": sl,
                    "tp1": tp1,
                    "atr": atr_val,
                    "entry_idx": i,
                    "entry_time": df.index[i],
                    "phase": 0,
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
            trade = self._make_trade(pos, df.iloc[-1]["close"], "end", df.index[-1], pos["qty"])
            trades.append(trade)
            equity += trade.pnl - trade.commission

        equity_series = pd.Series(equity_curve, index=df.index)
        metrics = self._compute_metrics(trades, equity_series, initial_capital)
        return {"trades": trades, "equity_curve": equity_series, "metrics": metrics}

    def _make_trade(self, pos, exit_price, reason, exit_time, qty):
        side = pos["side"]
        entry = pos["entry"]
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
    def _calc_adx(df, period=14):
        """計算 ADX"""
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

        atr = tr.rolling(period).mean()
        plus_di = 100 * (plus_dm.rolling(period).mean() / atr)
        minus_di = 100 * (minus_dm.rolling(period).mean() / atr)

        dx = (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan) * 100
        adx = dx.rolling(period).mean()
        return adx
