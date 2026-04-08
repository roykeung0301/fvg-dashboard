"""
FVCL3 V2 — 趨勢跟蹤 Long-Only

核心: 讓利潤奔跑，截斷虧損
- 無固定 TP — 用趨勢反轉 + trailing stop 出場
- 寬 SL 避免被正常回調掃止損
- 日線定方向 → 4h 回調確認 → 1h 入場
- 持倉最多 30 天 (趨勢交易)
"""

from __future__ import annotations

from typing import Optional, List, Dict

import numpy as np
import pandas as pd

from strategies.base_strategy import BaseStrategy, TradeRecord


class FVCL3V2Strategy(BaseStrategy):

    name = "FVCL3 V2 Trend Follow"
    description = "Long-only 趨勢跟蹤 + Trailing Stop"

    commission_rate = 0.0004
    slippage_rate = 0.0001
    max_position_pct = 0.95

    def __init__(
        self,
        # 日線
        daily_ema_fast: int = 20,
        daily_ema_slow: int = 50,

        # 4h 回調
        h4_rsi_period: int = 14,
        h4_rsi_pullback: float = 45,

        # 1h 入場
        h1_rsi_period: int = 7,
        h1_rsi_os: float = 35,
        entry_min_score: int = 2,

        # 風控
        sl_atr: float = 4.0,            # 初始 SL (寬)
        trail_start_atr: float = 3.0,   # 盈利 N*ATR 後開始 trail
        trail_step_atr: float = 2.5,    # trail 距離 (寬)
        breakeven_atr: float = 2.0,     # 盈利 N*ATR 後移至打和

        # 共用
        atr_period: int = 14,
        cooldown: int = 12,             # 冷卻 (12h)
        max_hold: int = 720,            # 最大持倉 30 天
    ):
        self.daily_ema_fast = daily_ema_fast
        self.daily_ema_slow = daily_ema_slow
        self.h4_rsi_period = h4_rsi_period
        self.h4_rsi_pullback = h4_rsi_pullback
        self.h1_rsi_period = h1_rsi_period
        self.h1_rsi_os = h1_rsi_os
        self.entry_min_score = entry_min_score
        self.sl_atr = sl_atr
        self.trail_start_atr = trail_start_atr
        self.trail_step_atr = trail_step_atr
        self.breakeven_atr = breakeven_atr
        self.atr_period = atr_period
        self.cooldown = cooldown
        self.max_hold = max_hold

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        price = df["close"]

        # ── 1h 指標 ──
        hl = df["high"] - df["low"]
        hc = (df["high"] - price.shift(1)).abs()
        lc = (df["low"] - price.shift(1)).abs()
        tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
        df["atr"] = tr.rolling(self.atr_period).mean()

        # 1h RSI
        delta = price.diff()
        g = delta.where(delta > 0, 0.0).ewm(span=self.h1_rsi_period, adjust=False).mean()
        l = (-delta.where(delta < 0, 0.0)).ewm(span=self.h1_rsi_period, adjust=False).mean()
        df["rsi_1h"] = 100 - (100 / (1 + g / l.replace(0, np.nan)))

        # 1h BB
        df["bb_mid"] = price.rolling(20).mean()
        std = price.rolling(20).std()
        df["bb_lower"] = df["bb_mid"] - 2.0 * std
        df["pct_b"] = (price - df["bb_lower"]) / (4.0 * std).replace(0, np.nan)

        # 成交量
        df["vol_ma"] = df["volume"].rolling(20).mean()

        # K 線
        body = (df["close"] - df["open"]).abs()
        full_range = df["high"] - df["low"]
        lower_shadow = pd.concat([df["open"], df["close"]], axis=1).min(axis=1) - df["low"]
        df["hammer"] = ((lower_shadow > body * 1.5) & (full_range > 0)).astype(int)
        df["bull_close"] = (df["close"] > df["open"]).astype(int)
        df["engulf_bull"] = (
            (df["close"].shift(1) < df["open"].shift(1)) &
            (df["close"] > df["open"]) &
            (df["close"] > df["open"].shift(1)) &
            (df["open"] < df["close"].shift(1))
        ).astype(int)

        # ── 4h RSI ──
        df_4h = df.resample("4h").agg({
            "open": "first", "high": "max", "low": "min",
            "close": "last", "volume": "sum",
        }).dropna()
        d4 = df_4h["close"].diff()
        g4 = d4.where(d4 > 0, 0.0).ewm(span=self.h4_rsi_period, adjust=False).mean()
        l4 = (-d4.where(d4 < 0, 0.0)).ewm(span=self.h4_rsi_period, adjust=False).mean()
        df_4h["rsi_4h"] = 100 - (100 / (1 + g4 / l4.replace(0, np.nan)))
        df["rsi_4h"] = df_4h["rsi_4h"].reindex(df.index, method="ffill")

        # ── 日線趨勢 ──
        df_d = df.resample("1D").agg({
            "open": "first", "high": "max", "low": "min",
            "close": "last", "volume": "sum",
        }).dropna()
        df_d["ema_fast"] = df_d["close"].ewm(span=self.daily_ema_fast, adjust=False).mean()
        df_d["ema_slow"] = df_d["close"].ewm(span=self.daily_ema_slow, adjust=False).mean()
        df_d["daily_trend"] = 0
        df_d.loc[df_d["ema_fast"] > df_d["ema_slow"], "daily_trend"] = 1
        df_d.loc[df_d["ema_fast"] < df_d["ema_slow"], "daily_trend"] = -1
        df["daily_trend"] = df_d["daily_trend"].reindex(df.index, method="ffill")

        # ── 信號 ──
        df["signal"] = 0
        df["stop_loss"] = np.nan
        df["take_profit"] = np.nan  # 不使用固定 TP

        warmup = 300
        last_signal = -self.cooldown - 1

        for i in range(warmup, len(df)):
            row = df.iloc[i]
            if pd.isna(row["atr"]) or row["atr"] == 0:
                continue
            if i - last_signal < self.cooldown:
                continue

            daily = row.get("daily_trend", 0)
            rsi_4h = row.get("rsi_4h", 50)

            if pd.isna(daily) or daily != 1:
                continue
            if pd.isna(rsi_4h) or rsi_4h > self.h4_rsi_pullback:
                continue

            p = row["close"]
            atr = row["atr"]

            # 1h 入場投票
            score = 0
            if row["rsi_1h"] <= self.h1_rsi_os:
                score += 1
            if row["pct_b"] <= 0.3:
                score += 1
            if row["hammer"] or row["engulf_bull"]:
                score += 1
            if row["bull_close"]:
                score += 1
            if row["volume"] > row["vol_ma"] * 0.7:
                score += 1

            if score >= self.entry_min_score:
                sl = p - atr * self.sl_atr
                df.iloc[i, df.columns.get_loc("signal")] = 1
                df.iloc[i, df.columns.get_loc("stop_loss")] = sl
                # TP 留空 — 由趨勢反轉 + trailing 出場
                last_signal = i

        return df

    def backtest(
        self,
        df: pd.DataFrame,
        initial_capital: float = 2000.0,
        max_holding_bars: int = 168,  # 被 self.max_hold 覆蓋
    ) -> Dict:
        """
        趨勢跟蹤回測:
        - 出場: SL / trailing SL / 日線趨勢反轉 / 超時
        - Trailing: bar 結束更新，下一 bar 生效
        """
        df = self.generate_signals(df.copy())
        trades: List[TradeRecord] = []
        equity = initial_capital
        equity_curve = []
        position = None

        for i in range(len(df)):
            row = df.iloc[i]
            price = row["close"]
            high = row["high"]
            low = row["low"]

            if position is not None:
                bars_held = i - position["entry_idx"]
                exit_price = None
                exit_reason = None
                entry = position["entry_price"]
                atr = position["atr"]

                # SL 檢查 (用上一 bar 更新的 SL)
                if low <= position["sl"]:
                    exit_price = max(position["sl"], low)  # 不低於 bar low
                    exit_reason = "sl"

                # 日線趨勢反轉出場
                if exit_price is None:
                    daily = row.get("daily_trend", 1)
                    if not pd.isna(daily) and daily != 1:
                        exit_price = price
                        exit_reason = "trend_rev"

                # 超時
                if exit_price is None and bars_held >= self.max_hold:
                    exit_price = price
                    exit_reason = "timeout"

                if exit_price is not None:
                    trade = self._close_position(position, exit_price, exit_reason, df.index[i])
                    trades.append(trade)
                    equity += trade.pnl - trade.commission
                    position = None
                else:
                    # Bar 結束更新 trailing (下一 bar 生效)
                    max_p = position.get("max_price", entry)
                    if high > max_p:
                        max_p = high
                        position["max_price"] = max_p

                    profit_atr = (max_p - entry) / atr if atr > 0 else 0

                    if profit_atr >= self.breakeven_atr and position["sl"] < entry:
                        position["sl"] = entry + atr * 0.05

                    if profit_atr >= self.trail_start_atr:
                        new_sl = max_p - atr * self.trail_step_atr
                        if new_sl > position["sl"]:
                            position["sl"] = new_sl

            # ── 進場 ──
            if position is None and row.get("signal", 0) == 1:
                sl = row.get("stop_loss", 0)
                atr_val = row.get("atr", price * 0.02)
                if pd.isna(sl) or sl == 0:
                    sl = price * 0.92
                if pd.isna(atr_val):
                    atr_val = price * 0.02

                quantity = (equity * self.max_position_pct) / price
                if equity > 1 and quantity > 0:
                    position = {
                        "side": "long",
                        "entry_price": price,
                        "quantity": quantity,
                        "entry_idx": i,
                        "entry_time": df.index[i],
                        "sl": sl,
                        "tp": price * 10,  # 極高 TP (幾乎不會觸發)
                        "atr": atr_val,
                        "max_price": price,
                    }

            mtm = equity
            if position is not None:
                unrealized = (price - position["entry_price"]) * position["quantity"]
                mtm = equity + unrealized
            equity_curve.append(mtm)

        if position is not None:
            trade = self._close_position(position, df.iloc[-1]["close"], "end", df.index[-1])
            trades.append(trade)
            equity += trade.pnl - trade.commission

        equity_series = pd.Series(equity_curve, index=df.index)
        metrics = self._compute_metrics(trades, equity_series, initial_capital)

        return {
            "trades": trades,
            "equity_curve": equity_series,
            "metrics": metrics,
        }

    @staticmethod
    def _calc_adx(df, period=14):
        high, low, close = df["high"], df["low"], df["close"]
        plus_dm = high.diff()
        minus_dm = -low.diff()
        plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0.0)
        minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0.0)
        hl = high - low
        hc = (high - close.shift(1)).abs()
        lc = (low - close.shift(1)).abs()
        tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
        atr = tr.ewm(span=period, adjust=False).mean()
        pdi = 100 * plus_dm.ewm(span=period, adjust=False).mean() / atr
        mdi = 100 * minus_dm.ewm(span=period, adjust=False).mean() / atr
        dx = (pdi - mdi).abs() / (pdi + mdi).replace(0, np.nan) * 100
        adx = dx.ewm(span=period, adjust=False).mean()
        return adx, pdi, mdi
