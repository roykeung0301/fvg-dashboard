"""
策略 2: 多時框確認均值回歸策略 (MTF Mean Reversion)

核心邏輯:
- 用高時框 (4h) 確定趨勢方向 (EMA 200)
- 在趨勢方向上，用低時框 (1h) 等待 RSI+Stochastic 極值回歸
- 只做順勢回歸 (上升趨勢中只做多回調，下降趨勢中只做空反彈)

高勝率原因:
- 順勢交易 + 回調入場 = 最穩健的組合
- 三重確認: 趨勢 + RSI + Stochastic
- 止盈目標設得近 (EMA 回歸)
- 逆勢信號完全過濾掉
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from strategies.base_strategy import BaseStrategy


class MTFReversionStrategy(BaseStrategy):

    name = "多時框順勢回歸"
    description = "4h 趨勢方向 + 1h RSI/Stochastic 極值 → 順勢回歸"

    def __init__(
        self,
        trend_ema: int = 200,         # 趨勢 EMA (在 1h 上等於 ~8天)
        fast_ema: int = 50,           # 快 EMA
        rsi_period: int = 10,
        rsi_oversold: float = 20,
        rsi_overbought: float = 80,
        stoch_period: int = 14,
        stoch_smooth: int = 3,
        stoch_oversold: float = 20,
        stoch_overbought: float = 80,
        tp_atr_mult: float = 1.0,     # 止盈 = ATR * mult
        sl_atr_mult: float = 2.0,     # 止損 = ATR * mult
    ):
        self.trend_ema = trend_ema
        self.fast_ema = fast_ema
        self.rsi_period = rsi_period
        self.rsi_oversold = rsi_oversold
        self.rsi_overbought = rsi_overbought
        self.stoch_period = stoch_period
        self.stoch_smooth = stoch_smooth
        self.stoch_oversold = stoch_oversold
        self.stoch_overbought = stoch_overbought
        self.tp_atr_mult = tp_atr_mult
        self.sl_atr_mult = sl_atr_mult

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        # ── EMA 趨勢 ──
        df["ema_trend"] = df["close"].ewm(span=self.trend_ema, adjust=False).mean()
        df["ema_fast"] = df["close"].ewm(span=self.fast_ema, adjust=False).mean()
        df["trend"] = np.where(df["close"] > df["ema_trend"], 1, -1)

        # ── RSI ──
        delta = df["close"].diff()
        gain = delta.where(delta > 0, 0.0).ewm(span=self.rsi_period, adjust=False).mean()
        loss = (-delta.where(delta < 0, 0.0)).ewm(span=self.rsi_period, adjust=False).mean()
        rs = gain / loss.replace(0, np.nan)
        df["rsi"] = 100 - (100 / (1 + rs))

        # ── Stochastic ──
        low_min = df["low"].rolling(self.stoch_period).min()
        high_max = df["high"].rolling(self.stoch_period).max()
        denom = high_max - low_min
        df["stoch_k"] = np.where(denom > 0, (df["close"] - low_min) / denom * 100, 50)
        df["stoch_d"] = pd.Series(df["stoch_k"]).rolling(self.stoch_smooth).mean()

        # ── ATR ──
        high_low = df["high"] - df["low"]
        high_close = (df["high"] - df["close"].shift(1)).abs()
        low_close = (df["low"] - df["close"].shift(1)).abs()
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        df["atr"] = tr.rolling(14).mean()

        # ── 價格與 EMA 的距離 (過度偏離才做) ──
        df["dist_from_ema"] = (df["close"] - df["ema_fast"]) / df["ema_fast"] * 100

        # ── 信號生成 ──
        df["signal"] = 0
        df["stop_loss"] = np.nan
        df["take_profit"] = np.nan

        warmup = max(self.trend_ema, self.stoch_period) + 10

        for i in range(warmup, len(df)):
            row = df.iloc[i]
            prev = df.iloc[i - 1]

            if pd.isna(row["atr"]) or row["atr"] == 0:
                continue

            price = row["close"]
            atr = row["atr"]

            # ── 上升趨勢 + 回調到超賣 → 做多 ──
            if (row["trend"] == 1 and
                row["rsi"] <= self.rsi_oversold and
                row["stoch_k"] <= self.stoch_oversold and
                row["dist_from_ema"] < -0.5):  # 偏離快 EMA 0.5% 以上

                # Stochastic 金叉確認 (K 線穿越 D 線向上)
                if row["stoch_k"] > row["stoch_d"] or prev["stoch_k"] <= prev["stoch_d"]:
                    df.iloc[i, df.columns.get_loc("signal")] = 1
                    df.iloc[i, df.columns.get_loc("stop_loss")] = price - atr * self.sl_atr_mult
                    df.iloc[i, df.columns.get_loc("take_profit")] = price + atr * self.tp_atr_mult

            # ── 下降趨勢 + 反彈到超買 → 做空 ──
            elif (row["trend"] == -1 and
                  row["rsi"] >= self.rsi_overbought and
                  row["stoch_k"] >= self.stoch_overbought and
                  row["dist_from_ema"] > 0.5):

                if row["stoch_k"] < row["stoch_d"] or prev["stoch_k"] >= prev["stoch_d"]:
                    df.iloc[i, df.columns.get_loc("signal")] = -1
                    df.iloc[i, df.columns.get_loc("stop_loss")] = price + atr * self.sl_atr_mult
                    df.iloc[i, df.columns.get_loc("take_profit")] = price - atr * self.tp_atr_mult

        return df
