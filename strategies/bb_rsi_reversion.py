"""
策略 1: BB + RSI 均值回歸策略

核心邏輯:
- 價格觸碰布林帶下軌 + RSI 超賣 → 做多 (價格會回歸均值)
- 價格觸碰布林帶上軌 + RSI 超買 → 做空 (價格會回歸均值)
- 止盈設在布林帶中軌 (均值)
- 止損設在布林帶外側一定距離

高勝率原因:
- 均值回歸是統計上最穩健的現象
- 雙重確認 (BB + RSI) 過濾假信號
- 止盈目標小 (只吃到中軌)，大部分時候能達到
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from strategies.base_strategy import BaseStrategy


class BBRSIReversionStrategy(BaseStrategy):

    name = "BB+RSI 均值回歸"
    description = "布林帶極值 + RSI 超賣/超買 → 反向回歸至中軌"

    def __init__(
        self,
        bb_period: int = 20,
        bb_std: float = 2.2,
        rsi_period: int = 14,
        rsi_oversold: float = 25,
        rsi_overbought: float = 75,
        sl_multiplier: float = 1.2,    # 止損距離 = BB 寬度 * multiplier
        tp_target: str = "middle",     # "middle" = BB 中軌
        volume_filter: bool = True,    # 成交量確認
        atr_filter: bool = True,       # ATR 波動率過濾 (避免低波動)
    ):
        self.bb_period = bb_period
        self.bb_std = bb_std
        self.rsi_period = rsi_period
        self.rsi_oversold = rsi_oversold
        self.rsi_overbought = rsi_overbought
        self.sl_multiplier = sl_multiplier
        self.tp_target = tp_target
        self.volume_filter = volume_filter
        self.atr_filter = atr_filter

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        # ── 布林帶 ──
        df["bb_middle"] = df["close"].rolling(self.bb_period).mean()
        df["bb_std"] = df["close"].rolling(self.bb_period).std()
        df["bb_upper"] = df["bb_middle"] + self.bb_std * df["bb_std"]
        df["bb_lower"] = df["bb_middle"] - self.bb_std * df["bb_std"]
        df["bb_width"] = df["bb_upper"] - df["bb_lower"]

        # %B 指標 (價格在布林帶中的相對位置)
        df["pct_b"] = (df["close"] - df["bb_lower"]) / df["bb_width"]

        # ── RSI ──
        delta = df["close"].diff()
        gain = delta.where(delta > 0, 0.0).rolling(self.rsi_period).mean()
        loss = (-delta.where(delta < 0, 0.0)).rolling(self.rsi_period).mean()
        rs = gain / loss.replace(0, np.nan)
        df["rsi"] = 100 - (100 / (1 + rs))

        # ── ATR (波動率過濾) ──
        high_low = df["high"] - df["low"]
        high_close = (df["high"] - df["close"].shift(1)).abs()
        low_close = (df["low"] - df["close"].shift(1)).abs()
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        df["atr"] = tr.rolling(14).mean()
        df["atr_pct"] = df["atr"] / df["close"] * 100  # ATR 佔價格百分比

        # ── 成交量 MA ──
        df["vol_ma"] = df["volume"].rolling(20).mean()

        # ── 信號生成 ──
        df["signal"] = 0
        df["stop_loss"] = np.nan
        df["take_profit"] = np.nan

        for i in range(max(self.bb_period, self.rsi_period) + 5, len(df)):
            row = df.iloc[i]

            # 跳過無效數據
            if pd.isna(row["bb_middle"]) or pd.isna(row["rsi"]):
                continue

            # 成交量過濾: 必須高於均量
            if self.volume_filter and row["volume"] < row["vol_ma"] * 0.8:
                continue

            # ATR 過濾: 波動率太低不做 (均值回歸空間不夠)
            if self.atr_filter and row["atr_pct"] < 0.3:
                continue

            # ── 做多信號: 價格觸碰/跌破下軌 + RSI 超賣 ──
            if row["close"] <= row["bb_lower"] and row["rsi"] <= self.rsi_oversold:
                df.iloc[i, df.columns.get_loc("signal")] = 1
                # 止損: 下軌再往下
                sl = row["close"] - row["bb_width"] * self.sl_multiplier * 0.3
                # 止盈: 中軌 (均值)
                tp = row["bb_middle"]
                df.iloc[i, df.columns.get_loc("stop_loss")] = sl
                df.iloc[i, df.columns.get_loc("take_profit")] = tp

            # ── 做空信號: 價格觸碰/突破上軌 + RSI 超買 ──
            elif row["close"] >= row["bb_upper"] and row["rsi"] >= self.rsi_overbought:
                df.iloc[i, df.columns.get_loc("signal")] = -1
                sl = row["close"] + row["bb_width"] * self.sl_multiplier * 0.3
                tp = row["bb_middle"]
                df.iloc[i, df.columns.get_loc("stop_loss")] = sl
                df.iloc[i, df.columns.get_loc("take_profit")] = tp

        return df
