"""
策略 3: 資金費率 + VWAP 反向策略

核心邏輯:
- 當市場情緒極端偏多 (資金費率很高) → 做空回歸
- 當市場情緒極端偏空 (資金費率很低) → 做多回歸
- VWAP 作為公允價值錨點
- RSI 背離確認反轉

注意: 由於歷史資金費率數據取得限制，
此策略在回測中模擬資金費率 (用 RSI + 成交量偏差代替)。
實盤時使用真實 Binance Funding Rate。

高勝率原因:
- 極端情緒總會修正 (均值回歸)
- 資金費率高 = 做多方要付錢，自然平倉壓力大
- VWAP 偏離是機構常用的回歸信號
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from strategies.base_strategy import BaseStrategy


class FundingContrarianStrategy(BaseStrategy):

    name = "情緒反向 + VWAP"
    description = "模擬資金費率極端 + VWAP 偏離 → 反向回歸"

    def __init__(
        self,
        vwap_period: int = 24,             # VWAP 計算週期 (24h rolling)
        vwap_dev_threshold: float = 1.5,   # VWAP 偏離閾值 (標準差倍數)
        rsi_period: int = 14,
        rsi_extreme_low: float = 20,
        rsi_extreme_high: float = 80,
        sentiment_lookback: int = 48,      # 情緒計算回看期
        sentiment_threshold: float = 0.7,  # 情緒極值閾值 (0-1)
        tp_pct: float = 0.8,              # 止盈百分比
        sl_pct: float = 1.5,              # 止損百分比
        cooldown: int = 4,                 # 信號冷卻期 (K 線數)
    ):
        self.vwap_period = vwap_period
        self.vwap_dev_threshold = vwap_dev_threshold
        self.rsi_period = rsi_period
        self.rsi_extreme_low = rsi_extreme_low
        self.rsi_extreme_high = rsi_extreme_high
        self.sentiment_lookback = sentiment_lookback
        self.sentiment_threshold = sentiment_threshold
        self.tp_pct = tp_pct
        self.sl_pct = sl_pct
        self.cooldown = cooldown

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        # ── Rolling VWAP ──
        typical_price = (df["high"] + df["low"] + df["close"]) / 3
        tp_volume = typical_price * df["volume"]
        df["vwap"] = (
            tp_volume.rolling(self.vwap_period).sum() /
            df["volume"].rolling(self.vwap_period).sum()
        )
        df["vwap_std"] = typical_price.rolling(self.vwap_period).std()
        df["vwap_dev"] = (df["close"] - df["vwap"]) / df["vwap_std"]

        # ── RSI ──
        delta = df["close"].diff()
        gain = delta.where(delta > 0, 0.0).rolling(self.rsi_period).mean()
        loss = (-delta.where(delta < 0, 0.0)).rolling(self.rsi_period).mean()
        rs = gain / loss.replace(0, np.nan)
        df["rsi"] = 100 - (100 / (1 + rs))

        # ── 模擬市場情緒分數 (0~1, 0.5=中性) ──
        # 使用: RSI 標準化 + 成交量偏離 + 價格動量
        rsi_norm = df["rsi"] / 100  # 0~1

        vol_ratio = df["volume"] / df["volume"].rolling(self.sentiment_lookback).mean()
        vol_norm = vol_ratio.clip(0, 3) / 3  # 正規化到 0~1

        momentum = df["close"].pct_change(12)  # 12h 動量
        mom_norm = (momentum.rank(pct=True))  # 百分位排名 0~1

        # 綜合情緒分數
        df["sentiment"] = (rsi_norm * 0.4 + vol_norm * 0.2 + mom_norm * 0.4)

        # ── EMA 作為趨勢過濾 ──
        df["ema_50"] = df["close"].ewm(span=50, adjust=False).mean()

        # ── 信號生成 ──
        df["signal"] = 0
        df["stop_loss"] = np.nan
        df["take_profit"] = np.nan

        warmup = max(self.vwap_period, self.sentiment_lookback, 50) + 10
        last_signal_idx = -self.cooldown - 1

        for i in range(warmup, len(df)):
            row = df.iloc[i]

            if pd.isna(row["vwap"]) or pd.isna(row["sentiment"]):
                continue

            # 冷卻期
            if i - last_signal_idx < self.cooldown:
                continue

            price = row["close"]

            # ── 做多: 情緒極度悲觀 + 價格低於 VWAP + RSI 超賣 ──
            if (row["sentiment"] < (1 - self.sentiment_threshold) and
                row["vwap_dev"] < -self.vwap_dev_threshold and
                row["rsi"] < self.rsi_extreme_low):

                df.iloc[i, df.columns.get_loc("signal")] = 1
                df.iloc[i, df.columns.get_loc("stop_loss")] = price * (1 - self.sl_pct / 100)
                df.iloc[i, df.columns.get_loc("take_profit")] = price * (1 + self.tp_pct / 100)
                last_signal_idx = i

            # ── 做空: 情緒極度樂觀 + 價格高於 VWAP + RSI 超買 ──
            elif (row["sentiment"] > self.sentiment_threshold and
                  row["vwap_dev"] > self.vwap_dev_threshold and
                  row["rsi"] > self.rsi_extreme_high):

                df.iloc[i, df.columns.get_loc("signal")] = -1
                df.iloc[i, df.columns.get_loc("stop_loss")] = price * (1 + self.sl_pct / 100)
                df.iloc[i, df.columns.get_loc("take_profit")] = price * (1 - self.tp_pct / 100)
                last_signal_idx = i

        return df
