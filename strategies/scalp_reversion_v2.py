"""
策略 V2: 高勝率微利回歸 (Scalp Mean Reversion)

設計原則 — 針對 85%+ 勝率:
1. 極嚴格的入場條件 (多重確認)
2. 小止盈 (快速獲利離場)
3. 寬止損 (給價格回歸的空間)
4. 波動率自適應 TP/SL
5. 趨勢過濾 (不在強趨勢中逆勢)
6. 時段過濾 (避開低流動性時段)

子策略:
A) 極端 RSI 回歸 — RSI 觸及極值必回
B) BB 擠壓回歸 — 布林帶縮窄後的突破回歸
C) VWAP 磁吸 — 偏離 VWAP 後回歸
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from strategies.base_strategy import BaseStrategy


class ScalpReversionV2(BaseStrategy):

    name = "高勝率微利回歸 V2"
    description = "極嚴格入場 + 小TP/寬SL + 波動率自適應"

    # 降低手續費影響: 用 Maker 掛單
    commission_rate = 0.0002  # Maker 0.02% 來回
    slippage_rate = 0.00005

    def __init__(
        self,
        # RSI 參數
        rsi_period: int = 7,           # 短週期 RSI 更敏感
        rsi_extreme_low: float = 15,   # 極端超賣
        rsi_extreme_high: float = 85,  # 極端超買

        # BB 參數
        bb_period: int = 20,
        bb_std: float = 2.5,           # 更寬的帶 = 更極端才觸發

        # TP/SL (ATR 倍數)
        tp_atr_mult: float = 0.5,     # 小止盈: 0.5x ATR
        sl_atr_mult: float = 3.0,     # 寬止損: 3x ATR

        # 過濾器
        trend_ema: int = 100,          # 趨勢 EMA
        trend_strength_threshold: float = 2.0,  # 趨勢太強不做 (ATR%)
        min_atr_pct: float = 0.2,     # 最低波動率
        max_atr_pct: float = 3.0,     # 最高波動率 (太劇烈不做)
        volume_mult: float = 0.7,     # 成交量 > MA * mult

        # 多重確認數量
        min_confirmations: int = 3,    # 至少 N 個條件同時滿足

        # 連續虧損保護
        max_consecutive_losses: int = 3,  # 連虧 N 次暫停
        cooldown_after_loss: int = 6,     # 連虧後冷卻 K 線數
    ):
        self.rsi_period = rsi_period
        self.rsi_extreme_low = rsi_extreme_low
        self.rsi_extreme_high = rsi_extreme_high
        self.bb_period = bb_period
        self.bb_std = bb_std
        self.tp_atr_mult = tp_atr_mult
        self.sl_atr_mult = sl_atr_mult
        self.trend_ema = trend_ema
        self.trend_strength_threshold = trend_strength_threshold
        self.min_atr_pct = min_atr_pct
        self.max_atr_pct = max_atr_pct
        self.volume_mult = volume_mult
        self.min_confirmations = min_confirmations
        self.max_consecutive_losses = max_consecutive_losses
        self.cooldown_after_loss = cooldown_after_loss

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        price = df["close"]

        # ── 技術指標計算 ──

        # RSI (短週期)
        delta = price.diff()
        gain = delta.where(delta > 0, 0.0).ewm(span=self.rsi_period, adjust=False).mean()
        loss = (-delta.where(delta < 0, 0.0)).ewm(span=self.rsi_period, adjust=False).mean()
        rs = gain / loss.replace(0, np.nan)
        df["rsi"] = 100 - (100 / (1 + rs))

        # 布林帶
        df["bb_mid"] = price.rolling(self.bb_period).mean()
        bb_std = price.rolling(self.bb_period).std()
        df["bb_upper"] = df["bb_mid"] + self.bb_std * bb_std
        df["bb_lower"] = df["bb_mid"] - self.bb_std * bb_std
        df["bb_width"] = (df["bb_upper"] - df["bb_lower"]) / df["bb_mid"] * 100
        df["pct_b"] = (price - df["bb_lower"]) / (df["bb_upper"] - df["bb_lower"])

        # ATR
        high_low = df["high"] - df["low"]
        high_close = (df["high"] - price.shift(1)).abs()
        low_close = (df["low"] - price.shift(1)).abs()
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        df["atr"] = tr.rolling(14).mean()
        df["atr_pct"] = df["atr"] / price * 100

        # 趨勢 EMA
        df["ema_trend"] = price.ewm(span=self.trend_ema, adjust=False).mean()
        df["trend_dist"] = (price - df["ema_trend"]) / df["ema_trend"] * 100

        # 成交量
        df["vol_ma"] = df["volume"].rolling(20).mean()

        # Stochastic RSI (更敏感的超賣超買)
        rsi = df["rsi"]
        rsi_min = rsi.rolling(14).min()
        rsi_max = rsi.rolling(14).max()
        rsi_range = rsi_max - rsi_min
        df["stoch_rsi"] = np.where(rsi_range > 0, (rsi - rsi_min) / rsi_range * 100, 50)

        # VWAP (24h)
        typical = (df["high"] + df["low"] + price) / 3
        tp_vol = typical * df["volume"]
        df["vwap"] = tp_vol.rolling(24).sum() / df["volume"].rolling(24).sum()
        df["vwap_dist"] = (price - df["vwap"]) / df["vwap"] * 100

        # 前一根 K 線方向 (用於確認反轉)
        df["prev_bearish"] = (df["close"].shift(1) < df["open"].shift(1)).astype(int)
        df["prev_bullish"] = (df["close"].shift(1) > df["open"].shift(1)).astype(int)

        # ── 信號生成 ──
        df["signal"] = 0
        df["stop_loss"] = np.nan
        df["take_profit"] = np.nan

        warmup = max(self.trend_ema, self.bb_period, 24) + 20
        consecutive_losses = 0
        cooldown_until = 0

        for i in range(warmup, len(df)):
            row = df.iloc[i]

            if pd.isna(row["atr"]) or row["atr"] == 0:
                continue

            # 冷卻期
            if i < cooldown_until:
                continue

            curr_price = row["close"]
            atr = row["atr"]
            atr_pct = row["atr_pct"]

            # ── 全局過濾器 ──

            # 波動率範圍
            if atr_pct < self.min_atr_pct or atr_pct > self.max_atr_pct:
                continue

            # 成交量過濾
            if row["volume"] < row["vol_ma"] * self.volume_mult:
                continue

            # ── 做多信號: 計算確認數 ──
            long_confirms = 0

            # 1. RSI 超賣
            if row["rsi"] <= self.rsi_extreme_low:
                long_confirms += 1

            # 2. Stochastic RSI 超賣
            if row["stoch_rsi"] <= 10:
                long_confirms += 1

            # 3. 價格在 BB 下軌以下或附近
            if row["pct_b"] <= 0.05:
                long_confirms += 1

            # 4. 價格低於 VWAP
            if row["vwap_dist"] < -0.5:
                long_confirms += 1

            # 5. 前一根是陰線 (賣壓衰竭確認)
            if row["prev_bearish"]:
                long_confirms += 1

            # 6. 不在強下跌趨勢中 (趨勢距離不超過閾值)
            trend_ok_long = row["trend_dist"] > -self.trend_strength_threshold
            if not trend_ok_long:
                long_confirms = 0  # 強下跌中不做多

            # ── 做空信號: 計算確認數 ──
            short_confirms = 0

            if row["rsi"] >= self.rsi_extreme_high:
                short_confirms += 1
            if row["stoch_rsi"] >= 90:
                short_confirms += 1
            if row["pct_b"] >= 0.95:
                short_confirms += 1
            if row["vwap_dist"] > 0.5:
                short_confirms += 1
            if row["prev_bullish"]:
                short_confirms += 1

            trend_ok_short = row["trend_dist"] < self.trend_strength_threshold
            if not trend_ok_short:
                short_confirms = 0

            # ── 產生信號 ──
            if long_confirms >= self.min_confirmations:
                df.iloc[i, df.columns.get_loc("signal")] = 1
                df.iloc[i, df.columns.get_loc("stop_loss")] = curr_price - atr * self.sl_atr_mult
                df.iloc[i, df.columns.get_loc("take_profit")] = curr_price + atr * self.tp_atr_mult

            elif short_confirms >= self.min_confirmations:
                df.iloc[i, df.columns.get_loc("signal")] = -1
                df.iloc[i, df.columns.get_loc("stop_loss")] = curr_price + atr * self.sl_atr_mult
                df.iloc[i, df.columns.get_loc("take_profit")] = curr_price - atr * self.tp_atr_mult

        return df
