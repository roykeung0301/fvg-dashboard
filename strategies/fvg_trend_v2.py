"""
FVG Trend Follow V2 — Enhanced with Smart Money filters to reduce losing trades

New filters (all optional, disabled by default to match V1 baseline):
1. Volatility Filter (min_atr_pct) — skip trades when ATR is too low (choppy market)
2. Adaptive SL (adaptive_sl) — SL width scales with volatility regime
3. Trend Strength Filter (min_ema_gap_pct) — require minimum EMA gap
4. FVG Quality Filter (fvg_min_size_pct) — already exists, just tune higher
5. Order Block / Volume Spike (ob_vol_mult) — require volume spike near FVG
6. Liquidity Sweep (liq_sweep) — require sweep of recent swing low/high before entry
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, List, Dict

import numpy as np
import pandas as pd

from strategies.base_strategy import BaseStrategy, TradeRecord


@dataclass
class FVG:
    bar_idx: int
    direction: str   # "bull" or "bear"
    top: float
    bottom: float
    ce: float        # midpoint


class FVGTrendV2Strategy(BaseStrategy):

    name = "FVG Trend Follow V2"
    description = "FVG entry + MTF trend following + Smart Money filters"

    commission_rate = 0.0004
    slippage_rate = 0.0001
    max_position_pct = 0.95

    def __init__(
        self,
        # Daily trend
        daily_ema_fast: int = 10,
        daily_ema_slow: int = 20,

        # FVG settings
        fvg_min_size_pct: float = 0.05,
        fvg_max_age: int = 100,
        max_active_fvgs: int = 30,

        # Entry filter
        entry_min_score: int = 1,

        # Risk
        sl_atr: float = 6.0,
        trail_start_atr: float = 99.0,
        breakeven_atr: float = 99.0,

        # Timing
        atr_period: int = 14,
        cooldown: int = 12,
        max_hold: int = 720,

        # ── NEW V2 Filters ──

        # 1. Volatility Filter: skip trades when ATR/price < this threshold
        #    0 = disabled. Suggested: 0.3 to 0.5 (ATR as % of price)
        min_atr_pct: float = 0.0,

        # 2. Adaptive SL: scale SL with volatility regime
        #    When enabled, sl_atr becomes the base, and actual SL = base * vol_ratio
        #    clamped to [adaptive_sl_min, adaptive_sl_max]
        #    Set adaptive_sl=False to use fixed SL
        adaptive_sl: bool = False,
        adaptive_sl_min: float = 4.0,
        adaptive_sl_max: float = 8.0,
        adaptive_sl_lookback: int = 168,  # 7 days of 1h bars

        # 3. Trend Strength Filter: min EMA gap as % of price
        #    0 = disabled. Suggested: 0.5 to 2.0
        min_ema_gap_pct: float = 0.0,

        # 4. Order Block / Volume Spike confirmation
        #    0 = disabled. Suggested: 1.5 to 2.5 (volume must be > X * avg)
        ob_vol_mult: float = 0.0,

        # 5. Liquidity Sweep: require price to sweep recent swing low/high
        #    before entering (Smart Money concept)
        #    0 = disabled. Suggested: 10 to 30 (lookback bars for swing)
        liq_sweep_lookback: int = 0,
    ):
        self.daily_ema_fast = daily_ema_fast
        self.daily_ema_slow = daily_ema_slow
        self.fvg_min_size_pct = fvg_min_size_pct
        self.fvg_max_age = fvg_max_age
        self.max_active_fvgs = max_active_fvgs
        self.entry_min_score = entry_min_score
        self.sl_atr = sl_atr
        self.trail_start_atr = trail_start_atr
        self.breakeven_atr = breakeven_atr
        self.atr_period = atr_period
        self.cooldown = cooldown
        self.max_hold = max_hold

        # V2 filters
        self.min_atr_pct = min_atr_pct
        self.adaptive_sl = adaptive_sl
        self.adaptive_sl_min = adaptive_sl_min
        self.adaptive_sl_max = adaptive_sl_max
        self.adaptive_sl_lookback = adaptive_sl_lookback
        self.min_ema_gap_pct = min_ema_gap_pct
        self.ob_vol_mult = ob_vol_mult
        self.liq_sweep_lookback = liq_sweep_lookback

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        price = df["close"]

        # ── 1h indicators ──
        hl = df["high"] - df["low"]
        hc = (df["high"] - price.shift(1)).abs()
        lc = (df["low"] - price.shift(1)).abs()
        tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
        df["atr"] = tr.rolling(self.atr_period).mean()

        # ATR as percentage of price (for volatility filter)
        df["atr_pct"] = df["atr"] / price * 100

        # Slow ATR for adaptive SL (volatility regime)
        if self.adaptive_sl:
            df["atr_slow"] = tr.rolling(self.adaptive_sl_lookback).mean()

        # RSI
        delta = price.diff()
        g = delta.where(delta > 0, 0.0).ewm(span=7, adjust=False).mean()
        l = (-delta.where(delta < 0, 0.0)).ewm(span=7, adjust=False).mean()
        df["rsi"] = 100 - (100 / (1 + g / l.replace(0, np.nan)))

        # Volume
        df["vol_ma"] = df["volume"].rolling(20).mean()

        # ── Daily trend ──
        df_d = df.resample("1D").agg({
            "open": "first", "high": "max", "low": "min",
            "close": "last", "volume": "sum",
        }).dropna()
        df_d["ema_f"] = df_d["close"].ewm(span=self.daily_ema_fast, adjust=False).mean()
        df_d["ema_s"] = df_d["close"].ewm(span=self.daily_ema_slow, adjust=False).mean()
        df_d["trend"] = 0
        df_d.loc[df_d["ema_f"] > df_d["ema_s"], "trend"] = 1
        df_d.loc[df_d["ema_f"] < df_d["ema_s"], "trend"] = -1

        # EMA gap as % of price (for trend strength filter)
        df_d["ema_gap_pct"] = (df_d["ema_f"] - df_d["ema_s"]).abs() / df_d["close"] * 100

        df["daily_trend"] = df_d["trend"].reindex(df.index, method="ffill")
        df["daily_ema_gap_pct"] = df_d["ema_gap_pct"].reindex(df.index, method="ffill")

        # ── Signals ──
        df["signal"] = 0
        df["stop_loss"] = np.nan
        df["take_profit"] = np.nan

        highs = df["high"].values
        lows = df["low"].values
        closes = df["close"].values
        volumes = df["volume"].values

        active_fvgs: List[FVG] = []
        last_sig = -self.cooldown - 1
        warmup = 300

        for i in range(2, len(df)):
            # Detect new FVG
            if highs[i - 2] < lows[i]:
                fvg = FVG(i, "bull", lows[i], highs[i - 2], (lows[i] + highs[i - 2]) / 2)
                size_pct = (fvg.top - fvg.bottom) / closes[i] * 100
                if size_pct >= self.fvg_min_size_pct:
                    active_fvgs.append(fvg)
            elif lows[i - 2] > highs[i]:
                fvg = FVG(i, "bear", lows[i - 2], highs[i], (lows[i - 2] + highs[i]) / 2)
                size_pct = (fvg.top - fvg.bottom) / closes[i] * 100
                if size_pct >= self.fvg_min_size_pct:
                    active_fvgs.append(fvg)

            # Purge filled / expired FVGs
            remaining = []
            for fvg in active_fvgs:
                age = i - fvg.bar_idx
                if age > self.fvg_max_age:
                    continue
                if fvg.direction == "bull" and lows[i] <= fvg.bottom:
                    continue
                if fvg.direction == "bear" and highs[i] >= fvg.top:
                    continue
                remaining.append(fvg)
            active_fvgs = remaining[-self.max_active_fvgs:]

            if i < warmup:
                continue
            if i - last_sig < self.cooldown:
                continue

            row = df.iloc[i]
            if pd.isna(row["atr"]) or row["atr"] == 0:
                continue

            p = closes[i]
            atr = row["atr"]
            daily = row.get("daily_trend", 0)
            if pd.isna(daily):
                continue

            # ── V2 FILTER 1: Volatility Filter ──
            if self.min_atr_pct > 0:
                atr_pct = row.get("atr_pct", 0)
                if pd.isna(atr_pct) or atr_pct < self.min_atr_pct:
                    continue

            # ── V2 FILTER 3: Trend Strength Filter ──
            if self.min_ema_gap_pct > 0:
                ema_gap = row.get("daily_ema_gap_pct", 0)
                if pd.isna(ema_gap) or ema_gap < self.min_ema_gap_pct:
                    continue

            # ── V2 FILTER 2: Adaptive SL calculation ──
            if self.adaptive_sl:
                atr_slow = row.get("atr_slow", atr)
                if pd.isna(atr_slow) or atr_slow == 0:
                    atr_slow = atr
                vol_ratio = atr / atr_slow  # >1 = high vol, <1 = low vol
                adaptive_mult = self.sl_atr * vol_ratio
                sl_mult = max(self.adaptive_sl_min, min(self.adaptive_sl_max, adaptive_mult))
            else:
                sl_mult = self.sl_atr

            # ── V2 FILTER 5: Liquidity Sweep check ──
            # For longs: price must have swept below recent swing low then recovered
            # For shorts: price must have swept above recent swing high then recovered
            def check_liq_sweep(direction, idx, lookback):
                if lookback <= 0:
                    return True  # disabled
                start = max(0, idx - lookback)
                if direction == "long":
                    # Find recent swing low in the lookback window
                    recent_low = np.min(lows[start:idx])
                    # Current bar or previous few bars must have swept below it
                    sweep_window = max(0, idx - 5)
                    swept = np.min(lows[sweep_window:idx + 1]) <= recent_low
                    # But current close is above it (recovery)
                    recovered = closes[idx] > recent_low
                    return swept and recovered
                else:
                    recent_high = np.max(highs[start:idx])
                    sweep_window = max(0, idx - 5)
                    swept = np.max(highs[sweep_window:idx + 1]) >= recent_high
                    recovered = closes[idx] < recent_high
                    return swept and recovered

            # ── LONG: daily uptrend + price enters bullish FVG ──
            if daily == 1:
                for fvg in active_fvgs:
                    if fvg.direction != "bull":
                        continue
                    if not (lows[i] <= fvg.top and highs[i] >= fvg.bottom):
                        continue

                    # V2 FILTER 4: Volume spike (Order Block confirmation)
                    if self.ob_vol_mult > 0:
                        vol_ma = row.get("vol_ma", 0)
                        if pd.isna(vol_ma) or vol_ma == 0:
                            continue
                        if volumes[i] < vol_ma * self.ob_vol_mult:
                            continue

                    # V2 FILTER 5: Liquidity Sweep
                    if not check_liq_sweep("long", i, self.liq_sweep_lookback):
                        continue

                    # Score
                    score = 0
                    if row["rsi"] <= 40:
                        score += 1
                    if row["volume"] > row["vol_ma"] * 0.7:
                        score += 1
                    if closes[i] > df.iloc[i - 1]["close"]:
                        score += 1
                    if p <= fvg.ce * 1.01:
                        score += 1

                    if score >= self.entry_min_score:
                        sl = p - atr * sl_mult
                        df.iloc[i, df.columns.get_loc("signal")] = 1
                        df.iloc[i, df.columns.get_loc("stop_loss")] = sl
                        last_sig = i
                        break

            # ── SHORT: daily downtrend + price enters bearish FVG ──
            elif daily == -1:
                for fvg in active_fvgs:
                    if fvg.direction != "bear":
                        continue
                    if not (lows[i] <= fvg.top and highs[i] >= fvg.bottom):
                        continue

                    # V2 FILTER 4: Volume spike
                    if self.ob_vol_mult > 0:
                        vol_ma = row.get("vol_ma", 0)
                        if pd.isna(vol_ma) or vol_ma == 0:
                            continue
                        if volumes[i] < vol_ma * self.ob_vol_mult:
                            continue

                    # V2 FILTER 5: Liquidity Sweep
                    if not check_liq_sweep("short", i, self.liq_sweep_lookback):
                        continue

                    score = 0
                    if row["rsi"] >= 60:
                        score += 1
                    if row["volume"] > row["vol_ma"] * 0.7:
                        score += 1
                    if closes[i] < df.iloc[i - 1]["close"]:
                        score += 1
                    if p >= fvg.ce * 0.99:
                        score += 1

                    if score >= self.entry_min_score:
                        sl = p + atr * sl_mult
                        df.iloc[i, df.columns.get_loc("signal")] = -1
                        df.iloc[i, df.columns.get_loc("stop_loss")] = sl
                        last_sig = i
                        break

        return df

    def backtest(
        self,
        df: pd.DataFrame,
        initial_capital: float = 2000.0,
        max_holding_bars: int = 720,
    ) -> Dict:
        """Trend following backtest -- exit on trend reversal / SL / timeout."""
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
                side = position["side"]
                entry = position["entry_price"]

                # SL
                if side == "long" and low <= position["sl"]:
                    exit_price = max(position["sl"], low)
                    exit_reason = "sl"
                elif side == "short" and high >= position["sl"]:
                    exit_price = min(position["sl"], high)
                    exit_reason = "sl"

                # Trend reversal
                if exit_price is None:
                    daily = row.get("daily_trend", 0)
                    if not pd.isna(daily):
                        if side == "long" and daily != 1:
                            exit_price = price
                            exit_reason = "trend_rev"
                        elif side == "short" and daily != -1:
                            exit_price = price
                            exit_reason = "trend_rev"

                # Timeout
                if exit_price is None and bars_held >= self.max_hold:
                    exit_price = price
                    exit_reason = "timeout"

                if exit_price is not None:
                    trade = self._close_position(position, exit_price, exit_reason, df.index[i])
                    trades.append(trade)
                    equity += trade.pnl - trade.commission
                    position = None

            # Entry
            if position is None and row.get("signal", 0) != 0:
                sig = row["signal"]
                side = "long" if sig == 1 else "short"
                sl = row.get("stop_loss", 0)
                atr_val = row.get("atr", price * 0.02)
                if pd.isna(sl) or sl == 0:
                    sl = price * (0.92 if side == "long" else 1.08)
                quantity = (equity * self.max_position_pct) / price
                if equity > 1 and quantity > 0:
                    position = {
                        "side": side,
                        "entry_price": price,
                        "quantity": quantity,
                        "entry_idx": i,
                        "entry_time": df.index[i],
                        "sl": sl,
                        "tp": price * (10 if side == "long" else 0.1),
                        "atr": atr_val if not pd.isna(atr_val) else price * 0.02,
                    }

            mtm = equity
            if position is not None:
                if position["side"] == "long":
                    mtm = equity + (price - position["entry_price"]) * position["quantity"]
                else:
                    mtm = equity + (position["entry_price"] - price) * position["quantity"]
            equity_curve.append(mtm)

        if position is not None:
            trade = self._close_position(position, df.iloc[-1]["close"], "end", df.index[-1])
            trades.append(trade)
            equity += trade.pnl - trade.commission

        equity_series = pd.Series(equity_curve, index=df.index)
        metrics = self._compute_metrics(trades, equity_series, initial_capital)
        return {"trades": trades, "equity_curve": equity_series, "metrics": metrics}
