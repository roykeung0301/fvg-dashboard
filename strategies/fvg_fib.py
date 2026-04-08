"""
FVG + Fibonacci Day Trading Strategy (Craig Percoco / ICT Style)

Smart Money Concepts strategy that combines Fair Value Gaps with Fibonacci
retracement levels for high-probability day trading entries on 5-minute candles.

Core idea:
- Detect unfilled Fair Value Gaps (institutional order flow imbalances)
- Find confluence with Fibonacci 50% / 61.8% retracement levels
- Enter at the Consequential Encroachment (midpoint) of the FVG
- Tight stop below FVG, target at Fib 1.618 extension for 3:1+ RR
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, List, Dict

import numpy as np
import pandas as pd

from strategies.base_strategy import BaseStrategy


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class FVG:
    """Represents an active (unfilled) Fair Value Gap."""
    index: int          # bar index where FVG was confirmed (candle i)
    direction: str      # "bullish" or "bearish"
    top: float          # upper boundary of the gap
    bottom: float       # lower boundary of the gap
    ce: float           # Consequential Encroachment (midpoint)


# ---------------------------------------------------------------------------
# Strategy
# ---------------------------------------------------------------------------

class FVGFibStrategy(BaseStrategy):
    """
    FVG + Fibonacci retracement confluence strategy.

    Looks for price retracing into an unfilled Fair Value Gap that overlaps
    with a key Fibonacci level (50% or 61.8%), then enters at the CE with
    a stop below the FVG and a target at the 1.618 Fib extension.
    """

    name = "fvg_fib"
    description = "Craig Percoco-style FVG + Fibonacci Smart Money day trading"

    def __init__(
        self,
        swing_lookback: int = 50,
        fib_confluence_tolerance: float = 0.005,
        rsi_period: int = 7,
        rsi_os: int = 35,
        rsi_ob: int = 65,
        ema_fast: int = 20,
        ema_slow: int = 50,
        sl_buffer_pct: float = 0.001,
        min_rr: float = 3.0,
        cooldown: int = 6,
        max_hold: int = 288,
    ) -> None:
        # Swing / Fibonacci
        self.swing_lookback = swing_lookback
        self.fib_confluence_tolerance = fib_confluence_tolerance

        # RSI filter
        self.rsi_period = rsi_period
        self.rsi_os = rsi_os
        self.rsi_ob = rsi_ob

        # Trend filter (EMA)
        self.ema_fast = ema_fast
        self.ema_slow = ema_slow

        # Risk management
        self.sl_buffer_pct = sl_buffer_pct
        self.min_rr = min_rr
        self.cooldown = cooldown
        self.max_hold = max_hold

    # ------------------------------------------------------------------
    # Indicator helpers (vectorized)
    # ------------------------------------------------------------------

    @staticmethod
    def _rsi(series: pd.Series, period: int) -> pd.Series:
        """Wilder-style RSI."""
        delta = series.diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.ewm(alpha=1.0 / period, min_periods=period).mean()
        avg_loss = loss.ewm(alpha=1.0 / period, min_periods=period).mean()
        rs = avg_gain / avg_loss.replace(0, np.nan)
        return 100.0 - 100.0 / (1.0 + rs)

    @staticmethod
    def _ema(series: pd.Series, span: int) -> pd.Series:
        return series.ewm(span=span, adjust=False).mean()

    # ------------------------------------------------------------------
    # Swing detection
    # ------------------------------------------------------------------

    @staticmethod
    def _find_swing_high(highs: np.ndarray, idx: int, lookback: int) -> Optional[int]:
        """Find most recent swing high before `idx` using a rolling window."""
        start = max(0, idx - lookback)
        segment = highs[start:idx]
        if len(segment) < 5:
            return None
        # A swing high: bar whose high is the max within +/-2 bars around it
        for j in range(len(segment) - 3, 1, -1):
            local = segment[max(0, j - 2): j + 3]
            if segment[j] == local.max():
                return start + j
        return None

    @staticmethod
    def _find_swing_low(lows: np.ndarray, idx: int, lookback: int) -> Optional[int]:
        """Find most recent swing low before `idx` using a rolling window."""
        start = max(0, idx - lookback)
        segment = lows[start:idx]
        if len(segment) < 5:
            return None
        for j in range(len(segment) - 3, 1, -1):
            local = segment[max(0, j - 2): j + 3]
            if segment[j] == local.min():
                return start + j
        return None

    # ------------------------------------------------------------------
    # Fibonacci helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _fib_retracement_levels(
        swing_low: float, swing_high: float
    ) -> Dict[str, float]:
        """Calculate Fib retracement levels from a swing range."""
        diff = swing_high - swing_low
        return {
            "0.382": swing_high - 0.382 * diff,
            "0.5": swing_high - 0.5 * diff,
            "0.618": swing_high - 0.618 * diff,
            "0.786": swing_high - 0.786 * diff,
        }

    @staticmethod
    def _fib_extension_1618(
        swing_low: float, swing_high: float, direction: str
    ) -> float:
        """1.618 Fibonacci extension target."""
        diff = swing_high - swing_low
        if direction == "long":
            return swing_high + 0.618 * diff   # 1.618 extension above
        else:
            return swing_low - 0.618 * diff     # 1.618 extension below

    # ------------------------------------------------------------------
    # FVG management
    # ------------------------------------------------------------------

    @staticmethod
    def _detect_fvg(
        high_prev2: float,
        low_prev2: float,
        high_curr: float,
        low_curr: float,
        bar_idx: int,
    ) -> Optional[FVG]:
        """
        Check for a Fair Value Gap on the current bar (candle i).
        Compares candle[i-2] vs candle[i].

        Bullish FVG: candle[i-2].high < candle[i].low  (gap up)
        Bearish FVG: candle[i-2].low  > candle[i].high (gap down)
        """
        if high_prev2 < low_curr:
            # Bullish FVG
            bottom = high_prev2
            top = low_curr
            ce = (top + bottom) / 2.0
            return FVG(index=bar_idx, direction="bullish",
                       top=top, bottom=bottom, ce=ce)

        if low_prev2 > high_curr:
            # Bearish FVG
            top = low_prev2
            bottom = high_curr
            ce = (top + bottom) / 2.0
            return FVG(index=bar_idx, direction="bearish",
                       top=top, bottom=bottom, ce=ce)

        return None

    @staticmethod
    def _purge_filled_fvgs(
        fvgs: List[FVG], high: float, low: float
    ) -> List[FVG]:
        """
        Remove FVGs that have been completely filled by the current bar.

        Bullish FVG filled when price trades down through its bottom.
        Bearish FVG filled when price trades up through its top.
        """
        remaining: List[FVG] = []
        for fvg in fvgs:
            if fvg.direction == "bullish" and low <= fvg.bottom:
                continue  # filled
            if fvg.direction == "bearish" and high >= fvg.top:
                continue  # filled
            remaining.append(fvg)
        return remaining

    # ------------------------------------------------------------------
    # FVG-Fib confluence check
    # ------------------------------------------------------------------

    def _check_confluence(
        self,
        fvg: FVG,
        fib_levels: Dict[str, float],
    ) -> bool:
        """
        Return True if the FVG zone overlaps with the Fib 50% or 61.8%
        level within the configured tolerance.
        """
        tol = self.fib_confluence_tolerance
        for key in ("0.5", "0.618"):
            level = fib_levels[key]
            # Check if the fib level falls within or near the FVG zone
            zone_top = fvg.top * (1 + tol)
            zone_bot = fvg.bottom * (1 - tol)
            if zone_bot <= level <= zone_top:
                return True
        return False

    # ------------------------------------------------------------------
    # Main signal generation
    # ------------------------------------------------------------------

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Iterate bar-by-bar over 5-minute OHLCV data:
        1. Compute indicators (RSI, EMA) vectorized.
        2. Sequentially track active FVGs, detect new ones, purge filled ones.
        3. On each bar, check if price enters an active FVG with Fib confluence.
        4. Apply trend + RSI filters and RR filter before emitting a signal.
        """
        df = df.copy()

        # -- Vectorized indicators ----------------------------------------
        df["rsi"] = self._rsi(df["close"], self.rsi_period)
        df["ema_fast"] = self._ema(df["close"], self.ema_fast)
        df["ema_slow"] = self._ema(df["close"], self.ema_slow)

        # -- Output columns -----------------------------------------------
        n = len(df)
        signals = np.zeros(n, dtype=int)
        stop_losses = np.full(n, np.nan)
        take_profits = np.full(n, np.nan)

        # -- Sequential state ---------------------------------------------
        highs = df["high"].values
        lows = df["low"].values
        closes = df["close"].values
        rsi_vals = df["rsi"].values
        ema_f = df["ema_fast"].values
        ema_s = df["ema_slow"].values

        active_fvgs: List[FVG] = []
        max_fvgs = 20
        last_signal_bar = -self.cooldown  # allow immediate first trade

        for i in range(2, n):
            # 1. Detect new FVG (needs candles i-2, i-1, i)
            new_fvg = self._detect_fvg(
                high_prev2=highs[i - 2],
                low_prev2=lows[i - 2],
                high_curr=highs[i],
                low_curr=lows[i],
                bar_idx=i,
            )
            if new_fvg is not None:
                active_fvgs.append(new_fvg)
                # Keep list bounded
                if len(active_fvgs) > max_fvgs:
                    active_fvgs = active_fvgs[-max_fvgs:]

            # 2. Purge filled FVGs
            active_fvgs = self._purge_filled_fvgs(active_fvgs, highs[i], lows[i])

            # 3. Cooldown check
            if (i - last_signal_bar) < self.cooldown:
                continue

            # 4. Find swing points for Fibonacci
            sh_idx = self._find_swing_high(highs, i, self.swing_lookback)
            sl_idx = self._find_swing_low(lows, i, self.swing_lookback)
            if sh_idx is None or sl_idx is None:
                continue

            swing_high = highs[sh_idx]
            swing_low = lows[sl_idx]
            if swing_high <= swing_low:
                continue

            fib_levels = self._fib_retracement_levels(swing_low, swing_high)

            # 5. Check each active FVG for entry conditions
            for fvg in active_fvgs:
                # Price must be inside the FVG zone on this bar
                bar_enters_fvg = (lows[i] <= fvg.top) and (highs[i] >= fvg.bottom)
                if not bar_enters_fvg:
                    continue

                # Confluence with Fib 50% or 61.8%
                if not self._check_confluence(fvg, fib_levels):
                    continue

                # --- LONG setup ---
                if fvg.direction == "bullish":
                    # Trend filter: EMA fast > EMA slow
                    if ema_f[i] <= ema_s[i]:
                        continue
                    # RSI oversold confirmation
                    if np.isnan(rsi_vals[i]) or rsi_vals[i] > self.rsi_os:
                        continue

                    entry = fvg.ce
                    sl = fvg.bottom * (1.0 - self.sl_buffer_pct)
                    tp = self._fib_extension_1618(swing_low, swing_high, "long")

                    # RR filter
                    risk = entry - sl
                    if risk <= 0:
                        continue
                    reward = tp - entry
                    if reward / risk < self.min_rr:
                        continue

                    signals[i] = 1
                    stop_losses[i] = sl
                    take_profits[i] = tp
                    last_signal_bar = i
                    break  # one signal per bar

                # --- SHORT setup ---
                elif fvg.direction == "bearish":
                    # Trend filter: EMA fast < EMA slow
                    if ema_f[i] >= ema_s[i]:
                        continue
                    # RSI overbought confirmation
                    if np.isnan(rsi_vals[i]) or rsi_vals[i] < self.rsi_ob:
                        continue

                    entry = fvg.ce
                    sl = fvg.top * (1.0 + self.sl_buffer_pct)
                    tp = self._fib_extension_1618(swing_low, swing_high, "short")

                    # RR filter
                    risk = sl - entry
                    if risk <= 0:
                        continue
                    reward = entry - tp
                    if reward / risk < self.min_rr:
                        continue

                    signals[i] = -1
                    stop_losses[i] = sl
                    take_profits[i] = tp
                    last_signal_bar = i
                    break  # one signal per bar

        df["signal"] = signals
        df["stop_loss"] = stop_losses
        df["take_profit"] = take_profits

        return df

    # ------------------------------------------------------------------
    # Convenience: run backtest with strategy-specific max_hold default
    # ------------------------------------------------------------------

    def backtest(
        self,
        df: pd.DataFrame,
        initial_capital: float = 2000.0,
        max_holding_bars: Optional[int] = None,
    ) -> Dict:
        if max_holding_bars is None:
            max_holding_bars = self.max_hold
        return super().backtest(df, initial_capital, max_holding_bars)
