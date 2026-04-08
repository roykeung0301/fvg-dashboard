"""
FVG Trend Follow — 結合 FVG 入場 + FVCL3 V2 趨勢跟蹤框架

取兩者之長:
- Craig Percoco: FVG 作為高概率入場點 (機構訂單流不平衡)
- FVCL3 V2: 多時間框架方向 + 趨勢反轉出場 + 讓利潤奔跑

入場: 日線上升趨勢 → 價格回調到 FVG zone → 入場
出場: 日線趨勢反轉 / 寬 SL / 超時
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


class FVGTrendStrategy(BaseStrategy):

    name = "FVG Trend Follow"
    description = "FVG entry + MTF trend following + trend reversal exit"

    commission_rate = 0.0004
    slippage_rate = 0.0001
    max_position_pct = 0.95

    def __init__(
        self,
        # Daily trend
        daily_ema_fast: int = 10,
        daily_ema_slow: int = 20,

        # FVG settings
        fvg_min_size_pct: float = 0.1,   # min FVG size as % of price
        fvg_max_age: int = 100,            # max bars before FVG expires
        max_active_fvgs: int = 30,

        # Entry filter
        entry_min_score: int = 2,          # out of 4

        # Risk
        sl_atr: float = 5.0,
        trail_start_atr: float = 2.0,      # activate trail after N ATR profit
        trail_atr: float = 3.0,            # trail distance behind best price
        breakeven_atr: float = 99.0,       # disabled by default

        # Timing
        atr_period: int = 14,
        cooldown: int = 12,
        max_hold: int = 720,
    ):
        self.daily_ema_fast = daily_ema_fast
        self.daily_ema_slow = daily_ema_slow
        self.fvg_min_size_pct = fvg_min_size_pct
        self.fvg_max_age = fvg_max_age
        self.max_active_fvgs = max_active_fvgs
        self.entry_min_score = entry_min_score
        self.sl_atr = sl_atr
        self.trail_start_atr = trail_start_atr
        self.trail_atr = trail_atr
        self.breakeven_atr = breakeven_atr
        self.atr_period = atr_period
        self.cooldown = cooldown
        self.max_hold = max_hold

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        price = df["close"]

        # ── 1h indicators ──
        hl = df["high"] - df["low"]
        hc = (df["high"] - price.shift(1)).abs()
        lc = (df["low"] - price.shift(1)).abs()
        tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
        df["atr"] = tr.rolling(self.atr_period).mean()

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
        df["daily_trend"] = df_d["trend"].reindex(df.index, method="ffill")

        # ── Signals ──
        df["signal"] = 0
        df["stop_loss"] = np.nan
        df["take_profit"] = np.nan
        df["trail_stop"] = np.nan

        highs = df["high"].values
        lows = df["low"].values
        closes = df["close"].values

        active_fvgs: List[FVG] = []
        last_sig = -self.cooldown - 1
        warmup = 300

        # Trailing stop state: tracked per active signal
        # When a signal fires, we track best_price and trail activation
        trail_active = False
        trail_side: Optional[str] = None
        trail_entry_price = 0.0
        trail_entry_atr = 0.0
        trail_best_price = 0.0
        trail_current_stop = np.nan

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

            # ── Update trailing stop for active trail ──
            if trail_side is not None:
                p_now = closes[i]
                if trail_side == "long":
                    trail_best_price = max(trail_best_price, highs[i])
                    profit_atr = (trail_best_price - trail_entry_price) / trail_entry_atr
                    if profit_atr >= self.trail_start_atr:
                        trail_active = True
                    if trail_active:
                        trail_current_stop = trail_best_price - self.trail_atr * trail_entry_atr
                else:  # short
                    trail_best_price = min(trail_best_price, lows[i])
                    profit_atr = (trail_entry_price - trail_best_price) / trail_entry_atr
                    if profit_atr >= self.trail_start_atr:
                        trail_active = True
                    if trail_active:
                        trail_current_stop = trail_best_price + self.trail_atr * trail_entry_atr

                if trail_active:
                    df.iloc[i, df.columns.get_loc("trail_stop")] = trail_current_stop

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

            # ── LONG: daily uptrend + price enters bullish FVG ──
            if daily == 1:
                for fvg in active_fvgs:
                    if fvg.direction != "bull":
                        continue
                    # Price must touch FVG zone
                    if not (lows[i] <= fvg.top and highs[i] >= fvg.bottom):
                        continue

                    # Score
                    score = 0
                    if row["rsi"] <= 40:
                        score += 1
                    if row["volume"] > row["vol_ma"] * 0.7:
                        score += 1
                    if closes[i] > df.iloc[i - 1]["close"]:  # bullish close
                        score += 1
                    if p <= fvg.ce * 1.01:  # near or below CE
                        score += 1

                    if score >= self.entry_min_score:
                        sl = p - atr * self.sl_atr
                        df.iloc[i, df.columns.get_loc("signal")] = 1
                        df.iloc[i, df.columns.get_loc("stop_loss")] = sl
                        last_sig = i
                        # Reset trailing stop state for new signal
                        trail_active = False
                        trail_side = "long"
                        trail_entry_price = p
                        trail_entry_atr = atr
                        trail_best_price = p
                        trail_current_stop = np.nan
                        break

            # ── SHORT: daily downtrend + price enters bearish FVG ──
            elif daily == -1:
                for fvg in active_fvgs:
                    if fvg.direction != "bear":
                        continue
                    if not (lows[i] <= fvg.top and highs[i] >= fvg.bottom):
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
                        sl = p + atr * self.sl_atr
                        df.iloc[i, df.columns.get_loc("signal")] = -1
                        df.iloc[i, df.columns.get_loc("stop_loss")] = sl
                        last_sig = i
                        # Reset trailing stop state for new signal
                        trail_active = False
                        trail_side = "short"
                        trail_entry_price = p
                        trail_entry_atr = atr
                        trail_best_price = p
                        trail_current_stop = np.nan
                        break

        return df

    def backtest(
        self,
        df: pd.DataFrame,
        initial_capital: float = 2000.0,
        max_holding_bars: int = 720,
    ) -> Dict:
        """Trend following backtest — exit on trend reversal / SL / timeout."""
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
                pos_atr = position["atr"]

                # ── Update trailing stop ──
                if side == "long":
                    position["best_price"] = max(position.get("best_price", entry), high)
                    profit_atr = (position["best_price"] - entry) / pos_atr if pos_atr > 0 else 0
                    if profit_atr >= self.trail_start_atr:
                        new_trail = position["best_price"] - self.trail_atr * pos_atr
                        position["trail_sl"] = max(position.get("trail_sl", 0), new_trail)
                else:
                    position["best_price"] = min(position.get("best_price", entry), low)
                    profit_atr = (entry - position["best_price"]) / pos_atr if pos_atr > 0 else 0
                    if profit_atr >= self.trail_start_atr:
                        new_trail = position["best_price"] + self.trail_atr * pos_atr
                        cur_trail = position.get("trail_sl", float("inf"))
                        position["trail_sl"] = min(cur_trail, new_trail)

                # SL (use tighter of fixed SL and trail SL)
                effective_sl = position["sl"]
                trail_sl = position.get("trail_sl")
                if trail_sl is not None:
                    if side == "long":
                        effective_sl = max(effective_sl, trail_sl)  # tighter = higher for long
                    else:
                        effective_sl = min(effective_sl, trail_sl)  # tighter = lower for short

                if side == "long" and low <= effective_sl:
                    exit_price = max(effective_sl, low)
                    exit_reason = "trail_sl" if trail_sl is not None and effective_sl == trail_sl else "sl"
                elif side == "short" and high >= effective_sl:
                    exit_price = min(effective_sl, high)
                    exit_reason = "trail_sl" if trail_sl is not None and effective_sl == trail_sl else "sl"

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
                        "best_price": price,
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
