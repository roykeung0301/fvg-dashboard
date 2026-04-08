"""
WR Optimization — Test parameter variations to improve Win Rate
while maintaining or improving Total Return.

Focus: entry_min_score, cooldown, fvg_min_size_pct, sl_atr
"""
from __future__ import annotations
import asyncio, logging, os, itertools
import numpy as np, pandas as pd
from data.historical_fetcher import ensure_data
from strategies.fvg_trend import FVGTrendStrategy

logging.basicConfig(level=logging.WARNING)

INITIAL_CAPITAL = 2000.0
START_DATE = "2024-04-01"
END_DATE   = "2026-04-01"
SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT"]

# Baseline Combo3
BASELINE = dict(
    daily_ema_fast=10, daily_ema_slow=20,
    fvg_min_size_pct=0.05, entry_min_score=1, sl_atr=6.0,
    fvg_max_age=100, max_active_fvgs=30,
    trail_start_atr=99.0, breakeven_atr=99.0,
    cooldown=12, max_hold=720,
)

# Parameter variations to test
PARAM_GRID = {
    "entry_min_score": [1, 2, 3],           # Higher = stricter entry filter
    "fvg_min_size_pct": [0.05, 0.10, 0.20], # Larger FVG = stronger signal
    "sl_atr": [4.0, 5.0, 6.0, 8.0],        # Tighter SL = less loss per trade but more stops
    "cooldown": [12, 18, 24],               # Longer cooldown = fewer trades, more selective
    "fvg_max_age": [50, 75, 100],           # Younger FVG = more relevant
}

COMMISSION_RATE = 0.0004
SLIPPAGE_RATE = 0.0001
RISK_PER_TRADE = 0.02
MAX_TOTAL_EXPOSURE = 0.70
ASSET_LIMITS = {"BTCUSDT": 0.35, "ETHUSDT": 0.30, "SOLUSDT": 0.20, "XRPUSDT": 0.20}


def portfolio_backtest(signal_dfs, symbols):
    """Simplified portfolio backtest with risk-based sizing."""
    all_indices = set()
    for df in signal_dfs.values():
        all_indices.update(df.index)
    timeline = sorted(all_indices)

    equity = INITIAL_CAPITAL
    positions = {}
    trades = []

    for ts in timeline:
        for sym in symbols:
            df = signal_dfs[sym]
            if ts not in df.index:
                continue
            row = df.loc[ts]
            price = row["close"]
            high = row["high"]
            low = row["low"]

            # Check exits
            if sym in positions:
                pos = positions[sym]
                exit_price = None
                side = pos["side"]
                hours_held = (ts - pos["entry_time"]).total_seconds() / 3600

                if side == "long" and low <= pos["sl"]:
                    exit_price = max(pos["sl"], low)
                elif side == "short" and high >= pos["sl"]:
                    exit_price = min(pos["sl"], high)

                if exit_price is None:
                    daily = row.get("daily_trend", 0)
                    if not pd.isna(daily):
                        if side == "long" and daily != 1:
                            exit_price = price
                        elif side == "short" and daily != -1:
                            exit_price = price

                if exit_price is None and hours_held >= 720:
                    exit_price = price

                if exit_price is not None:
                    qty = pos["quantity"]
                    entry = pos["entry_price"]
                    if side == "long":
                        pnl = (exit_price - entry) * qty
                    else:
                        pnl = (entry - exit_price) * qty
                    comm = (entry * qty + exit_price * qty) * (COMMISSION_RATE + SLIPPAGE_RATE)
                    trades.append(pnl - comm)
                    equity += pnl - comm
                    del positions[sym]

            # Check entries
            if sym not in positions and row.get("signal", 0) != 0:
                sig = row["signal"]
                side = "long" if sig == 1 else "short"
                sl = row.get("stop_loss", 0)
                if pd.isna(sl) or sl == 0:
                    sl = price * (0.92 if side == "long" else 1.08)

                sl_distance = abs(price - sl)
                if sl_distance < price * 0.001:
                    continue

                risk_amount = equity * RISK_PER_TRADE
                quantity = risk_amount / sl_distance
                notional = quantity * price

                asset_limit = ASSET_LIMITS.get(sym, 0.20)
                if notional > equity * asset_limit:
                    quantity = (equity * asset_limit) / price
                    notional = quantity * price

                current_exp = sum(
                    p["quantity"] * signal_dfs[s].loc[ts]["close"]
                    for s, p in positions.items()
                    if ts in signal_dfs[s].index
                )
                if (current_exp + notional) > equity * MAX_TOTAL_EXPOSURE:
                    remaining = equity * MAX_TOTAL_EXPOSURE - current_exp
                    if remaining <= 0:
                        continue
                    quantity = remaining / price

                if equity < 10 or quantity * price < 10:
                    continue

                positions[sym] = {
                    "side": side, "entry_price": price,
                    "quantity": quantity, "entry_time": ts, "sl": sl,
                }

    # Close remaining
    last_ts = timeline[-1]
    for sym in list(positions.keys()):
        pos = positions[sym]
        if last_ts in signal_dfs[sym].index:
            exit_price = signal_dfs[sym].loc[last_ts]["close"]
        else:
            exit_price = pos["entry_price"]
        qty = pos["quantity"]
        entry = pos["entry_price"]
        if pos["side"] == "long":
            pnl = (exit_price - entry) * qty
        else:
            pnl = (entry - exit_price) * qty
        comm = (entry * qty + exit_price * qty) * (COMMISSION_RATE + SLIPPAGE_RATE)
        trades.append(pnl - comm)
        equity += pnl - comm

    wins = sum(1 for p in trades if p > 0)
    wr = wins / len(trades) * 100 if trades else 0
    total_ret = (equity - INITIAL_CAPITAL) / INITIAL_CAPITAL * 100
    gross_profit = sum(p for p in trades if p > 0)
    gross_loss = abs(sum(p for p in trades if p <= 0))
    pf = gross_profit / gross_loss if gross_loss > 0 else float("inf")

    return {
        "trades": len(trades), "wr": wr, "return": total_ret,
        "final_eq": equity, "pf": pf,
    }


async def main():
    # Load data
    print("Loading data...")
    dfs_raw = {}
    for sym in SYMBOLS:
        df_full = await ensure_data(sym, "1h", "2021-04-01", END_DATE)
        dfs_raw[sym] = df_full[df_full.index >= "2024-01-01"].copy()
        print(f"  {sym}: {len(dfs_raw[sym])} bars")

    # Test individual parameter changes first (one-at-a-time)
    print(f"\n{'='*100}")
    print(f"  PHASE 1: One-at-a-time parameter sweep")
    print(f"{'='*100}")

    results = []

    # Baseline
    print("\n  Running baseline...")
    strategy = FVGTrendStrategy(**BASELINE)
    signal_dfs = {}
    for sym in SYMBOLS:
        df = dfs_raw[sym].copy()
        df = strategy.generate_signals(df)
        df.loc[df.index < START_DATE, "signal"] = 0
        signal_dfs[sym] = df
    baseline = portfolio_backtest(signal_dfs, SYMBOLS)
    baseline["name"] = "BASELINE"
    baseline["params"] = ""
    results.append(baseline)
    print(f"  Baseline: {baseline['trades']} trades | WR {baseline['wr']:.1f}% | Ret +{baseline['return']:.1f}% | PF {baseline['pf']:.2f}")

    # Sweep each parameter
    for param_name, values in PARAM_GRID.items():
        for val in values:
            if val == BASELINE.get(param_name):
                continue  # skip baseline value
            config = dict(BASELINE)
            config[param_name] = val

            strategy = FVGTrendStrategy(**config)
            signal_dfs = {}
            for sym in SYMBOLS:
                df = dfs_raw[sym].copy()
                df = strategy.generate_signals(df)
                df.loc[df.index < START_DATE, "signal"] = 0
                signal_dfs[sym] = df

            r = portfolio_backtest(signal_dfs, SYMBOLS)
            r["name"] = f"{param_name}={val}"
            r["params"] = f"{param_name}={val}"
            results.append(r)
            print(f"  {param_name}={val}: {r['trades']} trades | WR {r['wr']:.1f}% | Ret +{r['return']:.1f}% | PF {r['pf']:.2f}")

    # Phase 2: Test promising combinations
    print(f"\n{'='*100}")
    print(f"  PHASE 2: Promising combinations")
    print(f"{'='*100}")

    # Find best WR-improving params from phase 1
    combos = [
        # Higher entry score + bigger FVG
        {"entry_min_score": 2, "fvg_min_size_pct": 0.10},
        {"entry_min_score": 2, "fvg_min_size_pct": 0.20},
        {"entry_min_score": 2, "fvg_min_size_pct": 0.10, "cooldown": 18},
        {"entry_min_score": 2, "fvg_min_size_pct": 0.10, "cooldown": 24},
        {"entry_min_score": 2, "fvg_min_size_pct": 0.10, "sl_atr": 5.0},
        {"entry_min_score": 2, "fvg_min_size_pct": 0.10, "sl_atr": 4.0},
        {"entry_min_score": 2, "fvg_min_size_pct": 0.10, "fvg_max_age": 50},
        {"entry_min_score": 2, "fvg_min_size_pct": 0.10, "fvg_max_age": 75},
        # Score 3 combos
        {"entry_min_score": 3, "fvg_min_size_pct": 0.10},
        {"entry_min_score": 3, "fvg_min_size_pct": 0.05, "cooldown": 18},
        # Tighter SL combos
        {"sl_atr": 4.0, "fvg_min_size_pct": 0.10},
        {"sl_atr": 4.0, "fvg_min_size_pct": 0.10, "entry_min_score": 2},
        {"sl_atr": 5.0, "fvg_min_size_pct": 0.10, "entry_min_score": 2},
        # Younger FVG combos
        {"fvg_max_age": 50, "fvg_min_size_pct": 0.10, "entry_min_score": 2},
        {"fvg_max_age": 75, "fvg_min_size_pct": 0.10, "entry_min_score": 2},
        {"fvg_max_age": 50, "entry_min_score": 2, "cooldown": 18},
    ]

    for combo in combos:
        config = dict(BASELINE)
        config.update(combo)
        name = " + ".join(f"{k}={v}" for k, v in combo.items())

        strategy = FVGTrendStrategy(**config)
        signal_dfs = {}
        for sym in SYMBOLS:
            df = dfs_raw[sym].copy()
            df = strategy.generate_signals(df)
            df.loc[df.index < START_DATE, "signal"] = 0
            signal_dfs[sym] = df

        r = portfolio_backtest(signal_dfs, SYMBOLS)
        r["name"] = name
        r["params"] = name
        results.append(r)
        print(f"  {name}: {r['trades']} trades | WR {r['wr']:.1f}% | Ret +{r['return']:.1f}% | PF {r['pf']:.2f}")

    # Final ranking
    print(f"\n{'='*100}")
    print(f"  FINAL RANKING (sorted by Total Return, min 50 trades)")
    print(f"{'='*100}")
    print(f"  {'Rank':>4} | {'Config':<60} | {'Trades':>6} | {'WR%':>6} | {'Return%':>8} | {'PF':>6} | {'Final$':>8}")
    print(f"  {'-'*4}-+-{'-'*60}-+-{'-'*6}-+-{'-'*6}-+-{'-'*8}-+-{'-'*6}-+-{'-'*8}")

    valid = [r for r in results if r["trades"] >= 50]
    valid.sort(key=lambda x: x["return"], reverse=True)

    for i, r in enumerate(valid):
        marker = " <<<" if r["name"] == "BASELINE" else ""
        wr_delta = r["wr"] - baseline["wr"]
        wr_tag = f" (WR{wr_delta:+.1f})" if r["name"] != "BASELINE" else ""
        print(f"  {i+1:>4} | {r['name']:<60} | {r['trades']:>6} | {r['wr']:>5.1f}% | +{r['return']:>6.1f}% | {r['pf']:>5.2f} | ${r['final_eq']:>7,.0f}{marker}{wr_tag}")

    # Also show WR-sorted ranking
    print(f"\n  TOP 10 by Win Rate (min 50 trades, return > baseline*0.5):")
    min_ret = baseline["return"] * 0.5
    wr_valid = [r for r in valid if r["return"] >= min_ret]
    wr_valid.sort(key=lambda x: x["wr"], reverse=True)
    for i, r in enumerate(wr_valid[:10]):
        marker = " <<<" if r["name"] == "BASELINE" else ""
        print(f"  {i+1:>4} | {r['name']:<60} | {r['trades']:>6} | {r['wr']:>5.1f}% | +{r['return']:>6.1f}% | {r['pf']:>5.2f}{marker}")


if __name__ == "__main__":
    asyncio.run(main())
