"""
FVG Trend V2 Optimization — Test each filter individually and in combination
across all 4 assets (BTCUSDT, ETHUSDT, SOLUSDT, BNBUSDT).

Tests:
  0. Baseline (Combo3 V1)
  1. Volatility Filter only
  2. Adaptive SL only
  3. Trend Strength Filter only
  4. Larger FVG size only
  5. Volume Spike (Order Block) only
  6. Liquidity Sweep only
  7-N. Best combinations
"""
from __future__ import annotations

import os
import sys
import time
import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from strategies.fvg_trend import FVGTrendStrategy
from strategies.fvg_trend_v2 import FVGTrendV2Strategy
from data.historical_fetcher import load_data

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)

INITIAL_CAPITAL = 2000.0
SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT"]

# ── Baseline: Combo3 (V1) ──
COMBO3_BASE = dict(
    daily_ema_fast=10,
    daily_ema_slow=20,
    fvg_min_size_pct=0.05,
    entry_min_score=1,
    sl_atr=6.0,
    fvg_max_age=100,
    max_active_fvgs=30,
    trail_start_atr=99.0,
    breakeven_atr=99.0,
    cooldown=12,
    max_hold=720,
)

# ── Individual Filter Configs ──
CONFIGS: Dict[str, dict] = {}

# 0. Baseline
CONFIGS["Baseline (Combo3)"] = {**COMBO3_BASE}

# 1. Volatility Filter variants
for atr_pct in [0.3, 0.5, 0.8]:
    CONFIGS[f"VolFilter ATR>{atr_pct}%"] = {**COMBO3_BASE, "min_atr_pct": atr_pct}

# 2. Adaptive SL
CONFIGS["AdaptiveSL 4-8"] = {
    **COMBO3_BASE,
    "adaptive_sl": True,
    "adaptive_sl_min": 4.0,
    "adaptive_sl_max": 8.0,
}
CONFIGS["AdaptiveSL 3-7"] = {
    **COMBO3_BASE,
    "adaptive_sl": True,
    "adaptive_sl_min": 3.0,
    "adaptive_sl_max": 7.0,
}

# 3. Trend Strength Filter
for gap in [0.5, 1.0, 1.5]:
    CONFIGS[f"TrendStr EMA>{gap}%"] = {**COMBO3_BASE, "min_ema_gap_pct": gap}

# 4. FVG Quality (larger min size)
for fvg_size in [0.1, 0.15, 0.2]:
    CONFIGS[f"FVG>{fvg_size}%"] = {**COMBO3_BASE, "fvg_min_size_pct": fvg_size}

# 5. Volume Spike (Order Block)
for vol_mult in [1.2, 1.5, 2.0]:
    CONFIGS[f"VolSpike>{vol_mult}x"] = {**COMBO3_BASE, "ob_vol_mult": vol_mult}

# 6. Liquidity Sweep
for lb in [15, 20, 30]:
    CONFIGS[f"LiqSweep lb={lb}"] = {**COMBO3_BASE, "liq_sweep_lookback": lb}

# 7-N. Combinations (will be built after individual results)
COMBO_CONFIGS: Dict[str, dict] = {}

# Combo A: VolFilter + TrendStr
COMBO_CONFIGS["Combo_A: Vol0.5+Trend1.0"] = {
    **COMBO3_BASE,
    "min_atr_pct": 0.5,
    "min_ema_gap_pct": 1.0,
}

# Combo B: VolFilter + AdaptiveSL
COMBO_CONFIGS["Combo_B: Vol0.5+AdaSL4-8"] = {
    **COMBO3_BASE,
    "min_atr_pct": 0.5,
    "adaptive_sl": True,
    "adaptive_sl_min": 4.0,
    "adaptive_sl_max": 8.0,
}

# Combo C: VolFilter + FVG quality
COMBO_CONFIGS["Combo_C: Vol0.5+FVG0.1"] = {
    **COMBO3_BASE,
    "min_atr_pct": 0.5,
    "fvg_min_size_pct": 0.1,
}

# Combo D: TrendStr + AdaptiveSL
COMBO_CONFIGS["Combo_D: Trend1.0+AdaSL"] = {
    **COMBO3_BASE,
    "min_ema_gap_pct": 1.0,
    "adaptive_sl": True,
    "adaptive_sl_min": 4.0,
    "adaptive_sl_max": 8.0,
}

# Combo E: VolFilter + TrendStr + AdaptiveSL (triple)
COMBO_CONFIGS["Combo_E: Vol0.5+Trend1.0+AdaSL"] = {
    **COMBO3_BASE,
    "min_atr_pct": 0.5,
    "min_ema_gap_pct": 1.0,
    "adaptive_sl": True,
    "adaptive_sl_min": 4.0,
    "adaptive_sl_max": 8.0,
}

# Combo F: Vol + LiqSweep
COMBO_CONFIGS["Combo_F: Vol0.5+LiqSweep20"] = {
    **COMBO3_BASE,
    "min_atr_pct": 0.5,
    "liq_sweep_lookback": 20,
}

# Combo G: VolFilter + TrendStr + VolSpike (aggressive filter)
COMBO_CONFIGS["Combo_G: Vol0.5+Trend0.5+OB1.2"] = {
    **COMBO3_BASE,
    "min_atr_pct": 0.5,
    "min_ema_gap_pct": 0.5,
    "ob_vol_mult": 1.2,
}

# Combo H: Moderate combo - slight filter on everything
COMBO_CONFIGS["Combo_H: Vol0.3+Trend0.5+FVG0.1"] = {
    **COMBO3_BASE,
    "min_atr_pct": 0.3,
    "min_ema_gap_pct": 0.5,
    "fvg_min_size_pct": 0.1,
}


def sep(title: str):
    print(f"\n{'='*110}\n  {title}\n{'='*110}")


def run_single(cfg: dict, df: pd.DataFrame, is_baseline: bool = False) -> dict:
    """Run a single backtest. Use V1 for baseline, V2 for everything else."""
    if is_baseline:
        strat = FVGTrendStrategy(**cfg)
    else:
        strat = FVGTrendV2Strategy(**cfg)
    result = strat.backtest(df, INITIAL_CAPITAL, cfg.get("max_hold", 720))
    return result["metrics"]


def print_header():
    print(
        f"  {'Config':42s} | {'Trades':>6s} | {'WR':>6s} | {'RR':>5s} | "
        f"{'PF':>6s} | {'Return':>10s} | {'MDD':>6s} | {'Sharpe':>7s} | "
        f"{'AvgWin':>7s} | {'AvgLoss':>7s} | {'SL%':>5s}"
    )
    print(f"  {'-'*42}-+-{'-'*6}-+-{'-'*6}-+-{'-'*5}-+-{'-'*6}-+-{'-'*10}-+-{'-'*6}-+-{'-'*7}-+-{'-'*7}-+-{'-'*7}-+-{'-'*5}")


def print_row(name: str, m: dict, baseline_trades: int = 0):
    trades = m.get("total_trades", 0)
    if trades == 0:
        print(f"  {name:42s} | NO TRADES")
        return

    # Calculate SL exit percentage
    exit_reasons = m.get("exit_reasons", {})
    sl_count = exit_reasons.get("sl", 0)
    sl_pct = sl_count / trades * 100 if trades > 0 else 0

    trade_marker = ""
    if baseline_trades > 0:
        ratio = trades / baseline_trades
        if ratio < 0.5:
            trade_marker = " [!LOW]"
        elif ratio < 0.7:
            trade_marker = " [-30%]"

    print(
        f"  {name:42s} | {trades:>6d}{trade_marker:7s} | {m['win_rate']:>5.1f}% | "
        f"{m['risk_reward_ratio']:>4.1f} | "
        f"{m['profit_factor']:>5.2f} | {m['total_return_pct']:>+9.1f}% | "
        f"{m['max_drawdown_pct']:>5.1f}% | {m['sharpe_ratio']:>6.2f} | "
        f"{m['avg_win']:>7.2f} | {m['avg_loss']:>7.2f} | {sl_pct:>4.1f}%"
    )


def main():
    # ── Load Data ──
    sep("Loading Data")
    datasets: Dict[str, pd.DataFrame] = {}
    for sym in SYMBOLS:
        df = load_data(sym, "1h")
        if df is not None and len(df) > 1000:
            datasets[sym] = df
            print(f"  {sym}: {len(df)} bars ({df.index[0].date()} to {df.index[-1].date()})")
        else:
            print(f"  {sym}: NO DATA (run run_cross_asset.py first to fetch)")

    if not datasets:
        print("\n  ERROR: No data available. Exiting.")
        return

    # ── Phase 1: Individual Filters ──
    sep("Phase 1: Individual Filter Tests")

    all_results: Dict[str, Dict[str, dict]] = {}  # {config_name: {symbol: metrics}}
    baseline_trades: Dict[str, int] = {}  # {symbol: baseline_trade_count}

    for cfg_name, cfg in CONFIGS.items():
        all_results[cfg_name] = {}
        is_baseline = ("Baseline" in cfg_name)

        for sym, df in datasets.items():
            try:
                t0 = time.time()
                m = run_single(cfg, df, is_baseline=is_baseline)
                elapsed = time.time() - t0

                if "error" in m:
                    all_results[cfg_name][sym] = {"total_trades": 0, "error": m["error"]}
                else:
                    all_results[cfg_name][sym] = m
                    if is_baseline:
                        baseline_trades[sym] = m["total_trades"]
            except Exception as e:
                print(f"  ERROR: {cfg_name} on {sym}: {e}")
                all_results[cfg_name][sym] = {"total_trades": 0, "error": str(e)}

    # Print results per symbol
    for sym in SYMBOLS:
        if sym not in datasets:
            continue
        sep(f"Results: {sym}")
        print_header()
        for cfg_name in CONFIGS:
            m = all_results[cfg_name].get(sym, {})
            if m.get("total_trades", 0) > 0:
                print_row(cfg_name, m, baseline_trades.get(sym, 0))
            else:
                print(f"  {cfg_name:42s} | NO TRADES / ERROR")

    # ── Phase 2: Combination Tests ──
    sep("Phase 2: Combination Tests")

    combo_results: Dict[str, Dict[str, dict]] = {}

    for cfg_name, cfg in COMBO_CONFIGS.items():
        combo_results[cfg_name] = {}
        for sym, df in datasets.items():
            try:
                m = run_single(cfg, df, is_baseline=False)
                if "error" in m:
                    combo_results[cfg_name][sym] = {"total_trades": 0}
                else:
                    combo_results[cfg_name][sym] = m
            except Exception as e:
                combo_results[cfg_name][sym] = {"total_trades": 0, "error": str(e)}

    for sym in SYMBOLS:
        if sym not in datasets:
            continue
        sep(f"Combo Results: {sym}")
        print_header()
        # Print baseline for reference
        bm = all_results.get("Baseline (Combo3)", {}).get(sym, {})
        if bm.get("total_trades", 0) > 0:
            print_row("Baseline (Combo3)", bm)
        for cfg_name in COMBO_CONFIGS:
            m = combo_results[cfg_name].get(sym, {})
            if m.get("total_trades", 0) > 0:
                print_row(cfg_name, m, baseline_trades.get(sym, 0))
            else:
                print(f"  {cfg_name:42s} | NO TRADES / ERROR")

    # ── Phase 3: Cross-Asset Summary ──
    sep("CROSS-ASSET SUMMARY")

    # Merge all results
    all_cfg_results = {**all_results, **combo_results}

    # Score each config: profitable on N assets, avg WR improvement, avg return
    print(f"\n  {'Config':42s} | {'Prof':>4s} | {'AvgWR':>6s} | {'AvgRet':>10s} | {'AvgPF':>6s} | {'AvgTrades':>9s} | {'Score':>6s}")
    print(f"  {'-'*42}-+-{'-'*4}-+-{'-'*6}-+-{'-'*10}-+-{'-'*6}-+-{'-'*9}-+-{'-'*6}")

    scored: List[Tuple[str, float, dict]] = []

    for cfg_name, sym_results in all_cfg_results.items():
        n_profitable = 0
        wrs, rets, pfs, trade_counts = [], [], [], []

        for sym in SYMBOLS:
            m = sym_results.get(sym, {})
            if m.get("total_trades", 0) == 0:
                continue
            if m.get("total_return_pct", 0) > 0:
                n_profitable += 1
            wrs.append(m.get("win_rate", 0))
            rets.append(m.get("total_return_pct", 0))
            pfs.append(m.get("profit_factor", 0))
            trade_counts.append(m.get("total_trades", 0))

        if not wrs:
            continue

        avg_wr = np.mean(wrs)
        avg_ret = np.mean(rets)
        avg_pf = np.mean(pfs)
        avg_trades = np.mean(trade_counts)

        # Check trade count constraint: must keep at least 50% of baseline trades
        trade_ok = True
        for sym in SYMBOLS:
            m = sym_results.get(sym, {})
            bt = baseline_trades.get(sym, 1)
            if m.get("total_trades", 0) < bt * 0.5:
                trade_ok = False

        # Composite score: prioritize WR improvement + profitability
        # Penalize configs that drop too many trades
        score = (avg_wr * 0.3 + avg_pf * 10 + n_profitable * 15)
        if not trade_ok:
            score -= 30  # penalty for too few trades

        mark = ""
        if not trade_ok:
            mark = " [LOW TRADES]"

        print(
            f"  {cfg_name:42s} | {n_profitable:>4d}/4 | {avg_wr:>5.1f}% | "
            f"{avg_ret:>+9.1f}% | {avg_pf:>5.2f} | {avg_trades:>9.0f} | "
            f"{score:>5.1f}{mark}"
        )
        scored.append((cfg_name, score, {
            "n_profitable": n_profitable,
            "avg_wr": avg_wr,
            "avg_ret": avg_ret,
            "avg_pf": avg_pf,
            "avg_trades": avg_trades,
            "trade_ok": trade_ok,
        }))

    # Sort by score
    scored.sort(key=lambda x: -x[1])

    sep("TOP 5 CONFIGURATIONS (by composite score)")
    for rank, (cfg_name, score, info) in enumerate(scored[:5], 1):
        ok = "OK" if info["trade_ok"] else "LOW"
        print(
            f"  #{rank}: {cfg_name}\n"
            f"       Prof: {info['n_profitable']}/4 | AvgWR: {info['avg_wr']:.1f}% | "
            f"AvgRet: {info['avg_ret']:+.1f}% | AvgPF: {info['avg_pf']:.2f} | "
            f"AvgTrades: {info['avg_trades']:.0f} | TradeCount: {ok} | Score: {score:.1f}\n"
        )

    # Print the winner's detailed per-asset breakdown
    if scored:
        winner = scored[0][0]
        sep(f"WINNER DETAIL: {winner}")
        print_header()
        # Baseline first
        for sym in SYMBOLS:
            bm = all_results.get("Baseline (Combo3)", {}).get(sym, {})
            if bm.get("total_trades", 0) > 0:
                print_row(f"[BASE] {sym}", bm)
        print()
        for sym in SYMBOLS:
            wm = all_cfg_results[winner].get(sym, {})
            if wm.get("total_trades", 0) > 0:
                print_row(f"[WIN]  {sym}", wm, baseline_trades.get(sym, 0))

    print(f"\n  Done. Tested {len(all_cfg_results)} configs across {len(datasets)} assets.")


if __name__ == "__main__":
    main()
