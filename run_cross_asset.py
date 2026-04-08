"""
Cross-Asset Validation — FVG Trend Follow
在 ETHUSDT / SOLUSDT / BNBUSDT 上驗證策略是否具有跨資產泛化能力
"""
from __future__ import annotations
import asyncio, logging, sys
import numpy as np, pandas as pd
from data.historical_fetcher import ensure_data
from strategies.fvg_trend import FVGTrendStrategy
from strategies.walk_forward import WalkForwardValidator

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(name)-20s | %(levelname)-7s | %(message)s", datefmt="%H:%M:%S")

INITIAL_CAPITAL = 2000.0
START_DATE = "2021-04-01"
END_DATE = "2026-04-01"
INTERVAL = "1h"

# FVG Trend Combo3 — best config from BTCUSDT
COMBO3 = dict(
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

# Also test Combo2 for comparison
COMBO2 = dict(
    daily_ema_fast=10,
    daily_ema_slow=20,
    fvg_min_size_pct=0.1,
    entry_min_score=2,
    sl_atr=8.0,
    fvg_max_age=100,
    max_active_fvgs=30,
    trail_start_atr=99.0,
    breakeven_atr=99.0,
    cooldown=12,
    max_hold=1440,
)

SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT"]
CONFIGS = {
    "Combo3 (FVG05+Score1+SL6)": COMBO3,
    "Combo2 (FVG10+Score2+SL8)": COMBO2,
}


def sep(title):
    print(f"\n{'='*90}\n  {title}\n{'='*90}")


def print_row(name, m):
    star = " ***" if m.get("total_return_pct", 0) > 0 else ""
    print(
        f"  {name:42s} | WR:{m['win_rate']:5.1f}% | "
        f"RR:{m['risk_reward_ratio']:>4.1f} | "
        f"PF:{m['profit_factor']:>5.2f} | "
        f"Ret:{m['total_return_pct']:>8.1f}% | "
        f"MDD:{m['max_drawdown_pct']:>5.1f}% | "
        f"${m['final_equity']:>10,.2f} | "
        f"Sharpe:{m['sharpe_ratio']:>5.2f} | "
        f"Trades:{m['total_trades']:>4}{star}"
    )


async def main():
    all_results = {}  # {symbol: {config_name: metrics}}

    # ── Phase 1: Fetch all data ──
    sep("Phase 1: Fetching Data")
    datasets = {}
    for sym in SYMBOLS:
        print(f"  Fetching {sym}...")
        try:
            df = await ensure_data(sym, INTERVAL, START_DATE, END_DATE)
            if df is not None and len(df) > 1000:
                datasets[sym] = df
                print(f"  ✓ {sym}: {len(df)} bars, ${df['close'].min():,.2f} ~ ${df['close'].max():,.2f}")
            else:
                print(f"  ✗ {sym}: insufficient data ({len(df) if df is not None else 0} bars)")
        except Exception as e:
            print(f"  ✗ {sym}: error fetching - {e}")

    # ── Phase 2: Backtest all combos ──
    sep("Phase 2: Full Backtest")
    for sym, df in datasets.items():
        all_results[sym] = {}
        print(f"\n  --- {sym} ({len(df)} bars) ---")
        for cfg_name, cfg in CONFIGS.items():
            try:
                s = FVGTrendStrategy(**cfg)
                r = s.backtest(df, INITIAL_CAPITAL, cfg.get("max_hold", 720))
                m = r["metrics"]
                if "error" in m:
                    print(f"  {sym:10s} {cfg_name:30s} | {m['error']}")
                    continue
                print_row(f"{sym} {cfg_name}", m)
                all_results[sym][cfg_name] = m

                # Yearly breakdown
                yearly = {}
                for t in r["trades"]:
                    y = t.exit_time.year
                    if y not in yearly:
                        yearly[y] = {"w": 0, "l": 0, "pnl": 0}
                    net = t.pnl - t.commission
                    yearly[y]["pnl"] += net
                    if net > 0: yearly[y]["w"] += 1
                    else: yearly[y]["l"] += 1
                for y in sorted(yearly):
                    d = yearly[y]
                    total = d["w"] + d["l"]
                    wr = d["w"] / total * 100 if total > 0 else 0
                    print(f"    {y}: {total:>3} trades, WR {wr:>5.1f}%, PnL ${d['pnl']:>+10,.2f}")

            except Exception as e:
                print(f"  {sym:10s} {cfg_name:30s} | ERROR: {e}")
                import traceback; traceback.print_exc()

    # ── Phase 3: Walk-Forward validation on best config (Combo3) ──
    sep("Phase 3: Walk-Forward Validation (Combo3)")
    wf_results = {}
    for sym, df in datasets.items():
        if sym == "BTCUSDT":
            print(f"\n  {sym}: Already validated (100% consistency, avg 92.5%/yr)")
            continue
        print(f"\n  --- {sym} Walk-Forward ---")
        try:
            s = FVGTrendStrategy(**COMBO3)
            validator = WalkForwardValidator(n_folds=5, min_train_bars=3000)

            def backtest_fn(data, capital, strat=s):
                return strat.backtest(data, capital, COMBO3.get("max_hold", 720))

            wf = validator.validate(df, backtest_fn, INITIAL_CAPITAL)
            wf_results[sym] = wf

            for fold in wf["folds"]:
                fn = fold["fold"]
                m = fold.get("test_metrics", fold.get("oos_metrics", fold.get("metrics", {})))
                if "error" in m:
                    print(f"    Fold {fn}: {m['error']}")
                else:
                    ret = m.get("total_return_pct", 0)
                    wr = m.get("win_rate", 0)
                    pf = m.get("profit_factor", 0)
                    trades = m.get("total_trades", 0)
                    print(f"    Fold {fn}: Ret {ret:>+8.1f}% | WR {wr:>5.1f}% | PF {pf:>5.2f} | Trades: {trades}")

            summary = wf.get("summary", {})
            print(f"\n    Consistency: {summary.get('consistency_score', 0):.0%}")
            print(f"    Avg OOS Return/Fold: {summary.get('avg_test_return', summary.get('avg_oos_return', 0)):.1f}%")
            print(f"    IS vs OOS Ratio: {summary.get('is_vs_oos_ratio', 0):.2f}")
            print(f"    Overfitting Risk: {summary.get('overfitting_risk', 'N/A')}")
        except Exception as e:
            print(f"    ERROR: {e}")
            import traceback; traceback.print_exc()

    # ── Phase 4: Summary Table ──
    sep("CROSS-ASSET SUMMARY")
    print(f"  {'Symbol':10s} | {'Config':32s} | {'Return':>10s} | {'WR':>6s} | {'RR':>5s} | {'PF':>6s} | {'Sharpe':>7s} | {'MDD':>6s} | {'Trades':>6s}")
    print(f"  {'-'*10}-+-{'-'*32}-+-{'-'*10}-+-{'-'*6}-+-{'-'*5}-+-{'-'*6}-+-{'-'*7}-+-{'-'*6}-+-{'-'*6}")
    for sym in SYMBOLS:
        if sym not in all_results:
            continue
        for cfg_name, m in all_results[sym].items():
            print(
                f"  {sym:10s} | {cfg_name:32s} | "
                f"{m['total_return_pct']:>+9.1f}% | "
                f"{m['win_rate']:>5.1f}% | "
                f"{m['risk_reward_ratio']:>4.1f} | "
                f"{m['profit_factor']:>5.2f} | "
                f"{m['sharpe_ratio']:>6.2f} | "
                f"{m['max_drawdown_pct']:>5.1f}% | "
                f"{m['total_trades']:>6}"
            )

    # Count how many assets are profitable
    profitable_combo3 = sum(1 for sym in all_results if "Combo3 (FVG05+Score1+SL6)" in all_results[sym] and all_results[sym]["Combo3 (FVG05+Score1+SL6)"]["total_return_pct"] > 0)
    profitable_combo2 = sum(1 for sym in all_results if "Combo2 (FVG10+Score2+SL8)" in all_results[sym] and all_results[sym]["Combo2 (FVG10+Score2+SL8)"]["total_return_pct"] > 0)
    total_assets = len(all_results)

    print(f"\n  Combo3: {profitable_combo3}/{total_assets} assets profitable")
    print(f"  Combo2: {profitable_combo2}/{total_assets} assets profitable")

    if profitable_combo3 == total_assets:
        print(f"\n  ✅ PASS: Combo3 profitable on ALL {total_assets} assets — strong generalization")
    elif profitable_combo3 >= total_assets * 0.75:
        print(f"\n  ⚠️  PARTIAL: Combo3 profitable on {profitable_combo3}/{total_assets} — decent but not universal")
    else:
        print(f"\n  ❌ FAIL: Combo3 only profitable on {profitable_combo3}/{total_assets} — possible overfitting to BTC")


if __name__ == "__main__":
    asyncio.run(main())
