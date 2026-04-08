"""
FVCL3 V2 Trend Follow + IS/OOS Block Bootstrap
"""
from __future__ import annotations
import argparse, asyncio, logging
import numpy as np, pandas as pd
from data.historical_fetcher import ensure_data
from strategies.fvcl3_v2 import FVCL3V2Strategy
from strategies.bootstrap_validator import BlockBootstrapValidator

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(name)-20s | %(levelname)-7s | %(message)s", datefmt="%H:%M:%S")

INITIAL_CAPITAL = 2000.0
START_DATE = "2021-04-01"
END_DATE = "2026-04-01"
SYMBOL = "BTCUSDT"
INTERVAL = "1h"
MAX_HOLDING = 168


def sep(title):
    print(f"\n{'='*80}\n  {title}\n{'='*80}")


def print_metrics(name, m):
    print(f"\n  --- {name} ---")
    for k, label in [
        ("total_trades", "Trades"), ("win_rate", "WR%"), ("total_pnl", "PnL$"),
        ("total_return_pct", "Ret%"), ("annual_return_pct", "AnnRet%"),
        ("max_drawdown_pct", "MDD%"), ("sharpe_ratio", "Sharpe"),
        ("sortino_ratio", "Sortino"), ("calmar_ratio", "Calmar"),
        ("profit_factor", "PF"), ("risk_reward_ratio", "RR"),
        ("avg_win", "AvgWin$"), ("avg_loss", "AvgLoss$"),
        ("best_trade", "Best$"), ("worst_trade", "Worst$"),
        ("trades_per_day", "Trades/Day"), ("max_win_streak", "MaxWin"),
        ("max_lose_streak", "MaxLose"), ("total_commission", "Comm$"),
        ("final_equity", "Final$"), ("exit_reasons", "Exits"),
    ]:
        v = m.get(k, "N/A")
        if isinstance(v, float):
            print(f"  {label:15s} {v:>12.2f}")
        else:
            print(f"  {label:15s} {v}")


def ov(base, **kw):
    return {**base, **kw}


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bootstrap", action="store_true")
    parser.add_argument("-n", type=int, default=1000)
    args = parser.parse_args()

    sep(f"FVCL3 V2 Trend Follow - {SYMBOL}")
    print(f"  Period: {START_DATE} -> {END_DATE}, Capital: ${INITIAL_CAPITAL:,.0f}")

    df = await ensure_data(SYMBOL, INTERVAL, START_DATE, END_DATE)
    print(f"  Data: {len(df)} bars, ${df['close'].min():,.0f} ~ ${df['close'].max():,.0f}")

    sep("Parameter Sweep")

    NT = {"trail_start_atr": 99.0, "breakeven_atr": 99.0}
    B3 = {**NT, "daily_ema_fast": 10, "daily_ema_slow": 20, "sl_atr": 5.0, "h4_rsi_pullback": 65}
    B6 = {**NT, "daily_ema_fast": 10, "daily_ema_slow": 20, "h4_rsi_pullback": 55,
          "entry_min_score": 1, "sl_atr": 6.0, "h1_rsi_os": 45, "cooldown": 6}

    configs = [
        ("C3 base", B3),
        ("C3 SL=4", ov(B3, sl_atr=4.0)),
        ("C3 SL=5.5", ov(B3, sl_atr=5.5)),
        ("C3 SL=6", ov(B3, sl_atr=6.0)),
        ("C3 SL=7", ov(B3, sl_atr=7.0)),
        ("C3 4hPB=50", ov(B3, h4_rsi_pullback=50)),
        ("C3 4hPB=58", ov(B3, h4_rsi_pullback=58)),
        ("C3 4hPB=60", ov(B3, h4_rsi_pullback=60)),
        ("C3 4hPB=65", ov(B3, h4_rsi_pullback=65)),
        ("C3 D8/18", ov(B3, daily_ema_fast=8, daily_ema_slow=18)),
        ("C3 D12/25", ov(B3, daily_ema_fast=12, daily_ema_slow=25)),
        ("C3 D15/30", ov(B3, daily_ema_fast=15, daily_ema_slow=30)),
        ("C3 Score=1", ov(B3, entry_min_score=1)),
        ("C3 Score=3", ov(B3, entry_min_score=3)),
        ("C3 1hRSI=40", ov(B3, h1_rsi_os=40)),
        ("C3 1hRSI=45", ov(B3, h1_rsi_os=45)),
        ("C3 CD=6", ov(B3, cooldown=6)),
        ("C3 CD=24", ov(B3, cooldown=24)),
        ("C3 Hold=1440", ov(B3, max_hold=1440)),
        ("C6 base", B6),
        ("C6 SL=5", ov(B6, sl_atr=5.0)),
        ("C6 SL=7", ov(B6, sl_atr=7.0)),
        ("C6 4hPB=60", ov(B6, h4_rsi_pullback=60)),
        ("C6 CD=12", ov(B6, cooldown=12)),
        ("C6 1hRSI=50", ov(B6, h1_rsi_os=50)),
        ("X1: SL6+4hPB60", ov(B3, sl_atr=6.0, h4_rsi_pullback=60)),
        ("X2: 4hPB60+S1", ov(B3, h4_rsi_pullback=60, entry_min_score=1)),
        ("X3: D8/18+S1", ov(B3, daily_ema_fast=8, daily_ema_slow=18, entry_min_score=1)),
        ("X4: SL6+4hPB60+RSI45", ov(B3, sl_atr=6.0, h4_rsi_pullback=60, h1_rsi_os=45)),
        ("X5: 4hPB60+H1440", ov(B3, h4_rsi_pullback=60, max_hold=1440)),
        ("X6: SL6+4hPB60+S1+R45", ov(B3, sl_atr=6.0, h4_rsi_pullback=60, entry_min_score=1, h1_rsi_os=45)),
        ("X7: SL6+4hPB60+CD6", ov(B3, sl_atr=6.0, h4_rsi_pullback=60, cooldown=6)),
    ]

    results = []
    for name, kwargs in configs:
        s = FVCL3V2Strategy(**kwargs)
        try:
            r = s.backtest(df, INITIAL_CAPITAL, MAX_HOLDING)
        except Exception as e:
            print(f"  {name:30s} | ERROR: {e}")
            continue

        m = r["metrics"]
        if "error" in m:
            print(f"  {name:30s} | {m['error']}")
            continue

        star = " ***" if m["total_return_pct"] > 0 else ""
        print(
            f"  {name:30s} | WR:{m['win_rate']:5.1f}% | "
            f"RR:{m['risk_reward_ratio']:>4.1f} | "
            f"PF:{m['profit_factor']:>5.2f} | "
            f"Ret:{m['total_return_pct']:>8.1f}% | "
            f"MDD:{m['max_drawdown_pct']:>5.1f}% | "
            f"${m['final_equity']:>9,.2f} | "
            f"Sharpe:{m['sharpe_ratio']:>5.2f} | "
            f"Trades:{m['total_trades']:>4}{star}"
        )
        results.append((name, m, r, s))

    if results:
        best_name, best_m, best_r, best_s = max(results, key=lambda x: x[1]["total_return_pct"])
        sep(f"Best: {best_name}")
        print_metrics(best_name, best_m)

        print(f"\n  Yearly Breakdown:")
        yearly = {}
        for t in best_r["trades"]:
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
            print(f"    {y}: {total:>4} trades, WR {wr:>5.1f}%, PnL ${d['pnl']:>+10,.2f}")

        if args.bootstrap:
            sep(f"IS/OOS Block Bootstrap (n={args.n})")
            validator = BlockBootstrapValidator(is_ratio=0.7, block_size=10, n_iterations=args.n)

            def backtest_fn(data, capital):
                return best_s.backtest(data, capital, MAX_HOLDING)

            result = validator.validate(df, backtest_fn, INITIAL_CAPITAL)
            print(f"\n  IS:  {result['is_period']}")
            print(f"  OOS: {result['oos_period']}")
            print(f"\n  IS  Ret: {result['is_metrics']['total_return_pct']:.1f}% | WR: {result['is_metrics']['win_rate']:.1f}% | PF: {result['is_metrics']['profit_factor']:.2f}")
            print(f"  OOS Ret: {result['oos_metrics']['total_return_pct']:.1f}% | WR: {result['oos_metrics']['win_rate']:.1f}% | PF: {result['oos_metrics']['profit_factor']:.2f}")

            bd = result["bootstrap_distribution"]
            print(f"\n  Bootstrap ({bd['n_samples']}x): Mean={bd['mean']:.1f}% Median={bd['median']:.1f}% Std={bd['std']:.1f}%")
            print(f"  5%ile={bd['pct_5']:.1f}% 95%ile={bd['pct_95']:.1f}%")
            print(f"  OOS Percentile: {bd['actual_percentile_rank']:.1f}%")
            print(f"  Profitable: {bd['pct_profitable']:.1f}%")
            print(f"\n  p-value: {result['p_value']:.4f}")
            sig = "SIGNIFICANT" if result["is_significant"] else "NOT significant"
            print(f"  Result: {sig} (a=0.05)")


if __name__ == "__main__":
    asyncio.run(main())
