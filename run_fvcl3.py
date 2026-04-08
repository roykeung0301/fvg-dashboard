"""
FVCL3 策略回測 + IS/OOS Block Bootstrap 驗證

使用方式:
    python run_fvcl3.py                  # 基本回測
    python run_fvcl3.py --bootstrap      # 含 Bootstrap 驗證
    python run_fvcl3.py --bootstrap -n 5000  # 5000 次迭代
"""

from __future__ import annotations

import argparse
import asyncio
import logging

import numpy as np
import pandas as pd

from data.historical_fetcher import ensure_data
from strategies.fvcl3_regime import FVCL3RegimeStrategy
from strategies.bootstrap_validator import BlockBootstrapValidator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)-20s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)

INITIAL_CAPITAL = 2000.0
START_DATE = "2021-04-01"
END_DATE = "2026-04-01"
SYMBOL = "BTCUSDT"
INTERVAL = "1h"
MAX_HOLDING = 168  # Swing: 7 days


def sep(title):
    print(f"\n{'='*80}")
    print(f"  {title}")
    print(f"{'='*80}")


def print_metrics(name, m):
    print(f"\n  --- {name} ---")
    print(f"  總交易數:       {m['total_trades']}")
    print(f"  勝率:           {m['win_rate']:.1f}%")
    print(f"  總損益:         ${m['total_pnl']:,.2f}")
    print(f"  總回報:         {m['total_return_pct']:.1f}%")
    print(f"  年化回報:       {m['annual_return_pct']:.1f}%")
    print(f"  最大回撤:       {m['max_drawdown_pct']:.1f}%")
    print(f"  Sharpe:         {m['sharpe_ratio']:.2f}")
    print(f"  Sortino:        {m['sortino_ratio']:.2f}")
    print(f"  Calmar:         {m['calmar_ratio']:.2f}")
    print(f"  Profit Factor:  {m['profit_factor']:.2f}")
    print(f"  風險報酬比:     {m['risk_reward_ratio']:.2f}")
    print(f"  平均獲利:       ${m['avg_win']:.2f}")
    print(f"  平均虧損:       ${m['avg_loss']:.2f}")
    print(f"  最佳交易:       ${m['best_trade']:.2f}")
    print(f"  最差交易:       ${m['worst_trade']:.2f}")
    print(f"  每日交易數:     {m['trades_per_day']:.2f}")
    print(f"  最大連勝:       {m['max_win_streak']}")
    print(f"  最大連敗:       {m['max_lose_streak']}")
    print(f"  總手續費:       ${m['total_commission']:.2f}")
    print(f"  最終資金:       ${m['final_equity']:,.2f}")
    print(f"  出場分佈:       {m['exit_reasons']}")
    if "trend_trade_count" in m:
        print(f"  趨勢模式出場:   {m['trend_trade_count']}")
        print(f"  震盪模式出場:   {m['range_trade_count']}")


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bootstrap", action="store_true", help="啟用 Block Bootstrap 驗證")
    parser.add_argument("-n", type=int, default=1000, help="Bootstrap 迭代次數")
    args = parser.parse_args()

    sep(f"FVCL3 趨勢/震盪切換策略 — {SYMBOL} Swing Trading")
    print(f"  期間: {START_DATE} → {END_DATE} (5 年)")
    print(f"  初始資金: ${INITIAL_CAPITAL:,.2f}")
    print(f"  時間框架: {INTERVAL}")
    print(f"  最大持倉: {MAX_HOLDING}h ({MAX_HOLDING // 24} 天)")

    # 下載數據
    df = await ensure_data(SYMBOL, INTERVAL, START_DATE, END_DATE)
    print(f"\n  數據: {len(df)} K 線")
    print(f"  價格: ${df['close'].min():,.2f} ~ ${df['close'].max():,.2f}")

    # ── 參數掃描 ──
    sep("參數掃描")

    configs = [
        ("基準 ADX25/20 Chop38/62", 25, 20, 38.2, 61.8, 20, 2.0, 2.5, 25, 75, 1.0, 2.0, 6),
        ("ADX25/18 Chop40/60",       25, 18, 40, 60, 20, 2.0, 2.5, 25, 75, 1.0, 2.0, 6),
        ("ADX30/20 Chop38/62",       30, 20, 38.2, 61.8, 20, 2.0, 2.5, 25, 75, 1.0, 2.0, 6),
        ("ADX25/20 寬SL=3.0",       25, 20, 38.2, 61.8, 20, 2.5, 3.0, 25, 75, 1.2, 2.5, 6),
        ("ADX25/20 緊RSI20/80",     25, 20, 38.2, 61.8, 20, 2.0, 2.5, 20, 80, 1.0, 2.0, 6),
        ("ADX25/20 Lock=12",        25, 20, 38.2, 61.8, 20, 2.0, 2.5, 25, 75, 1.0, 2.0, 12),
        ("ADX25/20 DC=30",          25, 20, 38.2, 61.8, 30, 2.0, 2.5, 25, 75, 1.0, 2.0, 6),
        ("ADX20/15 寬鬆",           20, 15, 40, 60, 20, 2.0, 2.5, 30, 70, 1.0, 2.0, 6),
        ("ADX25/20 Trail=1.5",      25, 20, 38.2, 61.8, 20, 1.5, 2.5, 25, 75, 1.0, 2.0, 6),
        ("ADX25/20 Trail=2.5",      25, 20, 38.2, 61.8, 20, 2.5, 2.5, 25, 75, 1.0, 2.0, 6),
    ]

    results = []
    for name, adx_t, adx_r, chop_t, chop_r, dc, trail, sl, rsi_l, rsi_h, tp_r, sl_r, lock in configs:
        s = FVCL3RegimeStrategy(
            adx_trend_threshold=adx_t, adx_range_threshold=adx_r,
            chop_trend_threshold=chop_t, chop_range_threshold=chop_r,
            trend_entry_breakout=dc,
            trend_trailing_atr=trail, trend_sl_atr=sl,
            range_rsi_oversold=rsi_l, range_rsi_overbought=rsi_h,
            range_tp_atr=tp_r, range_sl_atr=sl_r,
            regime_lock_bars=lock,
        )
        r = s.backtest(df, INITIAL_CAPITAL, MAX_HOLDING)
        m = r["metrics"]
        if "error" not in m:
            star = " ***" if m["total_return_pct"] > 0 else ""
            print(
                f"  {name:30s} | WR:{m['win_rate']:5.1f}% | "
                f"PF:{m['profit_factor']:>5.2f} | "
                f"Ret:{m['total_return_pct']:>8.1f}%{star} | "
                f"MDD:{m['max_drawdown_pct']:>5.1f}% | "
                f"${m['final_equity']:>9,.2f} | "
                f"Sharpe:{m['sharpe_ratio']:>5.2f} | "
                f"Trades:{m['total_trades']:>4} | "
                f"T:{m.get('trend_trade_count',0)} R:{m.get('range_trade_count',0)}"
            )
            results.append((name, m, r, s))

    # 最佳策略詳細報告
    if results:
        best_name, best_m, best_r, best_s = max(results, key=lambda x: x[1]["total_return_pct"])
        sep(f"最佳策略詳細: {best_name}")
        print_metrics(best_name, best_m)

        # 年度分解
        print(f"\n  年度分解:")
        yearly = {}
        for t in best_r["trades"]:
            y = t.exit_time.year
            if y not in yearly:
                yearly[y] = {"w": 0, "l": 0, "pnl": 0}
            net = t.pnl - t.commission
            yearly[y]["pnl"] += net
            if net > 0:
                yearly[y]["w"] += 1
            else:
                yearly[y]["l"] += 1

        for y in sorted(yearly):
            d = yearly[y]
            total = d["w"] + d["l"]
            wr = d["w"] / total * 100 if total > 0 else 0
            print(f"    {y}: {total:>4} trades, WR {wr:>5.1f}%, PnL ${d['pnl']:>+10,.2f}")

        # ── Bootstrap 驗證 ──
        if args.bootstrap:
            sep(f"IS/OOS Block Bootstrap 驗證 (n={args.n})")

            validator = BlockBootstrapValidator(
                is_ratio=0.7,
                block_size=10,
                n_iterations=args.n,
            )

            def backtest_fn(data, capital):
                return best_s.backtest(data, capital, MAX_HOLDING)

            result = validator.validate(df, backtest_fn, INITIAL_CAPITAL)

            print(f"\n  IS 期間:  {result['is_period']}")
            print(f"  OOS 期間: {result['oos_period']}")
            print(f"\n  IS 績效:")
            print(f"    回報: {result['is_metrics']['total_return_pct']:.1f}%")
            print(f"    勝率: {result['is_metrics']['win_rate']:.1f}%")
            print(f"    PF:   {result['is_metrics']['profit_factor']:.2f}")

            print(f"\n  OOS 績效:")
            print(f"    回報: {result['oos_metrics']['total_return_pct']:.1f}%")
            print(f"    勝率: {result['oos_metrics']['win_rate']:.1f}%")
            print(f"    PF:   {result['oos_metrics']['profit_factor']:.2f}")

            bd = result["bootstrap_distribution"]
            print(f"\n  Bootstrap 分佈 ({bd['n_samples']} 次):")
            print(f"    平均:   {bd['mean']:.1f}%")
            print(f"    中位數: {bd['median']:.1f}%")
            print(f"    標準差: {bd['std']:.1f}%")
            print(f"    5%ile:  {bd['pct_5']:.1f}%")
            print(f"    95%ile: {bd['pct_95']:.1f}%")
            print(f"    OOS 在分佈中排名: {bd['actual_percentile_rank']:.1f}%")
            print(f"    可獲利比例: {bd['pct_profitable']:.1f}%")

            print(f"\n  p-value: {result['p_value']:.4f}")
            sig = "✓ 統計顯著" if result["is_significant"] else "✗ 不顯著"
            print(f"  結論: {sig} (α={1-0.95})")


if __name__ == "__main__":
    asyncio.run(main())
