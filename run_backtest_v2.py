"""
V2 回測: 針對 85%+ 勝率優化的策略

測試不同參數組合:
- 不同的 TP/SL 比例
- 不同的確認數門檻
- 不同的 RSI 極值
"""

from __future__ import annotations

import asyncio
import logging

import numpy as np
import pandas as pd

from data.historical_fetcher import ensure_data
from strategies.scalp_reversion_v2 import ScalpReversionV2

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)-20s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("backtest_v2")

INITIAL_CAPITAL = 2000.0
START_DATE = "2021-04-01"
END_DATE = "2026-04-01"
SYMBOL = "BTCUSDT"
INTERVAL = "1h"
MAX_HOLDING_BARS = 24


def print_result(name: str, metrics: dict):
    wr = metrics["win_rate"]
    star = " ***" if wr >= 85 else (" ** " if wr >= 80 else "")
    print(
        f"  {name:45s} | "
        f"WR: {wr:5.1f}%{star} | "
        f"Trades: {metrics['total_trades']:>4} | "
        f"PF: {metrics['profit_factor']:>5.2f} | "
        f"Return: {metrics['total_return_pct']:>7.1f}% | "
        f"MDD: {metrics['max_drawdown_pct']:>5.1f}% | "
        f"Final: ${metrics['final_equity']:>9,.2f} | "
        f"Sharpe: {metrics['sharpe_ratio']:>5.2f} | "
        f"Exit: tp={metrics['exit_reasons'].get('tp',0)} sl={metrics['exit_reasons'].get('sl',0)} "
        f"to={metrics['exit_reasons'].get('timeout',0)}"
    )


async def main():
    print(f"\n{'='*120}")
    print(f"  高勝率策略參數掃描 — {SYMBOL} — ${INITIAL_CAPITAL} — 5 年")
    print(f"{'='*120}")

    df = await ensure_data(SYMBOL, INTERVAL, START_DATE, END_DATE)
    if df.empty:
        print("ERROR: 無數據")
        return

    print(f"  數據: {len(df)} K 線 | {df.index[0]} → {df.index[-1]}\n")

    # ── 參數組合 ──
    configs = [
        # (名稱, tp_atr, sl_atr, rsi_low, rsi_high, min_confirms, bb_std)
        ("基準: TP0.5/SL3.0 RSI15/85 3確認",   0.5, 3.0, 15, 85, 3, 2.5),
        ("TP0.3/SL3.0 RSI15/85 3確認",         0.3, 3.0, 15, 85, 3, 2.5),
        ("TP0.3/SL4.0 RSI15/85 3確認",         0.3, 4.0, 15, 85, 3, 2.5),
        ("TP0.5/SL4.0 RSI15/85 3確認",         0.5, 4.0, 15, 85, 3, 2.5),
        ("TP0.5/SL5.0 RSI15/85 3確認",         0.5, 5.0, 15, 85, 3, 2.5),
        ("TP0.3/SL3.0 RSI10/90 3確認",         0.3, 3.0, 10, 90, 3, 2.5),
        ("TP0.3/SL4.0 RSI10/90 3確認",         0.3, 4.0, 10, 90, 3, 2.5),
        ("TP0.5/SL4.0 RSI10/90 3確認",         0.5, 4.0, 10, 90, 3, 2.5),
        ("TP0.3/SL5.0 RSI10/90 3確認",         0.3, 5.0, 10, 90, 3, 2.5),
        ("TP0.5/SL3.0 RSI10/90 2確認",         0.5, 3.0, 10, 90, 2, 2.5),
        ("TP0.3/SL4.0 RSI10/90 2確認",         0.3, 4.0, 10, 90, 2, 2.5),
        ("TP0.5/SL4.0 RSI10/90 2確認",         0.5, 4.0, 10, 90, 2, 2.5),
        ("TP0.3/SL3.0 RSI10/90 4確認",         0.3, 3.0, 10, 90, 4, 2.5),
        ("TP0.5/SL5.0 RSI10/90 4確認",         0.5, 5.0, 10, 90, 4, 2.5),
        ("TP0.3/SL4.0 RSI10/90 4確認 BB3.0",   0.3, 4.0, 10, 90, 4, 3.0),
        ("TP0.5/SL5.0 RSI10/90 4確認 BB3.0",   0.5, 5.0, 10, 90, 4, 3.0),
        ("TP0.4/SL4.0 RSI12/88 3確認",         0.4, 4.0, 12, 88, 3, 2.5),
        ("TP0.4/SL5.0 RSI12/88 3確認",         0.4, 5.0, 12, 88, 3, 2.5),
        ("TP0.6/SL5.0 RSI10/90 3確認",         0.6, 5.0, 10, 90, 3, 2.5),
        ("TP0.8/SL5.0 RSI10/90 3確認",         0.8, 5.0, 10, 90, 3, 2.5),
    ]

    results = []
    for name, tp, sl, rsi_l, rsi_h, confirms, bb in configs:
        strategy = ScalpReversionV2(
            tp_atr_mult=tp,
            sl_atr_mult=sl,
            rsi_extreme_low=rsi_l,
            rsi_extreme_high=rsi_h,
            min_confirmations=confirms,
            bb_std=bb,
        )
        result = strategy.backtest(df, INITIAL_CAPITAL, MAX_HOLDING_BARS)
        m = result["metrics"]

        if "error" not in m and m["total_trades"] >= 20:
            print_result(name, m)
            results.append((name, m, result))

    # ── 找出勝率 >= 85% 的策略 ──
    print(f"\n{'='*120}")
    print("  勝率 >= 80% 的策略:")
    print(f"{'='*120}")

    high_wr = [(n, m, r) for n, m, r in results if m["win_rate"] >= 80]
    if not high_wr:
        print("  無策略達到 80%+ 勝率。最佳勝率:")
        best = max(results, key=lambda x: x[1]["win_rate"])
        print_result(best[0], best[1])
    else:
        high_wr.sort(key=lambda x: -x[1]["win_rate"])
        for n, m, r in high_wr:
            print_result(n, m)

        # 在 80%+ 中選出最佳 (綜合評分)
        print(f"\n  80%+ 中綜合最佳:")
        best = max(high_wr, key=lambda x: (
            x[1]["win_rate"] * 0.3 +
            min(x[1]["profit_factor"], 5) * 20 * 0.25 +
            min(x[1]["total_return_pct"] + 100, 200) * 0.25 +
            (100 - min(x[1]["max_drawdown_pct"], 100)) * 0.2
        ))
        print_result(best[0], best[1])

        # 年度分解
        trades = best[2]["trades"]
        print(f"\n  年度分解 [{best[0]}]:")
        yearly = {}
        for t in trades:
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
            print(f"    {y}: {total:>3} trades, WR {wr:>5.1f}%, PnL ${d['pnl']:>+9,.2f}")


if __name__ == "__main__":
    asyncio.run(main())
