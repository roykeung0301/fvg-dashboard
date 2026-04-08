"""V3 回測: 分段止盈 + 保本止損 + 追蹤止損"""

from __future__ import annotations
import asyncio, logging
import pandas as pd, numpy as np
from data.historical_fetcher import ensure_data
from strategies.smart_reversion_v3 import SmartReversionV3

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(name)-20s | %(levelname)-7s | %(message)s", datefmt="%H:%M:%S")

INITIAL_CAPITAL = 2000.0

def pr(name, m):
    wr = m["win_rate"]
    star = " ***" if wr >= 85 else (" ** " if wr >= 80 else "")
    print(
        f"  {name:50s} | WR:{wr:5.1f}%{star} | "
        f"Trades:{m['total_trades']:>4} | PF:{m['profit_factor']:>5.2f} | "
        f"Ret:{m['total_return_pct']:>7.1f}% | MDD:{m['max_drawdown_pct']:>5.1f}% | "
        f"${m['final_equity']:>9,.2f} | Sharpe:{m['sharpe_ratio']:>5.2f} | "
        f"Exit: tp1={m['exit_reasons'].get('tp1',0)} tp2={m['exit_reasons'].get('tp2',0)} "
        f"sl={m['exit_reasons'].get('sl',0)} trail={m['exit_reasons'].get('trail_sl',0)} "
        f"to={m['exit_reasons'].get('timeout',0)}"
    )

async def main():
    print(f"\n{'='*130}")
    print(f"  V3 智能均值回歸 — 分段止盈 + 保本止損 + 追蹤")
    print(f"{'='*130}")

    df = await ensure_data("BTCUSDT", "1h", "2021-04-01", "2026-04-01")
    print(f"  數據: {len(df)} K 線\n")

    configs = [
        # name, tp1, tp2, sl, breakeven, trailing, tp1_pct, confirms, adx_max, rsi_lo, rsi_hi, bb
        ("TP1=0.5/TP2=1.5/SL=2.5 3確認 ADX30",  0.5, 1.5, 2.5, 0.4, 1.0, 0.6, 3, 30, 15, 85, 2.5),
        ("TP1=0.5/TP2=2.0/SL=2.5 3確認 ADX30",  0.5, 2.0, 2.5, 0.4, 1.0, 0.6, 3, 30, 15, 85, 2.5),
        ("TP1=0.5/TP2=2.0/SL=3.0 3確認 ADX25",  0.5, 2.0, 3.0, 0.4, 1.0, 0.6, 3, 25, 15, 85, 2.5),
        ("TP1=0.3/TP2=1.5/SL=2.5 3確認 ADX30",  0.3, 1.5, 2.5, 0.3, 0.8, 0.6, 3, 30, 15, 85, 2.5),
        ("TP1=0.3/TP2=2.0/SL=3.0 3確認 ADX25",  0.3, 2.0, 3.0, 0.3, 1.0, 0.6, 3, 25, 15, 85, 2.5),
        ("TP1=0.5/TP2=2.0/SL=3.0 3確認 ADX25 BB3",0.5,2.0,3.0, 0.4, 1.0, 0.6, 3, 25, 15, 85, 3.0),
        ("TP1=0.4/TP2=1.5/SL=2.0 3確認 ADX30",  0.4, 1.5, 2.0, 0.3, 0.8, 0.6, 3, 30, 12, 88, 2.5),
        ("TP1=0.5/TP2=1.5/SL=2.5 4確認 ADX30",  0.5, 1.5, 2.5, 0.4, 1.0, 0.6, 4, 30, 15, 85, 2.5),
        ("TP1=0.5/TP2=2.0/SL=3.0 4確認 ADX25",  0.5, 2.0, 3.0, 0.4, 1.0, 0.6, 4, 25, 15, 85, 2.5),
        ("TP1=0.3/TP2=2.0/SL=3.0 4確認 ADX25 BB3",0.3,2.0,3.0, 0.3, 1.0, 0.6, 4, 25, 10, 90, 3.0),
        ("TP1=0.5/TP2=3.0/SL=2.5 3確認 ADX30",  0.5, 3.0, 2.5, 0.4, 1.2, 0.5, 3, 30, 15, 85, 2.5),
        ("TP1=0.5/TP2=3.0/SL=3.0 3確認 ADX25",  0.5, 3.0, 3.0, 0.4, 1.2, 0.5, 3, 25, 15, 85, 2.5),
        ("TP1=0.6/TP2=2.0/SL=2.5 3確認 ADX30 70%", 0.6, 2.0, 2.5, 0.5, 1.0, 0.7, 3, 30, 15, 85, 2.5),
        ("TP1=0.4/TP2=2.5/SL=3.0 3確認 ADX25",  0.4, 2.5, 3.0, 0.3, 1.0, 0.5, 3, 25, 12, 88, 2.5),
    ]

    results = []
    for name, tp1, tp2, sl, be, trail, tp1p, conf, adx, rsi_l, rsi_h, bb in configs:
        s = SmartReversionV3(
            tp1_atr=tp1, tp2_atr=tp2, sl_atr=sl,
            breakeven_at=be, trailing_atr=trail, tp1_pct=tp1p,
            min_confirms=conf, adx_max=adx,
            rsi_extreme_low=rsi_l, rsi_extreme_high=rsi_h, bb_std=bb,
        )
        r = s.backtest(df, INITIAL_CAPITAL, 24)
        m = r["metrics"]
        if "error" not in m and m["total_trades"] >= 20:
            pr(name, m)
            results.append((name, m, r))

    # 找出可獲利且高勝率的
    print(f"\n{'='*130}")
    profitable = [(n,m,r) for n,m,r in results if m["total_return_pct"] > 0]
    if profitable:
        print("  可獲利策略:")
        for n,m,r in sorted(profitable, key=lambda x: -x[1]["total_return_pct"]):
            pr(n, m)
    else:
        print("  尚無可獲利策略。按綜合評分排序:")
        scored = sorted(results, key=lambda x: (
            x[1]["profit_factor"] * 30 +
            x[1]["win_rate"] * 0.3 +
            (100 - x[1]["max_drawdown_pct"]) * 0.2 +
            x[1]["total_return_pct"] * 0.1
        ), reverse=True)
        for n,m,r in scored[:5]:
            pr(n, m)

    # 年度分解 (最佳)
    if results:
        best = max(results, key=lambda x: x[1]["profit_factor"])
        trades = best[2]["trades"]
        print(f"\n  年度分解 [{best[0]}]:")
        yearly = {}
        for t in trades:
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
            wr = d["w"]/total*100 if total > 0 else 0
            print(f"    {y}: {total:>4} trades, WR {wr:>5.1f}%, PnL ${d['pnl']:>+10,.2f}")

if __name__ == "__main__":
    asyncio.run(main())
