"""
5 年回測: 3 個高勝率 Day Trading 策略比較

使用方式:
    python run_backtest.py
"""

from __future__ import annotations

import asyncio
import logging
import sys

import pandas as pd
import numpy as np

from data.historical_fetcher import ensure_data
from strategies.bb_rsi_reversion import BBRSIReversionStrategy
from strategies.mtf_reversion import MTFReversionStrategy
from strategies.funding_contrarian import FundingContrarianStrategy

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)-20s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("backtest")

INITIAL_CAPITAL = 2000.0
START_DATE = "2021-04-01"
END_DATE = "2026-04-01"
SYMBOL = "BTCUSDT"
INTERVAL = "1h"
MAX_HOLDING_BARS = 24  # Day trading: 最多持倉 24 根 1h K 線 = 1 天


def print_separator(title: str):
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}")


def print_metrics(name: str, metrics: dict):
    """格式化輸出策略績效"""
    print(f"\n  --- {name} ---")
    print(f"  總交易數:       {metrics['total_trades']}")
    print(f"  勝率:           {metrics['win_rate']:.1f}%")
    print(f"  總損益:         ${metrics['total_pnl']:,.2f}")
    print(f"  總回報:         {metrics['total_return_pct']:.1f}%")
    print(f"  年化回報:       {metrics['annual_return_pct']:.1f}%")
    print(f"  最大回撤:       {metrics['max_drawdown_pct']:.1f}%")
    print(f"  Sharpe Ratio:   {metrics['sharpe_ratio']:.2f}")
    print(f"  Sortino Ratio:  {metrics['sortino_ratio']:.2f}")
    print(f"  Calmar Ratio:   {metrics['calmar_ratio']:.2f}")
    print(f"  Profit Factor:  {metrics['profit_factor']:.2f}")
    print(f"  風險報酬比:     {metrics['risk_reward_ratio']:.2f}")
    print(f"  平均獲利:       ${metrics['avg_win']:.2f}")
    print(f"  平均虧損:       ${metrics['avg_loss']:.2f}")
    print(f"  最佳交易:       ${metrics['best_trade']:.2f}")
    print(f"  最差交易:       ${metrics['worst_trade']:.2f}")
    print(f"  每日交易數:     {metrics['trades_per_day']:.2f}")
    print(f"  最大連勝:       {metrics['max_win_streak']}")
    print(f"  最大連敗:       {metrics['max_lose_streak']}")
    print(f"  總手續費:       ${metrics['total_commission']:.2f}")
    print(f"  最終資金:       ${metrics['final_equity']:,.2f}")
    print(f"  出場分佈:       {metrics['exit_reasons']}")


def print_comparison_table(results: dict):
    """比較表格"""
    print_separator("策略比較總表")

    headers = [
        ("策略", 22),
        ("勝率", 8),
        ("總回報%", 10),
        ("年化%", 8),
        ("MDD%", 8),
        ("Sharpe", 8),
        ("PF", 6),
        ("交易數", 8),
        ("最終$", 12),
    ]

    header_line = "  "
    for name, width in headers:
        header_line += f"{name:>{width}}"
    print(header_line)
    print("  " + "-" * sum(w for _, w in headers))

    for strat_name, result in results.items():
        m = result["metrics"]
        line = "  "
        line += f"{strat_name:>22}"
        line += f"{m['win_rate']:>7.1f}%"
        line += f"{m['total_return_pct']:>9.1f}%"
        line += f"{m['annual_return_pct']:>7.1f}%"
        line += f"{m['max_drawdown_pct']:>7.1f}%"
        line += f"{m['sharpe_ratio']:>8.2f}"
        line += f"{m['profit_factor']:>6.2f}"
        line += f"{m['total_trades']:>8}"
        line += f"  ${m['final_equity']:>9,.2f}"
        print(line)


def print_winner(results: dict):
    """選出最佳策略"""
    print_separator("最佳策略推薦")

    # 評分: 勝率 * 0.3 + Sharpe * 0.2 + PF * 0.2 + 年化 * 0.15 + (100-MDD) * 0.15
    scores = {}
    for name, result in results.items():
        m = result["metrics"]
        score = (
            m["win_rate"] * 0.30 +
            min(m["sharpe_ratio"], 5) * 20 * 0.20 +  # Sharpe 標準化到 0~100
            min(m["profit_factor"], 5) * 20 * 0.20 +  # PF 標準化
            min(m["annual_return_pct"], 200) / 2 * 0.15 +  # 年化標準化
            (100 - min(m["max_drawdown_pct"], 100)) * 0.15
        )
        scores[name] = round(score, 2)

    for name, score in sorted(scores.items(), key=lambda x: -x[1]):
        marker = " <-- BEST" if score == max(scores.values()) else ""
        print(f"  {name:25s} 綜合評分: {score:.1f}{marker}")

    best = max(scores, key=scores.get)
    m = results[best]["metrics"]
    print(f"\n  推薦策略: {best}")
    print(f"  理由: 勝率 {m['win_rate']:.1f}%, Sharpe {m['sharpe_ratio']:.2f}, "
          f"年化 {m['annual_return_pct']:.1f}%, MDD {m['max_drawdown_pct']:.1f}%")


def print_yearly_breakdown(results: dict):
    """年度績效分解"""
    print_separator("年度績效分解")

    for strat_name, result in results.items():
        trades = result["trades"]
        if not trades:
            continue

        print(f"\n  [{strat_name}]")

        # 按年份分組
        yearly = {}
        for t in trades:
            year = t.exit_time.year
            if year not in yearly:
                yearly[year] = {"wins": 0, "losses": 0, "pnl": 0}
            net = t.pnl - t.commission
            yearly[year]["pnl"] += net
            if net > 0:
                yearly[year]["wins"] += 1
            else:
                yearly[year]["losses"] += 1

        print(f"  {'年份':>6}  {'交易數':>6}  {'勝率':>7}  {'損益':>12}")
        print(f"  {'-'*40}")
        for year in sorted(yearly.keys()):
            y = yearly[year]
            total = y["wins"] + y["losses"]
            wr = y["wins"] / total * 100 if total > 0 else 0
            print(f"  {year:>6}  {total:>6}  {wr:>6.1f}%  ${y['pnl']:>10,.2f}")


async def main():
    print_separator(f"加密貨幣 Day Trading 回測 — {SYMBOL}")
    print(f"  期間: {START_DATE} → {END_DATE} (5 年)")
    print(f"  初始資金: ${INITIAL_CAPITAL:,.2f}")
    print(f"  時間框架: {INTERVAL}")
    print(f"  最大持倉: {MAX_HOLDING_BARS} 根 K 線")

    # ── 1. 下載數據 ──
    print_separator("下載歷史數據")
    df = await ensure_data(SYMBOL, INTERVAL, START_DATE, END_DATE)

    if df.empty or len(df) < 1000:
        print("  ERROR: 數據不足，無法回測")
        sys.exit(1)

    print(f"  數據量: {len(df)} 根 K 線")
    print(f"  期間:   {df.index[0]} → {df.index[-1]}")
    print(f"  價格:   ${df['close'].min():,.2f} ~ ${df['close'].max():,.2f}")

    # ── 2. 定義策略 ──
    strategies = {
        "BB+RSI 均值回歸": BBRSIReversionStrategy(
            bb_period=20,
            bb_std=2.2,
            rsi_period=14,
            rsi_oversold=25,
            rsi_overbought=75,
            sl_multiplier=1.2,
        ),
        "多時框順勢回歸": MTFReversionStrategy(
            trend_ema=200,
            fast_ema=50,
            rsi_period=10,
            rsi_oversold=20,
            rsi_overbought=80,
            tp_atr_mult=1.0,
            sl_atr_mult=2.0,
        ),
        "情緒反向+VWAP": FundingContrarianStrategy(
            vwap_period=24,
            vwap_dev_threshold=1.5,
            rsi_extreme_low=20,
            rsi_extreme_high=80,
            sentiment_threshold=0.7,
            tp_pct=0.8,
            sl_pct=1.5,
            cooldown=4,
        ),
    }

    # ── 3. 執行回測 ──
    results = {}
    for name, strategy in strategies.items():
        print_separator(f"回測: {name}")
        logger.info(f"開始回測: {name}")
        result = strategy.backtest(df, INITIAL_CAPITAL, MAX_HOLDING_BARS)
        results[name] = result

        if "error" in result["metrics"]:
            print(f"  ERROR: {result['metrics']['error']}")
        else:
            print_metrics(name, result["metrics"])

    # ── 4. 比較 & 推薦 ──
    valid = {k: v for k, v in results.items() if "error" not in v["metrics"]}
    if valid:
        print_comparison_table(valid)
        print_yearly_breakdown(valid)
        print_winner(valid)


if __name__ == "__main__":
    asyncio.run(main())
