"""
Detailed Backtest Report — FVG Trend Follow Combo3
Generates equity curve charts + trade-by-trade details for BTC & ETH (2 years)
"""
from __future__ import annotations
import asyncio, logging, os
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import FancyArrowPatch
from data.historical_fetcher import ensure_data
from strategies.fvg_trend import FVGTrendStrategy

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(name)-20s | %(levelname)-7s | %(message)s", datefmt="%H:%M:%S")

INITIAL_CAPITAL = 2000.0
START_DATE = "2024-04-01"
END_DATE   = "2026-04-01"
INTERVAL   = "1h"
OUT_DIR    = "reports"

COMBO3 = dict(
    daily_ema_fast=10, daily_ema_slow=20,
    fvg_min_size_pct=0.05, entry_min_score=1, sl_atr=6.0,
    fvg_max_age=100, max_active_fvgs=30,
    trail_start_atr=99.0, breakeven_atr=99.0,
    cooldown=12, max_hold=720,
)

SYMBOLS = ["BTCUSDT", "ETHUSDT"]


def make_equity_chart(symbol, equity, trades, price_series, out_path):
    """Generate a 2-panel chart: price + entries/exits, equity curve."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(20, 12), gridspec_kw={"height_ratios": [1.2, 1]})
    fig.suptitle(f"FVG Trend Follow Combo3 — {symbol} (2024-04 to 2026-04)", fontsize=16, fontweight="bold")

    # ── Panel 1: Price with trade markers ──
    ax1.plot(price_series.index, price_series.values, color="#555555", linewidth=0.8, alpha=0.9, label="Price")

    # Draw trades
    for t in trades:
        color = "#22c55e" if t.pnl > 0 else "#ef4444"
        marker_entry = "^" if t.side == "long" else "v"
        marker_exit = "x"

        ax1.scatter(t.entry_time, t.entry_price, marker=marker_entry, color=color, s=60, zorder=5, edgecolors="black", linewidths=0.5)
        ax1.scatter(t.exit_time, t.exit_price, marker=marker_exit, color=color, s=50, zorder=5, linewidths=1.5)

        # Connect entry to exit with a line
        ax1.plot([t.entry_time, t.exit_time], [t.entry_price, t.exit_price],
                 color=color, alpha=0.3, linewidth=1, linestyle="--")

    ax1.set_ylabel(f"{symbol} Price (USD)", fontsize=12)
    ax1.legend(["Price", "Win Entry/Exit", "Loss Entry/Exit"], loc="upper left")
    ax1.grid(True, alpha=0.3)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=2))

    # ── Panel 2: Equity curve ──
    eq_color = "#3b82f6"
    ax2.fill_between(equity.index, equity.values[0], equity.values, alpha=0.15, color=eq_color)
    ax2.plot(equity.index, equity.values, color=eq_color, linewidth=1.5, label="Equity")

    # Drawdown shading
    eq_arr = equity.values
    peak = np.maximum.accumulate(eq_arr)
    dd_pct = (peak - eq_arr) / peak * 100
    ax2_dd = ax2.twinx()
    ax2_dd.fill_between(equity.index, 0, dd_pct, alpha=0.15, color="#ef4444")
    ax2_dd.set_ylabel("Drawdown %", color="#ef4444", fontsize=11)
    ax2_dd.set_ylim(0, max(dd_pct) * 3)  # invert visual weight
    ax2_dd.invert_yaxis()
    ax2_dd.tick_params(axis="y", colors="#ef4444")

    ax2.set_ylabel("Equity (USD)", fontsize=12)
    ax2.set_xlabel("Date", fontsize=12)
    ax2.legend(loc="upper left")
    ax2.grid(True, alpha=0.3)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=2))

    # Add key stats as text box
    final_eq = eq_arr[-1]
    start_eq = eq_arr[0]
    total_ret = (final_eq - start_eq) / start_eq * 100
    wins = sum(1 for t in trades if t.pnl - t.commission > 0)
    wr = wins / len(trades) * 100 if trades else 0
    max_dd = max(dd_pct)

    stats_text = (
        f"Final: ${final_eq:,.0f}  |  Return: +{total_ret:.0f}%  |  "
        f"Trades: {len(trades)}  |  WR: {wr:.1f}%  |  Max DD: {max_dd:.1f}%"
    )
    fig.text(0.5, 0.02, stats_text, ha="center", fontsize=12, fontweight="bold",
             bbox=dict(boxstyle="round,pad=0.5", facecolor="#f0f9ff", edgecolor="#3b82f6", alpha=0.9))

    plt.tight_layout(rect=[0, 0.04, 1, 0.96])
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Chart saved: {out_path}")


def make_monthly_chart(symbol, equity, out_path):
    """Monthly returns heatmap-style bar chart."""
    daily = equity.resample("D").last().dropna()
    monthly = daily.resample("ME").last().dropna()
    monthly_ret = monthly.pct_change().dropna() * 100

    fig, ax = plt.subplots(figsize=(18, 5))
    colors = ["#22c55e" if r > 0 else "#ef4444" for r in monthly_ret.values]
    labels = [d.strftime("%Y-%m") for d in monthly_ret.index]
    bars = ax.bar(range(len(monthly_ret)), monthly_ret.values, color=colors, edgecolor="white", linewidth=0.5)

    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Monthly Return %", fontsize=12)
    ax.set_title(f"FVG Trend Follow Combo3 — {symbol} Monthly Returns", fontsize=14, fontweight="bold")
    ax.axhline(y=0, color="black", linewidth=0.8)
    ax.grid(True, alpha=0.3, axis="y")

    # Add value labels on bars
    for bar, val in zip(bars, monthly_ret.values):
        if abs(val) > 2:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                    f"{val:+.0f}%", ha="center", va="bottom" if val > 0 else "top",
                    fontsize=7, fontweight="bold")

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Monthly chart saved: {out_path}")


def print_trades_table(symbol, trades, start_equity=None):
    """Print detailed trade-by-trade table."""
    print(f"\n{'='*130}")
    print(f"  {symbol} — Trade Details ({len(trades)} trades)")
    print(f"{'='*130}")
    print(f"  {'#':>3} | {'Side':>5} | {'Entry Date':>19} | {'Exit Date':>19} | {'Entry$':>10} | {'Exit$':>10} | "
          f"{'P&L$':>10} | {'P&L%':>7} | {'Bars':>5} | {'Exit Reason':>12} | {'Equity$':>10}")
    print(f"  {'-'*3}-+-{'-'*5}-+-{'-'*19}-+-{'-'*19}-+-{'-'*10}-+-{'-'*10}-+-"
          f"{'-'*10}-+-{'-'*7}-+-{'-'*5}-+-{'-'*12}-+-{'-'*10}")

    running_equity = start_equity if start_equity else INITIAL_CAPITAL
    for i, t in enumerate(trades):
        net = t.pnl - t.commission
        running_equity += net
        bars = int((t.exit_time - t.entry_time).total_seconds() / 3600)  # hours
        pnl_marker = "+" if net > 0 else ""

        print(
            f"  {i+1:>3} | {t.side:>5} | {t.entry_time.strftime('%Y-%m-%d %H:%M'):>19} | "
            f"{t.exit_time.strftime('%Y-%m-%d %H:%M'):>19} | "
            f"${t.entry_price:>9,.2f} | ${t.exit_price:>9,.2f} | "
            f"{pnl_marker}${net:>9,.2f} | {pnl_marker}{t.pnl_pct:>5.1f}% | "
            f"{bars:>5} | {t.exit_reason:>12} | ${running_equity:>9,.2f}"
        )

    # Summary stats
    wins = [t for t in trades if t.pnl - t.commission > 0]
    losses = [t for t in trades if t.pnl - t.commission <= 0]
    avg_win_pct = np.mean([t.pnl_pct for t in wins]) if wins else 0
    avg_loss_pct = np.mean([t.pnl_pct for t in losses]) if losses else 0
    best = max(trades, key=lambda t: t.pnl_pct)
    worst = min(trades, key=lambda t: t.pnl_pct)
    avg_bars = np.mean([(t.exit_time - t.entry_time).total_seconds() / 3600 for t in trades])

    print(f"\n  --- Summary ---")
    print(f"  Total Trades: {len(trades)} | Wins: {len(wins)} | Losses: {len(losses)} | WR: {len(wins)/len(trades)*100:.1f}%")
    print(f"  Avg Win:  +{avg_win_pct:.2f}% | Avg Loss: {avg_loss_pct:.2f}%")
    print(f"  Best:  +{best.pnl_pct:.2f}% ({best.entry_time.strftime('%Y-%m-%d')}) | Worst: {worst.pnl_pct:.2f}% ({worst.entry_time.strftime('%Y-%m-%d')})")
    print(f"  Avg Hold: {avg_bars:.0f} hours ({avg_bars/24:.1f} days)")
    print(f"  Final Equity: ${running_equity:,.2f} | Return: +{(running_equity-INITIAL_CAPITAL)/INITIAL_CAPITAL*100:.1f}%")

    # Exit reason breakdown
    reasons = {}
    for t in trades:
        reasons[t.exit_reason] = reasons.get(t.exit_reason, 0) + 1
    print(f"  Exit Reasons: {reasons}")


async def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    for symbol in SYMBOLS:
        print(f"\n{'#'*90}")
        print(f"  Processing {symbol}...")
        print(f"{'#'*90}")

        df_full = await ensure_data(symbol, INTERVAL, "2021-04-01", END_DATE)
        # Filter to 2-year window — but keep some lookback for indicator warmup
        warmup_start = "2024-01-01"  # 3 months warmup before 2024-04
        df = df_full[df_full.index >= warmup_start].copy()
        print(f"  Data: {len(df)} bars (with warmup), {df.index[0]} -> {df.index[-1]}")
        print(f"  Price range: ${df['close'].min():,.2f} ~ ${df['close'].max():,.2f}")

        s = FVGTrendStrategy(**COMBO3)
        result = s.backtest(df, INITIAL_CAPITAL, COMBO3["max_hold"])
        trades_all = result["trades"]
        equity_all = result["equity_curve"]

        # Filter to display window (2024-04 onwards)
        cutoff = pd.Timestamp(START_DATE)
        trades = [t for t in trades_all if t.entry_time >= cutoff]
        equity = equity_all[equity_all.index >= cutoff]
        price_display = df["close"][df.index >= cutoff]

        # Recalculate metrics for display period
        if trades:
            from strategies.base_strategy import BaseStrategy
            metrics = BaseStrategy._compute_metrics(trades, equity, equity.iloc[0])
        else:
            metrics = result["metrics"]

        # Print trade table
        start_eq = equity.iloc[0]
        print_trades_table(symbol, trades, start_equity=start_eq)

        # Generate charts
        make_equity_chart(symbol, equity, trades, price_display, os.path.join(OUT_DIR, f"{symbol}_equity.png"))
        make_monthly_chart(symbol, equity, os.path.join(OUT_DIR, f"{symbol}_monthly.png"))

        print(f"\n  Key Metrics:")
        print(f"  Return: +{metrics['total_return_pct']:.1f}% | WR: {metrics['win_rate']:.1f}% | RR: {metrics['risk_reward_ratio']:.1f}")
        print(f"  PF: {metrics['profit_factor']:.2f} | Sharpe: {metrics['sharpe_ratio']:.2f} | MDD: {metrics['max_drawdown_pct']:.1f}%")


if __name__ == "__main__":
    asyncio.run(main())
