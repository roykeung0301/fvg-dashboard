"""
Portfolio Backtest — FVG Trend Follow Combo3
Multi-asset simulation with risk-based position sizing

- $2,000 initial capital, shared equity pool
- 2% risk per trade (position_size = risk_amount / sl_distance)
- Per-asset limits: BTC 35%, ETH 30%, SOL 20%, XRP 20%
- Total crypto exposure <= 70%
- Long + Short allowed, 1x spot only
"""
from __future__ import annotations
import asyncio, logging, os
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from data.historical_fetcher import ensure_data
from strategies.fvg_trend import FVGTrendStrategy
from strategies.base_strategy import TradeRecord

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(name)-20s | %(levelname)-7s | %(message)s", datefmt="%H:%M:%S")

# ── Config ──
INITIAL_CAPITAL = 5000.0
START_DATE = "2024-04-01"
END_DATE   = "2026-04-01"
INTERVAL   = "1h"
OUT_DIR    = "reports"
RISK_PER_TRADE = 0.02       # 2% equity risk per trade
MAX_TOTAL_EXPOSURE = 0.70   # 70% max total exposure

COMBO3 = dict(
    daily_ema_fast=10, daily_ema_slow=20,
    fvg_min_size_pct=0.10, entry_min_score=2, sl_atr=10.0,
    fvg_max_age=75, max_active_fvgs=30,
    trail_start_atr=99.0, trail_atr=5.0, breakeven_atr=99.0,
    cooldown=12, max_hold=360,
)

SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT"]
ASSET_LIMITS = {
    "BTCUSDT": 0.35,
    "ETHUSDT": 0.30,
    "SOLUSDT": 0.20,
    "XRPUSDT": 0.20,
}

COMMISSION_RATE = 0.0004
SLIPPAGE_RATE = 0.0001


def generate_all_signals(dfs: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """Run FVG strategy signal generation for each symbol."""
    strategy = FVGTrendStrategy(**COMBO3)
    result = {}
    for sym, df in dfs.items():
        result[sym] = strategy.generate_signals(df.copy())
    return result


def run_portfolio_backtest(signal_dfs: dict[str, pd.DataFrame]):
    """
    Unified multi-asset backtest:
    - All assets share the same equity pool
    - Risk-based position sizing: risk_amount = equity * 2%, qty = risk_amount / |entry - sl|
    - Per-asset and total exposure limits enforced
    """
    # Build unified timeline
    all_indices = set()
    for df in signal_dfs.values():
        all_indices.update(df.index)
    timeline = sorted(all_indices)

    equity = INITIAL_CAPITAL
    positions = {}  # symbol -> {side, entry_price, quantity, entry_idx_time, sl, atr, notional_at_entry}
    all_trades = []  # (symbol, TradeRecord)
    equity_curve = []
    equity_times = []

    # Pre-index dataframes for fast lookup
    sym_data = {}
    for sym, df in signal_dfs.items():
        sym_data[sym] = df

    for ts in timeline:
        # Process each symbol at this timestamp
        for sym in SYMBOLS:
            df = sym_data[sym]
            if ts not in df.index:
                continue
            row = df.loc[ts]
            price = row["close"]
            high = row["high"]
            low = row["low"]

            # ── Check exits for open position ──
            if sym in positions:
                pos = positions[sym]
                exit_price = None
                exit_reason = None
                side = pos["side"]
                entry = pos["entry_price"]

                # Time held
                hours_held = (ts - pos["entry_time"]).total_seconds() / 3600

                # Update trailing stop
                pos_atr = pos.get("atr", price * 0.02)
                if side == "long":
                    pos["best_price"] = max(pos.get("best_price", entry), high)
                    profit_atr = (pos["best_price"] - entry) / pos_atr if pos_atr > 0 else 0
                    if profit_atr >= COMBO3.get("trail_start_atr", 99):
                        new_trail = pos["best_price"] - COMBO3.get("trail_atr", 3.0) * pos_atr
                        if pos.get("trail_sl") is None or new_trail > pos["trail_sl"]:
                            pos["trail_sl"] = new_trail
                else:
                    pos["best_price"] = min(pos.get("best_price", entry), low)
                    profit_atr = (entry - pos["best_price"]) / pos_atr if pos_atr > 0 else 0
                    if profit_atr >= COMBO3.get("trail_start_atr", 99):
                        new_trail = pos["best_price"] + COMBO3.get("trail_atr", 3.0) * pos_atr
                        if pos.get("trail_sl") is None or new_trail < pos["trail_sl"]:
                            pos["trail_sl"] = new_trail

                # Stop loss (fixed)
                if side == "long" and low <= pos["sl"]:
                    exit_price = max(pos["sl"], low)
                    exit_reason = "sl"
                elif side == "short" and high >= pos["sl"]:
                    exit_price = min(pos["sl"], high)
                    exit_reason = "sl"

                # Trailing stop
                trail = pos.get("trail_sl")
                if exit_price is None and trail is not None:
                    if side == "long" and low <= trail:
                        exit_price = max(trail, low)
                        exit_reason = "trail_sl"
                    elif side == "short" and high >= trail:
                        exit_price = min(trail, high)
                        exit_reason = "trail_sl"

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
                if exit_price is None and hours_held >= COMBO3["max_hold"]:
                    exit_price = price
                    exit_reason = "timeout"

                if exit_price is not None:
                    qty = pos["quantity"]
                    if side == "long":
                        pnl = (exit_price - entry) * qty
                    else:
                        pnl = (entry - exit_price) * qty
                    pnl_pct = pnl / (entry * qty) * 100
                    commission = (entry * qty + exit_price * qty) * (COMMISSION_RATE + SLIPPAGE_RATE)
                    trade = TradeRecord(
                        entry_time=pos["entry_time"], exit_time=ts,
                        side=side, entry_price=entry, exit_price=exit_price,
                        quantity=qty, pnl=pnl, pnl_pct=pnl_pct,
                        commission=commission, exit_reason=exit_reason,
                    )
                    all_trades.append((sym, trade))
                    equity += pnl - commission
                    del positions[sym]

            # ── Check entries ──
            if sym not in positions and row.get("signal", 0) != 0:
                sig = row["signal"]
                side = "long" if sig == 1 else "short"
                sl = row.get("stop_loss", 0)
                atr_val = row.get("atr", price * 0.02)

                if pd.isna(sl) or sl == 0:
                    sl = price * (0.92 if side == "long" else 1.08)

                sl_distance = abs(price - sl)
                if sl_distance < price * 0.001:
                    continue  # SL too tight, skip

                # Risk-based sizing: risk 2% of equity
                risk_amount = equity * RISK_PER_TRADE
                quantity = risk_amount / sl_distance
                notional = quantity * price

                # Per-asset limit check
                asset_limit = ASSET_LIMITS.get(sym, 0.20)
                max_notional_asset = equity * asset_limit
                if notional > max_notional_asset:
                    quantity = max_notional_asset / price
                    notional = quantity * price

                # Total exposure check
                current_exposure = sum(
                    p["quantity"] * sym_data[s].loc[ts]["close"]
                    for s, p in positions.items()
                    if ts in sym_data[s].index
                )
                if (current_exposure + notional) > equity * MAX_TOTAL_EXPOSURE:
                    remaining = equity * MAX_TOTAL_EXPOSURE - current_exposure
                    if remaining <= 0:
                        continue
                    quantity = remaining / price
                    notional = quantity * price

                if equity < 10 or quantity * price < 10:
                    continue

                # Volatility regime adjustment
                vol_pct = (atr_val / price * 100) if not pd.isna(atr_val) and price > 0 else 2.0
                if vol_pct > 5.0:
                    quantity *= 0.50
                elif vol_pct > 3.0:
                    quantity *= 0.75
                elif vol_pct < 1.5:
                    quantity = min(quantity * 1.25, max_notional_asset / price)

                if quantity * price < 10:
                    continue

                positions[sym] = {
                    "side": side,
                    "entry_price": price,
                    "quantity": quantity,
                    "entry_time": ts,
                    "sl": sl,
                    "atr": atr_val if not pd.isna(atr_val) else price * 0.02,
                    "best_price": price,
                    "trail_sl": None,
                }

        # Mark-to-market equity
        mtm = equity
        for sym, pos in positions.items():
            df = sym_data[sym]
            if ts in df.index:
                p = df.loc[ts]["close"]
                if pos["side"] == "long":
                    mtm += (p - pos["entry_price"]) * pos["quantity"]
                else:
                    mtm += (pos["entry_price"] - p) * pos["quantity"]
        equity_curve.append(mtm)
        equity_times.append(ts)

    # Close any remaining positions at end
    last_ts = timeline[-1]
    for sym in list(positions.keys()):
        pos = positions[sym]
        df = sym_data[sym]
        if last_ts in df.index:
            exit_price = df.loc[last_ts]["close"]
        else:
            # find closest
            exit_price = pos["entry_price"]
        qty = pos["quantity"]
        entry = pos["entry_price"]
        side = pos["side"]
        if side == "long":
            pnl = (exit_price - entry) * qty
        else:
            pnl = (entry - exit_price) * qty
        pnl_pct = pnl / (entry * qty) * 100
        commission = (entry * qty + exit_price * qty) * (COMMISSION_RATE + SLIPPAGE_RATE)
        trade = TradeRecord(
            entry_time=pos["entry_time"], exit_time=last_ts,
            side=side, entry_price=entry, exit_price=exit_price,
            quantity=qty, pnl=pnl, pnl_pct=pnl_pct,
            commission=commission, exit_reason="end",
        )
        all_trades.append((sym, trade))
        equity += pnl - commission

    equity_series = pd.Series(equity_curve, index=equity_times)
    return all_trades, equity_series, equity


def make_portfolio_chart(equity, trades_by_sym, out_path):
    """Portfolio equity curve + per-asset breakdown."""
    fig, axes = plt.subplots(2, 1, figsize=(20, 12), gridspec_kw={"height_ratios": [1.2, 1]})
    fig.suptitle("FVG Trend Follow Combo3 — Portfolio (BTC+ETH+SOL+XRP)\n$2,000 Initial | 2% Risk/Trade | 2024-04 to 2026-04",
                 fontsize=15, fontweight="bold")

    # ── Panel 1: Equity Curve ──
    ax1 = axes[0]
    eq_color = "#3b82f6"
    ax1.fill_between(equity.index, equity.values[0], equity.values, alpha=0.15, color=eq_color)
    ax1.plot(equity.index, equity.values, color=eq_color, linewidth=1.5, label="Portfolio Equity")

    # Drawdown
    eq_arr = equity.values
    peak = np.maximum.accumulate(eq_arr)
    dd_pct = (peak - eq_arr) / peak * 100
    ax1_dd = ax1.twinx()
    ax1_dd.fill_between(equity.index, 0, dd_pct, alpha=0.15, color="#ef4444")
    ax1_dd.set_ylabel("Drawdown %", color="#ef4444", fontsize=11)
    ax1_dd.set_ylim(0, max(dd_pct.max(), 1) * 3)
    ax1_dd.invert_yaxis()
    ax1_dd.tick_params(axis="y", colors="#ef4444")

    ax1.set_ylabel("Equity (USD)", fontsize=12)
    ax1.legend(loc="upper left")
    ax1.grid(True, alpha=0.3)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=2))

    # ── Panel 2: Trade P&L by asset ──
    ax2 = axes[1]
    sym_colors = {"BTCUSDT": "#f7931a", "ETHUSDT": "#627eea", "SOLUSDT": "#9945ff", "XRPUSDT": "#23292f"}
    for sym in SYMBOLS:
        trades = trades_by_sym.get(sym, [])
        if not trades:
            continue
        times = [t.exit_time for t in trades]
        pnls = [t.pnl - t.commission for t in trades]
        cumulative = np.cumsum(pnls)
        ax2.plot(times, cumulative, color=sym_colors.get(sym, "#888"),
                 linewidth=1.2, label=f"{sym} ({len(trades)} trades, ${sum(pnls):+,.0f})", marker=".", markersize=3)

    ax2.axhline(y=0, color="black", linewidth=0.8, linestyle="--")
    ax2.set_ylabel("Cumulative P&L (USD)", fontsize=12)
    ax2.set_xlabel("Date", fontsize=12)
    ax2.legend(loc="upper left", fontsize=9)
    ax2.grid(True, alpha=0.3)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=2))

    # Stats text
    final_eq = eq_arr[-1]
    total_ret = (final_eq - INITIAL_CAPITAL) / INITIAL_CAPITAL * 100
    total_trades = sum(len(v) for v in trades_by_sym.values())
    all_pnls = [t.pnl - t.commission for trades in trades_by_sym.values() for t in trades]
    wins = sum(1 for p in all_pnls if p > 0)
    wr = wins / len(all_pnls) * 100 if all_pnls else 0
    max_dd = dd_pct.max()

    stats_text = (
        f"Final: ${final_eq:,.0f}  |  Return: {'+' if total_ret > 0 else ''}{total_ret:.0f}%  |  "
        f"Trades: {total_trades}  |  WR: {wr:.1f}%  |  Max DD: {max_dd:.1f}%"
    )
    fig.text(0.5, 0.02, stats_text, ha="center", fontsize=12, fontweight="bold",
             bbox=dict(boxstyle="round,pad=0.5", facecolor="#f0f9ff", edgecolor="#3b82f6", alpha=0.9))

    plt.tight_layout(rect=[0, 0.04, 1, 0.95])
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Portfolio chart saved: {out_path}")


def make_monthly_chart(equity, out_path):
    """Monthly returns bar chart for the portfolio."""
    daily = equity.resample("D").last().dropna()
    monthly = daily.resample("ME").last().dropna()
    monthly_ret = monthly.pct_change().dropna() * 100

    fig, ax = plt.subplots(figsize=(18, 5))
    colors = ["#22c55e" if r > 0 else "#ef4444" for r in monthly_ret.values]
    labels = [d.strftime("%Y-%m") for d in monthly_ret.index]
    bars = ax.bar(range(len(monthly_ret)), monthly_ret.values, color=colors, edgecolor="white", linewidth=0.5)

    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("Monthly Return %", fontsize=12)
    ax.set_title("FVG Trend Follow Combo3 — Portfolio Monthly Returns", fontsize=14, fontweight="bold")
    ax.axhline(y=0, color="black", linewidth=0.8)
    ax.grid(True, alpha=0.3, axis="y")

    for bar, val in zip(bars, monthly_ret.values):
        if abs(val) > 2:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                    f"{val:+.0f}%", ha="center", va="bottom" if val > 0 else "top",
                    fontsize=8, fontweight="bold")

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Monthly chart saved: {out_path}")


def print_portfolio_summary(trades_by_sym, equity_series):
    """Print per-asset and overall summary."""
    print(f"\n{'='*120}")
    print(f"  PORTFOLIO BACKTEST RESULTS — FVG Trend Follow Combo3")
    print(f"  Initial: ${INITIAL_CAPITAL:,.0f} | Risk: {RISK_PER_TRADE*100:.0f}%/trade | Period: {START_DATE} to {END_DATE}")
    print(f"{'='*120}")

    total_trades_all = 0
    total_pnl_all = 0

    for sym in SYMBOLS:
        trades = trades_by_sym.get(sym, [])
        if not trades:
            print(f"\n  {sym}: No trades")
            continue

        net_pnls = [t.pnl - t.commission for t in trades]
        wins = [p for p in net_pnls if p > 0]
        losses = [p for p in net_pnls if p <= 0]
        total_pnl = sum(net_pnls)
        wr = len(wins) / len(trades) * 100
        avg_win = np.mean([t.pnl_pct for t in trades if t.pnl - t.commission > 0]) if wins else 0
        avg_loss = np.mean([t.pnl_pct for t in trades if t.pnl - t.commission <= 0]) if losses else 0
        pf = sum(wins) / abs(sum(losses)) if losses and sum(losses) != 0 else float("inf")

        # Exit reasons
        reasons = {}
        for t in trades:
            reasons[t.exit_reason] = reasons.get(t.exit_reason, 0) + 1

        longs = [t for t in trades if t.side == "long"]
        shorts = [t for t in trades if t.side == "short"]

        print(f"\n  {sym}:")
        print(f"    Trades: {len(trades)} (L:{len(longs)} S:{len(shorts)}) | WR: {wr:.1f}% | PF: {pf:.2f}")
        print(f"    P&L: ${total_pnl:+,.2f} | Avg Win: +{avg_win:.2f}% | Avg Loss: {avg_loss:.2f}%")
        print(f"    Exits: {reasons}")

        total_trades_all += len(trades)
        total_pnl_all += total_pnl

    # Overall
    eq_arr = equity_series.values
    peak = np.maximum.accumulate(eq_arr)
    dd = (peak - eq_arr) / peak * 100
    max_dd = dd.max()
    final_eq = eq_arr[-1]
    total_ret = (final_eq - INITIAL_CAPITAL) / INITIAL_CAPITAL * 100

    total_days = (equity_series.index[-1] - equity_series.index[0]).days
    total_years = total_days / 365.25 if total_days > 0 else 1
    annual_ret = ((final_eq / INITIAL_CAPITAL) ** (1/total_years) - 1) * 100

    daily_eq = equity_series.resample("D").last().dropna()
    daily_ret = daily_eq.pct_change().dropna()
    sharpe = float(daily_ret.mean() / daily_ret.std() * np.sqrt(365)) if len(daily_ret) > 30 and daily_ret.std() > 0 else 0
    downside = daily_ret[daily_ret < 0]
    sortino = float(daily_ret.mean() / downside.std() * np.sqrt(365)) if len(downside) > 0 and downside.std() > 0 else 0
    calmar = annual_ret / max_dd if max_dd > 0 else 0

    all_net = [t.pnl - t.commission for trades in trades_by_sym.values() for t in trades]
    all_wins = sum(1 for p in all_net if p > 0)
    overall_wr = all_wins / len(all_net) * 100 if all_net else 0

    print(f"\n  {'─'*80}")
    print(f"  PORTFOLIO OVERALL:")
    print(f"    Initial Capital:  ${INITIAL_CAPITAL:,.0f}")
    print(f"    Final Equity:     ${final_eq:,.2f}")
    print(f"    Total Return:     {'+' if total_ret > 0 else ''}{total_ret:.1f}%")
    print(f"    Annual Return:    {'+' if annual_ret > 0 else ''}{annual_ret:.1f}%")
    print(f"    Max Drawdown:     {max_dd:.1f}%")
    print(f"    Sharpe Ratio:     {sharpe:.2f}")
    print(f"    Sortino Ratio:    {sortino:.2f}")
    print(f"    Calmar Ratio:     {calmar:.2f}")
    print(f"    Total Trades:     {total_trades_all}")
    print(f"    Overall WR:       {overall_wr:.1f}%")
    print(f"    Total P&L:        ${total_pnl_all:+,.2f}")
    print(f"    Period:           {total_days} days ({total_years:.1f} years)")


def print_trades_detail(trades_by_sym):
    """Print trade-by-trade details for each asset."""
    for sym in SYMBOLS:
        trades = trades_by_sym.get(sym, [])
        if not trades:
            continue

        print(f"\n{'='*130}")
        print(f"  {sym} — Trade Details ({len(trades)} trades)")
        print(f"{'='*130}")
        print(f"  {'#':>3} | {'Side':>5} | {'Entry Date':>19} | {'Exit Date':>19} | {'Entry$':>10} | {'Exit$':>10} | "
              f"{'P&L$':>10} | {'P&L%':>7} | {'Qty':>12} | {'Exit':>10}")
        print(f"  {'-'*3}-+-{'-'*5}-+-{'-'*19}-+-{'-'*19}-+-{'-'*10}-+-{'-'*10}-+-"
              f"{'-'*10}-+-{'-'*7}-+-{'-'*12}-+-{'-'*10}")

        for i, t in enumerate(trades):
            net = t.pnl - t.commission
            marker = "+" if net > 0 else ""
            print(
                f"  {i+1:>3} | {t.side:>5} | {t.entry_time.strftime('%Y-%m-%d %H:%M'):>19} | "
                f"{t.exit_time.strftime('%Y-%m-%d %H:%M'):>19} | "
                f"${t.entry_price:>9,.2f} | ${t.exit_price:>9,.2f} | "
                f"{marker}${net:>9,.2f} | {marker}{t.pnl_pct:>5.1f}% | "
                f"{t.quantity:>12.6f} | {t.exit_reason:>10}"
            )


async def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    # Load data
    print("Loading data for all symbols...")
    dfs = {}
    for sym in SYMBOLS:
        df_full = await ensure_data(sym, INTERVAL, "2021-04-01", END_DATE)
        # Keep warmup period for indicators
        warmup_start = "2024-01-01"
        df = df_full[df_full.index >= warmup_start].copy()
        dfs[sym] = df
        print(f"  {sym}: {len(df)} bars, {df.index[0]} -> {df.index[-1]}")

    # Generate signals
    print("\nGenerating FVG signals...")
    signal_dfs = generate_all_signals(dfs)

    # Filter signals to display period only for entry (but keep warmup for indicators)
    cutoff = pd.Timestamp(START_DATE)
    for sym in SYMBOLS:
        df = signal_dfs[sym]
        # Zero out signals before START_DATE so no trades open before our window
        mask = df.index < cutoff
        df.loc[mask, "signal"] = 0
        signal_dfs[sym] = df

    # Run portfolio backtest
    print("\nRunning portfolio backtest...")
    all_trades, equity_series, final_equity = run_portfolio_backtest(signal_dfs)

    # Filter equity to display period
    equity_display = equity_series[equity_series.index >= cutoff]

    # Group trades by symbol
    trades_by_sym = {}
    for sym, trade in all_trades:
        if trade.entry_time >= cutoff:
            trades_by_sym.setdefault(sym, []).append(trade)

    # Print results
    print_portfolio_summary(trades_by_sym, equity_display)
    print_trades_detail(trades_by_sym)

    # Generate charts
    print("\nGenerating charts...")
    make_portfolio_chart(equity_display, trades_by_sym, os.path.join(OUT_DIR, "portfolio_equity.png"))
    make_monthly_chart(equity_display, os.path.join(OUT_DIR, "portfolio_monthly.png"))

    print(f"\nDone! Charts saved to {OUT_DIR}/")


if __name__ == "__main__":
    asyncio.run(main())
