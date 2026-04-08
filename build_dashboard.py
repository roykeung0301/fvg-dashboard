"""
Build a self-contained dashboard HTML with embedded data.
Output: docs/index.html (ready for GitHub Pages / Netlify / any hosting)

Usage:
  python3 build_dashboard.py          # Build once
  python3 build_dashboard.py --serve  # Build + open in browser
"""
from __future__ import annotations
import asyncio, json, os, sys, webbrowser
import numpy as np, pandas as pd
from data.historical_fetcher import ensure_data
from strategies.fvg_trend import FVGTrendStrategy

INITIAL_CAPITAL = 5000.0
START_DATE = "2024-04-01"
END_DATE   = "2026-04-01"
SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT"]
COMBO3 = dict(
    daily_ema_fast=10, daily_ema_slow=20,
    fvg_min_size_pct=0.10, entry_min_score=2, sl_atr=10.0,
    fvg_max_age=75, max_active_fvgs=30,
    trail_start_atr=99.0, breakeven_atr=99.0,
    cooldown=12, max_hold=360,
)
ASSET_LIMITS = {"BTCUSDT": 0.35, "ETHUSDT": 0.30, "SOLUSDT": 0.20, "XRPUSDT": 0.20}
COMMISSION_RATE = 0.0004
SLIPPAGE_RATE = 0.0001
RISK_PER_TRADE = 0.02
MAX_TOTAL_EXPOSURE = 0.70
OUT_DIR = "docs"


def run_portfolio_backtest(signal_dfs):
    all_indices = set()
    for df in signal_dfs.values():
        all_indices.update(df.index)
    timeline = sorted(all_indices)

    equity = INITIAL_CAPITAL
    positions = {}
    all_trades = []
    equity_curve = []

    for ts in timeline:
        for sym in SYMBOLS:
            df = signal_dfs[sym]
            if ts not in df.index:
                continue
            row = df.loc[ts]
            price = row["close"]
            high = row["high"]
            low = row["low"]

            if sym in positions:
                pos = positions[sym]
                exit_price = None
                exit_reason = None
                side = pos["side"]
                hours_held = (ts - pos["entry_time"]).total_seconds() / 3600

                if side == "long" and low <= pos["sl"]:
                    exit_price = max(pos["sl"], low)
                    exit_reason = "sl"
                elif side == "short" and high >= pos["sl"]:
                    exit_price = min(pos["sl"], high)
                    exit_reason = "sl"

                if exit_price is None:
                    daily = row.get("daily_trend", 0)
                    if not pd.isna(daily):
                        if side == "long" and daily != 1:
                            exit_price = price; exit_reason = "trend_rev"
                        elif side == "short" and daily != -1:
                            exit_price = price; exit_reason = "trend_rev"

                if exit_price is None and hours_held >= COMBO3["max_hold"]:
                    exit_price = price; exit_reason = "timeout"

                if exit_price is not None:
                    qty = pos["quantity"]; entry = pos["entry_price"]
                    pnl = (exit_price - entry) * qty if side == "long" else (entry - exit_price) * qty
                    pnl_pct = pnl / (entry * qty) * 100
                    comm = (entry * qty + exit_price * qty) * (COMMISSION_RATE + SLIPPAGE_RATE)
                    all_trades.append({
                        "symbol": sym, "side": side,
                        "entry_time": pos["entry_time"].isoformat(),
                        "exit_time": ts.isoformat(),
                        "entry_price": round(entry, 2), "exit_price": round(exit_price, 2),
                        "quantity": round(qty, 6), "pnl": round(pnl, 2),
                        "pnl_pct": round(pnl_pct, 2), "commission": round(comm, 2),
                        "net_pnl": round(pnl - comm, 2), "exit_reason": exit_reason,
                    })
                    equity += pnl - comm
                    del positions[sym]

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
                    if remaining <= 0: continue
                    quantity = remaining / price
                if equity < 10 or quantity * price < 10:
                    continue
                positions[sym] = {
                    "side": side, "entry_price": price,
                    "quantity": quantity, "entry_time": ts, "sl": sl,
                }

        mtm = equity
        for sym, pos in positions.items():
            if ts in signal_dfs[sym].index:
                p = signal_dfs[sym].loc[ts]["close"]
                if pos["side"] == "long": mtm += (p - pos["entry_price"]) * pos["quantity"]
                else: mtm += (pos["entry_price"] - p) * pos["quantity"]
        equity_curve.append({"time": ts.isoformat(), "value": round(mtm, 2)})

    last_ts = timeline[-1]
    for sym in list(positions.keys()):
        pos = positions[sym]
        exit_price = signal_dfs[sym].loc[last_ts]["close"] if last_ts in signal_dfs[sym].index else pos["entry_price"]
        qty = pos["quantity"]; entry = pos["entry_price"]; side = pos["side"]
        pnl = (exit_price - entry) * qty if side == "long" else (entry - exit_price) * qty
        pnl_pct = pnl / (entry * qty) * 100
        comm = (entry * qty + exit_price * qty) * (COMMISSION_RATE + SLIPPAGE_RATE)
        all_trades.append({
            "symbol": sym, "side": side,
            "entry_time": pos["entry_time"].isoformat(), "exit_time": last_ts.isoformat(),
            "entry_price": round(entry, 2), "exit_price": round(exit_price, 2),
            "quantity": round(qty, 6), "pnl": round(pnl, 2), "pnl_pct": round(pnl_pct, 2),
            "commission": round(comm, 2), "net_pnl": round(pnl - comm, 2), "exit_reason": "end",
        })
        equity += pnl - comm

    eq_df = pd.DataFrame(equity_curve)
    eq_df["time"] = pd.to_datetime(eq_df["time"])
    eq_daily = eq_df.set_index("time").resample("D").last().dropna().reset_index()
    eq_daily_list = [{"time": r["time"].isoformat()[:10], "value": r["value"]} for _, r in eq_daily.iterrows()]

    return {
        "initial_capital": INITIAL_CAPITAL, "start_date": START_DATE, "end_date": END_DATE,
        "symbols": SYMBOLS, "strategy": "FVG Trend Follow Combo3 Optimized",
        "params": COMBO3, "trades": all_trades, "equity_curve": eq_daily_list,
        "final_equity": round(equity, 2),
    }


async def generate_data():
    print("Loading data...")
    dfs = {}
    for sym in SYMBOLS:
        df_full = await ensure_data(sym, "1h", "2021-04-01", END_DATE)
        dfs[sym] = df_full[df_full.index >= "2024-01-01"].copy()
        print(f"  {sym}: {len(dfs[sym])} bars")

    print("Generating signals...")
    strategy = FVGTrendStrategy(**COMBO3)
    signal_dfs = {}
    cutoff = pd.Timestamp(START_DATE)
    for sym in SYMBOLS:
        df = strategy.generate_signals(dfs[sym].copy())
        df.loc[df.index < cutoff, "signal"] = 0
        signal_dfs[sym] = df

    print("Running portfolio backtest...")
    return run_portfolio_backtest(signal_dfs)


def build_html(data):
    """Read dashboard.html and embed data into it."""
    template_path = os.path.join(os.path.dirname(__file__), "dashboard.html")
    with open(template_path, "r") as f:
        html = f.read()

    # Replace the loadData function to use embedded data
    data_json = json.dumps(data)
    embedded_loader = f"""async function loadData() {{
  DATA = {data_json};
  calendarMonth = new Date(DATA.end_date);
  calendarMonth.setDate(1);
  initPeriods();
  render();
}}"""

    # Find and replace the loadData function
    import re
    html = re.sub(
        r'async function loadData\(\)\s*\{.*?\n\}',
        embedded_loader,
        html,
        flags=re.DOTALL,
    )

    return html


def load_live_data() -> dict | None:
    """Load live/paper trading data if available."""
    live_path = os.path.join(os.path.dirname(__file__), "data", "live_portfolio.json")
    if os.path.exists(live_path):
        with open(live_path) as f:
            data = json.load(f)
        return data
    return None


def main():
    # Always run backtest for historical performance
    backtest_data = asyncio.run(generate_data())
    print(f"Backtest: {len(backtest_data['trades'])} trades, ${backtest_data['final_equity']:,.2f}")

    # Load live/paper trading data
    live_data = load_live_data()
    if live_data and "--backtest-only" not in sys.argv:
        live_trades = live_data.get("trades", [])
        live_positions = live_data.get("positions", {})
        live_equity = live_data.get("equity", INITIAL_CAPITAL)
        live_mode = live_data.get("mode", "paper")
        live_updated = live_data.get("last_updated", "")
        live_curve = live_data.get("equity_curve", [])

        # Merge: backtest as history, live as current
        backtest_data["live"] = {
            "mode": live_mode,
            "equity": live_equity,
            "initial_capital": live_data.get("initial_capital", INITIAL_CAPITAL),
            "positions": live_positions,
            "trades": live_trades,
            "equity_curve": live_curve,
            "last_updated": live_updated,
            "strategy": live_data.get("strategy", ""),
        }
        print(f"Live ({live_mode}): {len(live_trades)} trades, "
              f"{len(live_positions)} positions, ${live_equity:,.2f} equity")
    else:
        backtest_data["live"] = None

    data = backtest_data
    print(f"Final backtest equity: ${data['final_equity']:,.2f}")

    os.makedirs(OUT_DIR, exist_ok=True)
    html = build_html(data)

    out_path = os.path.join(OUT_DIR, "index.html")
    with open(out_path, "w") as f:
        f.write(html)
    print(f"Built: {out_path} ({len(html) / 1024:.0f} KB)")

    # Also save data JSON for external use
    data_path = os.path.join(os.path.dirname(__file__), "dashboard_data.json")
    with open(data_path, "w") as f:
        json.dump(data, f)

    if "--serve" in sys.argv:
        webbrowser.open(f"file://{os.path.abspath(out_path)}")


if __name__ == "__main__":
    main()
