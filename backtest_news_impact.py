"""
回測新聞監控對策略的影響

方法：用歷史價格數據識別「極端事件」和「重大負面事件」，
模擬新聞系統在這些時段的反應，比較有/無新聞監控的績效差異。

識別邏輯（模擬新聞觸發）：
- Severity 3 (極端): 單日跌幅 > 10% → 暫停所有交易 24h
- Severity 2 (重大): 單日跌幅 > 5% → 平多倉，禁止做多 12h

Usage: python3 backtest_news_impact.py
"""
from __future__ import annotations
import asyncio
import numpy as np
import pandas as pd
from data.historical_fetcher import ensure_data
from strategies.fvg_trend import FVGTrendStrategy

INITIAL_CAPITAL = 2000.0
START_DATE = "2024-04-01"
END_DATE = "2026-04-01"
SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT"]
COMBO3 = dict(
    daily_ema_fast=10, daily_ema_slow=20,
    fvg_min_size_pct=0.10, entry_min_score=2, sl_atr=6.0,
    fvg_max_age=75, max_active_fvgs=30,
    trail_start_atr=99.0, breakeven_atr=99.0,
    cooldown=12, max_hold=720,
)
ASSET_LIMITS = {"BTCUSDT": 0.35, "ETHUSDT": 0.30, "SOLUSDT": 0.20, "XRPUSDT": 0.20}
COMMISSION_RATE = 0.0004
SLIPPAGE_RATE = 0.0001
RISK_PER_TRADE = 0.02
MAX_TOTAL_EXPOSURE = 0.70

# News simulation thresholds (overridable)
EXTREME_DROP_PCT = -10.0
MAJOR_DROP_PCT = -5.0
EXTREME_PAUSE_HOURS = 24
MAJOR_BLOCK_HOURS = 12


def detect_news_events(signal_dfs: dict, extreme_pct=None, major_pct=None) -> dict:
    extreme_thresh = extreme_pct or EXTREME_DROP_PCT
    major_thresh = major_pct or MAJOR_DROP_PCT
    """Scan price data to find simulated news events."""
    events = []

    for sym in SYMBOLS:
        df = signal_dfs[sym]
        # Calculate rolling 24h return (24 bars for 1h data)
        df["ret_24h"] = df["close"].pct_change(24) * 100

        for ts, row in df.iterrows():
            ret = row.get("ret_24h", 0)
            if pd.isna(ret):
                continue

            if ret <= extreme_thresh:
                events.append({
                    "time": ts, "symbol": sym, "severity": 3,
                    "drop_pct": round(ret, 1),
                    "pause_until": ts + pd.Timedelta(hours=EXTREME_PAUSE_HOURS),
                })
            elif ret <= major_thresh:
                events.append({
                    "time": ts, "symbol": sym, "severity": 2,
                    "drop_pct": round(ret, 1),
                    "block_longs_until": ts + pd.Timedelta(hours=MAJOR_BLOCK_HOURS),
                })

    # Deduplicate: keep one event per 24h window per severity
    events.sort(key=lambda e: e["time"])
    filtered = []
    last_extreme = None
    last_major = {}
    for ev in events:
        if ev["severity"] == 3:
            if last_extreme is None or ev["time"] > last_extreme + pd.Timedelta(hours=24):
                filtered.append(ev)
                last_extreme = ev["time"]
        elif ev["severity"] == 2:
            sym = ev["symbol"]
            if sym not in last_major or ev["time"] > last_major[sym] + pd.Timedelta(hours=12):
                filtered.append(ev)
                last_major[sym] = ev["time"]

    return filtered


def run_backtest(signal_dfs, news_events=None, mode="baseline"):
    """Run portfolio backtest with optional news monitoring."""
    all_indices = set()
    for df in signal_dfs.values():
        all_indices.update(df.index)
    timeline = sorted(all_indices)

    equity = INITIAL_CAPITAL
    positions = {}
    all_trades = []
    equity_curve = []

    # News state
    paused_until = None           # severity 3: no trading until this time
    block_longs_until = {}        # severity 2: {symbol: block until time}

    for ts in timeline:
        # ── Check news events ──
        if news_events:
            for ev in news_events:
                if ev["time"] == ts:
                    if ev["severity"] == 3:
                        paused_until = ev["pause_until"]
                        # Close ALL positions immediately
                        for sym in list(positions.keys()):
                            pos = positions[sym]
                            if ts in signal_dfs[sym].index:
                                exit_price = signal_dfs[sym].loc[ts]["close"]
                            else:
                                exit_price = pos["entry_price"]
                            qty = pos["quantity"]; entry = pos["entry_price"]; side = pos["side"]
                            pnl = (exit_price - entry) * qty if side == "long" else (entry - exit_price) * qty
                            comm = (entry * qty + exit_price * qty) * (COMMISSION_RATE + SLIPPAGE_RATE)
                            all_trades.append({
                                "symbol": sym, "side": side,
                                "entry_time": pos["entry_time"].isoformat(),
                                "exit_time": ts.isoformat(),
                                "entry_price": round(entry, 2), "exit_price": round(exit_price, 2),
                                "quantity": round(qty, 6), "pnl": round(pnl, 2),
                                "net_pnl": round(pnl - comm, 2), "exit_reason": "news_extreme",
                            })
                            equity += pnl - comm
                            del positions[sym]

                    elif ev["severity"] == 2:
                        sym = ev["symbol"]
                        block_longs_until[sym] = ev["block_longs_until"]
                        # Close long position for this symbol
                        if sym in positions and positions[sym]["side"] == "long":
                            pos = positions[sym]
                            if ts in signal_dfs[sym].index:
                                exit_price = signal_dfs[sym].loc[ts]["close"]
                            else:
                                exit_price = pos["entry_price"]
                            qty = pos["quantity"]; entry = pos["entry_price"]
                            pnl = (exit_price - entry) * qty
                            comm = (entry * qty + exit_price * qty) * (COMMISSION_RATE + SLIPPAGE_RATE)
                            all_trades.append({
                                "symbol": sym, "side": "long",
                                "entry_time": pos["entry_time"].isoformat(),
                                "exit_time": ts.isoformat(),
                                "entry_price": round(entry, 2), "exit_price": round(exit_price, 2),
                                "quantity": round(qty, 6), "pnl": round(pnl, 2),
                                "net_pnl": round(pnl - comm, 2), "exit_reason": "news_major",
                            })
                            equity += pnl - comm
                            del positions[sym]

        # Check if trading is paused
        is_paused = paused_until and ts < paused_until

        for sym in SYMBOLS:
            df = signal_dfs[sym]
            if ts not in df.index:
                continue
            row = df.loc[ts]
            price = row["close"]
            high = row["high"]
            low = row["low"]

            # ── Check existing positions (exits still work during pause) ──
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
                    comm = (entry * qty + exit_price * qty) * (COMMISSION_RATE + SLIPPAGE_RATE)
                    all_trades.append({
                        "symbol": sym, "side": side,
                        "entry_time": pos["entry_time"].isoformat(),
                        "exit_time": ts.isoformat(),
                        "entry_price": round(entry, 2), "exit_price": round(exit_price, 2),
                        "quantity": round(qty, 6), "pnl": round(pnl, 2),
                        "net_pnl": round(pnl - comm, 2), "exit_reason": exit_reason,
                    })
                    equity += pnl - comm
                    del positions[sym]

            # ── New entries (blocked during pause) ──
            if is_paused:
                continue

            # Check long block for this symbol
            sym_blocked = sym in block_longs_until and ts < block_longs_until[sym]

            if sym not in positions and row.get("signal", 0) != 0:
                sig = row["signal"]
                side = "long" if sig == 1 else "short"

                # Block longs during major negative
                if sym_blocked and side == "long":
                    continue

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
                    if remaining <= 0:
                        continue
                    quantity = remaining / price
                if equity < 10 or quantity * price < 10:
                    continue
                positions[sym] = {
                    "side": side, "entry_price": price,
                    "quantity": quantity, "entry_time": ts, "sl": sl,
                }

        # MTM equity
        mtm = equity
        for sym, pos in positions.items():
            if ts in signal_dfs[sym].index:
                p = signal_dfs[sym].loc[ts]["close"]
                if pos["side"] == "long":
                    mtm += (p - pos["entry_price"]) * pos["quantity"]
                else:
                    mtm += (pos["entry_price"] - p) * pos["quantity"]
        equity_curve.append({"time": ts, "value": round(mtm, 2)})

    # Close remaining positions
    last_ts = timeline[-1]
    for sym in list(positions.keys()):
        pos = positions[sym]
        exit_price = signal_dfs[sym].loc[last_ts]["close"] if last_ts in signal_dfs[sym].index else pos["entry_price"]
        qty = pos["quantity"]; entry = pos["entry_price"]; side = pos["side"]
        pnl = (exit_price - entry) * qty if side == "long" else (entry - exit_price) * qty
        comm = (entry * qty + exit_price * qty) * (COMMISSION_RATE + SLIPPAGE_RATE)
        all_trades.append({
            "symbol": sym, "side": side,
            "entry_time": pos["entry_time"].isoformat(), "exit_time": last_ts.isoformat(),
            "entry_price": round(entry, 2), "exit_price": round(exit_price, 2),
            "quantity": round(qty, 6), "pnl": round(pnl, 2),
            "net_pnl": round(pnl - comm, 2), "exit_reason": "end",
        })
        equity += pnl - comm

    return {
        "mode": mode,
        "trades": all_trades,
        "equity_curve": equity_curve,
        "final_equity": round(equity, 2),
    }


def calc_stats(result: dict) -> dict:
    """Calculate performance statistics."""
    trades = result["trades"]
    eq_curve = result["equity_curve"]

    n_trades = len(trades)
    wins = [t for t in trades if t["net_pnl"] > 0]
    losses = [t for t in trades if t["net_pnl"] <= 0]
    wr = len(wins) / n_trades * 100 if n_trades else 0
    total_pnl = sum(t["net_pnl"] for t in trades)
    ret_pct = total_pnl / INITIAL_CAPITAL * 100

    # Max drawdown
    peak = INITIAL_CAPITAL
    max_dd = 0
    for pt in eq_curve:
        v = pt["value"]
        if v > peak:
            peak = v
        dd = (peak - v) / peak * 100
        if dd > max_dd:
            max_dd = dd

    # Profit factor
    gross_profit = sum(t["net_pnl"] for t in wins)
    gross_loss = abs(sum(t["net_pnl"] for t in losses))
    pf = gross_profit / gross_loss if gross_loss > 0 else float("inf")

    # Avg win/loss
    avg_win = np.mean([t["net_pnl"] for t in wins]) if wins else 0
    avg_loss = np.mean([t["net_pnl"] for t in losses]) if losses else 0

    # News-specific exits
    news_exits = [t for t in trades if t["exit_reason"] in ("news_extreme", "news_major")]

    return {
        "trades": n_trades, "wins": len(wins), "losses": len(losses),
        "wr": round(wr, 1), "total_pnl": round(total_pnl, 2),
        "return_pct": round(ret_pct, 1), "final_equity": result["final_equity"],
        "max_drawdown": round(max_dd, 1), "profit_factor": round(pf, 2),
        "avg_win": round(avg_win, 2), "avg_loss": round(avg_loss, 2),
        "news_exits": len(news_exits),
    }


async def main():
    print("=" * 60)
    print("  新聞監控回測 — 多閾值比較")
    print("=" * 60)
    print(f"  期間: {START_DATE} → {END_DATE}")
    print(f"  資金: ${INITIAL_CAPITAL:,.0f}")
    print()

    # Load data
    print("Loading data...")
    dfs = {}
    for sym in SYMBOLS:
        df_full = await ensure_data(sym, "1h", "2021-04-01", END_DATE)
        dfs[sym] = df_full[df_full.index >= "2024-01-01"].copy()
        print(f"  {sym}: {len(dfs[sym])} bars")

    # Generate signals
    print("\nGenerating signals...")
    strategy = FVGTrendStrategy(**COMBO3)
    signal_dfs = {}
    cutoff = pd.Timestamp(START_DATE)
    for sym in SYMBOLS:
        df = strategy.generate_signals(dfs[sym].copy())
        df.loc[df.index < cutoff, "signal"] = 0
        signal_dfs[sym] = df

    # Baseline
    print("\nRunning baseline (no news)...")
    baseline = run_backtest(signal_dfs, news_events=None, mode="baseline")
    stats_base = calc_stats(baseline)

    # Test multiple thresholds
    scenarios = [
        {"name": "寬鬆 (15%/8%)", "extreme": -15.0, "major": -8.0},
        {"name": "中等 (10%/5%)", "extreme": -10.0, "major": -5.0},
    ]

    all_stats = [("無新聞監控", stats_base)]

    for sc in scenarios:
        print(f"\nScanning events: {sc['name']}...")
        events = detect_news_events(signal_dfs, extreme_pct=sc["extreme"], major_pct=sc["major"])
        sev3 = [e for e in events if e["severity"] == 3]
        sev2 = [e for e in events if e["severity"] == 2]
        print(f"  🔴 極端: {len(sev3)} 次 | 🟠 重大: {len(sev2)} 次")

        if sev3:
            print("  極端事件:")
            for e in sev3[:5]:
                print(f"    {e['time'].strftime('%Y-%m-%d %H:%M')} | {e['symbol']} | {e['drop_pct']}%")

        result = run_backtest(signal_dfs, news_events=events, mode=sc["name"])
        stats = calc_stats(result)
        all_stats.append((sc["name"], stats))

    # ── Summary table ──
    print("\n" + "=" * 78)
    header = f"{'指標':<18}"
    for name, _ in all_stats:
        header += f" {name:>18}"
    print(header)
    print("-" * 78)

    rows = [
        ("Final Equity", lambda s: f"${s['final_equity']:,.2f}"),
        ("Return %", lambda s: f"{s['return_pct']:+.1f}%"),
        ("Trades", lambda s: str(s['trades'])),
        ("Win Rate", lambda s: f"{s['wr']}%"),
        ("Profit Factor", lambda s: str(s['profit_factor'])),
        ("Max Drawdown", lambda s: f"{s['max_drawdown']}%"),
        ("Avg Win", lambda s: f"${s['avg_win']:,.2f}"),
        ("Avg Loss", lambda s: f"${s['avg_loss']:,.2f}"),
        ("News Exits", lambda s: str(s['news_exits'])),
    ]

    for label, fmt in rows:
        line = f"  {label:<16}"
        for _, stats in all_stats:
            line += f" {fmt(stats):>18}"
        print(line)

    print("=" * 78)

    # Verdict per scenario
    for name, stats in all_stats[1:]:
        better_ret = stats["return_pct"] > stats_base["return_pct"]
        less_dd = stats["max_drawdown"] < stats_base["max_drawdown"]
        dd_diff = stats_base["max_drawdown"] - stats["max_drawdown"]
        ret_diff = stats["return_pct"] - stats_base["return_pct"]

        if better_ret and less_dd:
            print(f"\n✅ {name}: 回報 {ret_diff:+.1f}% + 回撤 {-dd_diff:+.1f}% → 強烈建議")
        elif less_dd:
            print(f"\n✅ {name}: 回撤減少 {dd_diff:.1f}%，回報損失 {abs(ret_diff):.1f}% → 風控更安全")
        elif better_ret:
            print(f"\n⚠️ {name}: 回報 {ret_diff:+.1f}%，回撤增加 {abs(dd_diff):.1f}%")
        else:
            print(f"\n⚠️ {name}: 額外風險保護，但回報和回撤都略差")


if __name__ == "__main__":
    asyncio.run(main())
