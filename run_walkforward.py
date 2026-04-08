"""
Walk-Forward Validation — 驗證策略是否真正有 edge

方法：
- 把 2 年數據分成多個窗口
- 每次用前 6 個月「訓練」（選最佳參數）
- 接下來 3 個月「測試」（用訓練選的參數，不能改）
- 重複滾動，拼出一條「從未見過的數據」的績效曲線

如果 walk-forward 結果也是正的，策略才真正有 edge。
如果只有 backtest 賺錢但 walk-forward 虧錢 → 過擬合。

Usage: python3 run_walkforward.py
"""
from __future__ import annotations
import asyncio
import numpy as np
import pandas as pd
from datetime import timedelta
from data.historical_fetcher import ensure_data
from strategies.fvg_trend import FVGTrendStrategy
from config.settings import settings

INITIAL_CAPITAL = 5000.0
SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT"]
ASSET_LIMITS = {"BTCUSDT": 0.35, "ETHUSDT": 0.30, "SOLUSDT": 0.20, "XRPUSDT": 0.20}
COMMISSION_RATE = 0.0004
SLIPPAGE_RATE = 0.0001
RISK_PER_TRADE = 0.02
MAX_TOTAL_EXPOSURE = 0.70

# Walk-forward windows
TRAIN_MONTHS = 6
TEST_MONTHS = 3
DATA_START = "2022-07-01"    # 需要更早的數據
DATA_END = "2026-04-01"
WF_START = "2023-04-01"      # walk-forward 起點（前面是第一個訓練窗口）

# Parameter grid to search in each training window (27 combos)
PARAM_GRID = [
    {"entry_min_score": s, "fvg_min_size_pct": f, "sl_atr": sl, "fvg_max_age": 75}
    for s in [1, 2, 3]
    for f in [0.05, 0.10, 0.20]
    for sl in [4.0, 6.0, 8.0]
]

BASE_PARAMS = dict(
    daily_ema_fast=10, daily_ema_slow=20,
    max_active_fvgs=30, trail_start_atr=99.0,
    breakeven_atr=99.0, cooldown=12, max_hold=360,
)


def run_single_backtest(signal_dfs: dict, start: str, end: str,
                        initial_equity: float = INITIAL_CAPITAL) -> dict:
    """Run backtest on a specific date range."""
    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)

    all_indices = set()
    for df in signal_dfs.values():
        mask = (df.index >= start_ts) & (df.index < end_ts)
        all_indices.update(df.index[mask])
    timeline = sorted(all_indices)

    if not timeline:
        return {"trades": [], "final_equity": initial_equity, "return_pct": 0}

    equity = initial_equity
    positions = {}
    trades = []

    for ts in timeline:
        for sym in SYMBOLS:
            df = signal_dfs[sym]
            if ts not in df.index:
                continue
            row = df.loc[ts]
            price = row["close"]
            high = row["high"]
            low = row["low"]

            # Check exits
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

                if exit_price is None and hours_held >= BASE_PARAMS["max_hold"]:
                    exit_price = price; exit_reason = "timeout"

                if exit_price is not None:
                    qty = pos["quantity"]; entry = pos["entry_price"]
                    pnl = (exit_price - entry) * qty if side == "long" else (entry - exit_price) * qty
                    comm = (entry * qty + exit_price * qty) * (COMMISSION_RATE + SLIPPAGE_RATE)
                    net = pnl - comm
                    trades.append({"symbol": sym, "pnl": round(net, 2), "side": side})
                    equity += net
                    del positions[sym]

            # Check entries
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
                current_exp = sum(
                    p["quantity"] * signal_dfs[s].loc[ts]["close"]
                    for s, p in positions.items()
                    if ts in signal_dfs[s].index
                )
                if (current_exp + quantity * price) > equity * MAX_TOTAL_EXPOSURE:
                    continue
                if equity < 10 or quantity * price < 10:
                    continue
                positions[sym] = {
                    "side": side, "entry_price": price,
                    "quantity": quantity, "entry_time": ts, "sl": sl,
                }

    # Close remaining
    if timeline:
        last_ts = timeline[-1]
        for sym in list(positions.keys()):
            pos = positions[sym]
            exit_price = signal_dfs[sym].loc[last_ts]["close"] if last_ts in signal_dfs[sym].index else pos["entry_price"]
            qty = pos["quantity"]; entry = pos["entry_price"]; side = pos["side"]
            pnl = (exit_price - entry) * qty if side == "long" else (entry - exit_price) * qty
            comm = (entry * qty + exit_price * qty) * (COMMISSION_RATE + SLIPPAGE_RATE)
            equity += pnl - comm

    ret = (equity - initial_equity) / initial_equity * 100
    return {"trades": trades, "final_equity": round(equity, 2), "return_pct": round(ret, 1)}


def generate_signals(dfs: dict, params: dict) -> dict:
    """Generate signals with given parameters."""
    merged = {**BASE_PARAMS, **params}
    strategy = FVGTrendStrategy(**merged)
    signal_dfs = {}
    for sym in SYMBOLS:
        signal_dfs[sym] = strategy.generate_signals(dfs[sym].copy())
    return signal_dfs


async def main():
    print("=" * 70)
    print("  WALK-FORWARD VALIDATION")
    print("  訓練 6 個月 → 測試 3 個月 → 滾動前進")
    print("=" * 70)

    # Load all data
    print("\nLoading data...")
    dfs = {}
    for sym in SYMBOLS:
        df_full = await ensure_data(sym, "1h", DATA_START, DATA_END)
        dfs[sym] = df_full.copy()
        print(f"  {sym}: {len(dfs[sym])} bars ({dfs[sym].index[0]} → {dfs[sym].index[-1]})")

    # Generate walk-forward windows
    wf_start = pd.Timestamp(WF_START)
    data_end = pd.Timestamp(DATA_END)
    windows = []
    current = wf_start
    while current + pd.DateOffset(months=TEST_MONTHS) <= data_end:
        train_start = current - pd.DateOffset(months=TRAIN_MONTHS)
        train_end = current
        test_start = current
        test_end = current + pd.DateOffset(months=TEST_MONTHS)
        windows.append({
            "train_start": train_start.strftime("%Y-%m-%d"),
            "train_end": train_end.strftime("%Y-%m-%d"),
            "test_start": test_start.strftime("%Y-%m-%d"),
            "test_end": test_end.strftime("%Y-%m-%d"),
        })
        current += pd.DateOffset(months=TEST_MONTHS)

    print(f"\n{len(windows)} walk-forward windows:")
    for i, w in enumerate(windows):
        print(f"  [{i+1}] Train: {w['train_start']} → {w['train_end']} | Test: {w['test_start']} → {w['test_end']}")

    # Also test with FIXED params (Combo3) for comparison
    combo3_params = {
        "entry_min_score": 2, "fvg_min_size_pct": 0.10,
        "sl_atr": 10.0, "fvg_max_age": 75,
    }

    # Run walk-forward
    print(f"\nSearching {len(PARAM_GRID)} parameter combinations per window...")
    print()

    wf_equity = INITIAL_CAPITAL
    wf_trades_total = 0
    wf_wins = 0
    fixed_equity = INITIAL_CAPITAL
    fixed_trades_total = 0
    fixed_wins = 0

    wf_results = []
    fixed_results = []

    for i, w in enumerate(windows):
        # ── Step 1: Find best params on training data ──
        best_return = -999
        best_params = combo3_params  # fallback
        best_pf = 0

        for params in PARAM_GRID:
            sig_dfs = generate_signals(dfs, params)
            result = run_single_backtest(sig_dfs, w["train_start"], w["train_end"])
            trades = result["trades"]
            if len(trades) < 10:
                continue

            wins = sum(1 for t in trades if t["pnl"] > 0)
            gross_w = sum(t["pnl"] for t in trades if t["pnl"] > 0)
            gross_l = abs(sum(t["pnl"] for t in trades if t["pnl"] <= 0))
            pf = gross_w / gross_l if gross_l > 0 else 0
            ret = result["return_pct"]

            # Select by profit factor (more stable than return)
            if pf > best_pf and ret > 0:
                best_pf = pf
                best_return = ret
                best_params = params

        # ── Step 2: Test best params on UNSEEN test data ──
        sig_dfs_wf = generate_signals(dfs, best_params)
        wf_result = run_single_backtest(sig_dfs_wf, w["test_start"], w["test_end"], wf_equity)
        wf_trades = wf_result["trades"]
        wf_equity = wf_result["final_equity"]
        wf_trades_total += len(wf_trades)
        wf_wins += sum(1 for t in wf_trades if t["pnl"] > 0)

        # ── Step 3: Test FIXED Combo3 params on same test data ──
        sig_dfs_fixed = generate_signals(dfs, combo3_params)
        fixed_result = run_single_backtest(sig_dfs_fixed, w["test_start"], w["test_end"], fixed_equity)
        fixed_trades = fixed_result["trades"]
        fixed_equity = fixed_result["final_equity"]
        fixed_trades_total += len(fixed_trades)
        fixed_wins += sum(1 for t in fixed_trades if t["pnl"] > 0)

        wf_ret = (wf_result["final_equity"] - INITIAL_CAPITAL) / INITIAL_CAPITAL * 100
        fx_ret = (fixed_equity - INITIAL_CAPITAL) / INITIAL_CAPITAL * 100

        print(
            f"  Window {i+1}: {w['test_start']} → {w['test_end']}\n"
            f"    Best train params: score={best_params['entry_min_score']} "
            f"fvg={best_params['fvg_min_size_pct']} sl={best_params['sl_atr']} "
            f"age={best_params['fvg_max_age']}\n"
            f"    WF test:    {len(wf_trades):3d} trades | "
            f"ret {wf_result['return_pct']:+6.1f}% (cumul {wf_ret:+.1f}%)\n"
            f"    Fixed test:  {len(fixed_trades):3d} trades | "
            f"ret {fixed_result['return_pct']:+6.1f}% (cumul {fx_ret:+.1f}%)"
        )
        print()

        wf_results.append({"window": i+1, "ret": wf_result["return_pct"], "trades": len(wf_trades)})
        fixed_results.append({"window": i+1, "ret": fixed_result["return_pct"], "trades": len(fixed_trades)})

    # ── Summary ──
    print("=" * 70)
    print("  WALK-FORWARD SUMMARY")
    print("=" * 70)

    wf_total_ret = (wf_equity - INITIAL_CAPITAL) / INITIAL_CAPITAL * 100
    fx_total_ret = (fixed_equity - INITIAL_CAPITAL) / INITIAL_CAPITAL * 100
    wf_wr = wf_wins / wf_trades_total * 100 if wf_trades_total > 0 else 0
    fx_wr = fixed_wins / fixed_trades_total * 100 if fixed_trades_total > 0 else 0

    wf_positive = sum(1 for r in wf_results if r["ret"] > 0)
    fx_positive = sum(1 for r in fixed_results if r["ret"] > 0)

    print(f"\n{'指標':<25} {'Walk-Forward':>15} {'Fixed Combo3':>15}")
    print("-" * 55)
    print(f"  {'Final Equity':<23} ${wf_equity:>13,.2f} ${fixed_equity:>13,.2f}")
    print(f"  {'Total Return':<23} {wf_total_ret:>13.1f}% {fx_total_ret:>13.1f}%")
    print(f"  {'Total Trades':<23} {wf_trades_total:>13d} {fixed_trades_total:>13d}")
    print(f"  {'Win Rate':<23} {wf_wr:>12.1f}% {fx_wr:>12.1f}%")
    print(f"  {'Positive Windows':<23} {wf_positive:>10d}/{len(windows)} {fx_positive:>10d}/{len(windows)}")
    print()

    # Verdict
    if wf_total_ret > 0 and fx_total_ret > 0:
        print("✅ 兩種方法都在未見過的數據上盈利 — 策略有一定的 edge")
        if wf_total_ret > fx_total_ret:
            print("   Walk-forward 動態選參更好 — 建議採用")
        else:
            print("   Fixed Combo3 更穩定 — 參數不敏感是好事，建議保留")
    elif fx_total_ret > 0 and wf_total_ret <= 0:
        print("⚠️ Fixed Combo3 盈利但 walk-forward 虧損 — 策略可能過擬合特定參數")
    elif wf_total_ret > 0 and fx_total_ret <= 0:
        print("⚠️ Walk-forward 盈利但 fixed 虧損 — 需要動態調參才有效")
    else:
        print("🚨 兩種方法都虧損 — 策略在未見數據上沒有 edge，不建議實盤")

    print()
    print("注意: Walk-forward 是策略驗證的金標準。")
    print("只有在 walk-forward 盈利的情況下才應考慮實盤交易。")


if __name__ == "__main__":
    asyncio.run(main())
