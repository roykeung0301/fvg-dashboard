"""
Dynamic Parameter Optimizer — 每季度自動重新優化策略參數

方法 (Walk-Forward 的實盤版):
- 每 3 個月用最近 6 個月數據重新搜索最佳參數
- 選 Profit Factor 最高的參數組合 (比單純 return 更穩定)
- 最少 10 筆交易才有統計意義
- 如果找不到更好的參數，保留現有參數

原理: Walk-forward 驗證顯示動態選參 (+228.8%) 優於固定參數 (+207.5%)
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime
from pathlib import Path

import pandas as pd

from strategies.fvg_trend import FVGTrendStrategy

logger = logging.getLogger(__name__)

# Parameter search grid (same as walk-forward validation, 27 combos)
PARAM_GRID = [
    {"entry_min_score": s, "fvg_min_size_pct": f, "sl_atr": sl}
    for s in [1, 2, 3]
    for f in [0.05, 0.10, 0.20]
    for sl in [4.0, 6.0, 8.0]
]

BASE_PARAMS = dict(
    daily_ema_fast=10, daily_ema_slow=20,
    fvg_max_age=75, max_active_fvgs=30,
    trail_start_atr=99.0, trail_atr=5.0,
    breakeven_atr=99.0, cooldown=12, max_hold=360,
)

# Backtest constants
SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT"]
ASSET_LIMITS = {"BTCUSDT": 0.35, "ETHUSDT": 0.30, "SOLUSDT": 0.20, "XRPUSDT": 0.20}
COMMISSION_RATE = 0.0004
SLIPPAGE_RATE = 0.0001
RISK_PER_TRADE = 0.02
MAX_TOTAL_EXPOSURE = 0.70
INITIAL_CAPITAL = 5000.0

# State file
STATE_FILE = Path(__file__).resolve().parent.parent / "data" / "optimizer_state.json"


def _backtest_params(dfs: dict, params: dict, start: str, end: str) -> dict:
    """Run a quick backtest with given params on a date range."""
    merged = {**BASE_PARAMS, **params}
    strategy = FVGTrendStrategy(**merged)
    signal_dfs = {}
    for sym in SYMBOLS:
        if sym in dfs:
            signal_dfs[sym] = strategy.generate_signals(dfs[sym].copy())

    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)

    all_indices = set()
    for df in signal_dfs.values():
        mask = (df.index >= start_ts) & (df.index < end_ts)
        all_indices.update(df.index[mask])
    timeline = sorted(all_indices)

    if not timeline:
        return {"trades": 0, "return_pct": 0, "profit_factor": 0}

    equity = INITIAL_CAPITAL
    positions = {}
    wins_pnl = 0.0
    losses_pnl = 0.0
    n_trades = 0

    for ts in timeline:
        for sym in SYMBOLS:
            if sym not in signal_dfs:
                continue
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
                side = pos["side"]
                hours = (ts - pos["entry_time"]).total_seconds() / 3600

                if side == "long" and low <= pos["sl"]:
                    exit_price = max(pos["sl"], low)
                elif side == "short" and high >= pos["sl"]:
                    exit_price = min(pos["sl"], high)

                if exit_price is None:
                    daily = row.get("daily_trend", 0)
                    if not pd.isna(daily):
                        if side == "long" and daily != 1:
                            exit_price = price
                        elif side == "short" and daily != -1:
                            exit_price = price

                if exit_price is None and hours >= merged.get("max_hold", 720):
                    exit_price = price

                if exit_price is not None:
                    qty = pos["quantity"]
                    entry = pos["entry_price"]
                    pnl = (exit_price - entry) * qty if side == "long" else (entry - exit_price) * qty
                    comm = (entry * qty + exit_price * qty) * (COMMISSION_RATE + SLIPPAGE_RATE)
                    net = pnl - comm
                    if net > 0:
                        wins_pnl += net
                    else:
                        losses_pnl += abs(net)
                    n_trades += 1
                    equity += net
                    del positions[sym]

            # Check entries
            if sym not in positions and row.get("signal", 0) != 0:
                sig = row["signal"]
                side = "long" if sig == 1 else "short"
                sl = row.get("stop_loss", 0)
                if pd.isna(sl) or sl == 0:
                    sl = price * (0.92 if side == "long" else 1.08)
                sl_dist = abs(price - sl)
                if sl_dist < price * 0.001:
                    continue
                risk_amt = equity * RISK_PER_TRADE
                qty = risk_amt / sl_dist
                notional = qty * price
                limit = ASSET_LIMITS.get(sym, 0.20)
                if notional > equity * limit:
                    qty = (equity * limit) / price
                curr_exp = sum(
                    p["quantity"] * signal_dfs[s].loc[ts]["close"]
                    for s, p in positions.items()
                    if ts in signal_dfs[s].index
                )
                if (curr_exp + qty * price) > equity * MAX_TOTAL_EXPOSURE:
                    continue
                if equity < 10 or qty * price < 10:
                    continue
                positions[sym] = {
                    "side": side, "entry_price": price,
                    "quantity": qty, "entry_time": ts, "sl": sl,
                }

    ret = (equity - INITIAL_CAPITAL) / INITIAL_CAPITAL * 100
    pf = wins_pnl / losses_pnl if losses_pnl > 0 else 0
    return {"trades": n_trades, "return_pct": round(ret, 1), "profit_factor": round(pf, 2)}


def optimize_params(dfs: dict, train_start: str, train_end: str) -> dict:
    """
    Search PARAM_GRID for best parameters on training data.
    Returns best params dict or None if no good params found.
    """
    best_pf = 0
    best_ret = -999
    best_params = None

    for params in PARAM_GRID:
        result = _backtest_params(dfs, params, train_start, train_end)
        if result["trades"] < 10:
            continue
        if result["profit_factor"] > best_pf and result["return_pct"] > 0:
            best_pf = result["profit_factor"]
            best_ret = result["return_pct"]
            best_params = params

    if best_params:
        logger.info(
            f"Optimizer found best params: {best_params} "
            f"(PF={best_pf}, ret={best_ret}%, trades={result['trades']})"
        )
    else:
        logger.warning("Optimizer found no profitable params, keeping current")

    return best_params


def load_optimizer_state() -> dict:
    """Load last optimization state."""
    if STATE_FILE.exists():
        with open(STATE_FILE) as f:
            return json.load(f)
    return {}


def save_optimizer_state(state: dict):
    """Save optimization state atomically."""
    import tempfile
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=STATE_FILE.parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(state, f, indent=2)
        os.replace(tmp, STATE_FILE)
    except Exception:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise


def should_reoptimize() -> bool:
    """Check if it's time to re-optimize (every 3 months)."""
    state = load_optimizer_state()
    last_opt = state.get("last_optimization")
    if not last_opt:
        return True
    last_dt = datetime.fromisoformat(last_opt)
    days_since = (datetime.now() - last_dt).days
    return days_since >= 90


def get_current_params() -> dict:
    """Get the currently active optimized params, or default Combo3."""
    state = load_optimizer_state()
    params = state.get("current_params")
    if params:
        return params
    # Default Combo3
    return {"entry_min_score": 2, "fvg_min_size_pct": 0.10, "sl_atr": 10.0}
