"""
Live Trade Logger — 記錄 paper/live 交易數據到 JSON
供 dashboard 讀取顯示實時持倉和 P&L
"""
from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone, timedelta
from pathlib import Path
from threading import Lock

logger = logging.getLogger("trade_logger")

HKT = timezone(timedelta(hours=8))
DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "live_portfolio.json"


class TradeLogger:
    """Write live trading state to JSON for dashboard consumption."""

    def __init__(self, initial_capital: float = 5000.0):
        self.initial_capital = initial_capital
        self.equity = initial_capital
        self.positions: dict = {}       # symbol -> position info
        self.trades: list = []          # completed trades
        self.equity_snapshots: list = []  # periodic equity points
        self._lock = Lock()

        # Load existing data if resuming
        self._load()

    def _load(self):
        """Load existing live data if available."""
        if DATA_PATH.exists():
            try:
                with open(DATA_PATH) as f:
                    data = json.load(f)
                self.equity = data.get("equity", self.initial_capital)
                self.trades = data.get("trades", [])
                self.positions = data.get("positions", {})
                self.equity_snapshots = data.get("equity_curve", [])
                self.initial_capital = data.get("initial_capital", self.initial_capital)
                logger.info(f"Loaded {len(self.trades)} existing trades, equity ${self.equity:,.2f}")
            except Exception as e:
                logger.error(f"Failed to load live data: {e}")

    def _save(self):
        """Save current state to JSON (atomic write to prevent corruption)."""
        import tempfile
        with self._lock:
            now = datetime.now(HKT)
            data = {
                "initial_capital": self.initial_capital,
                "equity": round(self.equity, 2),
                "start_date": self.trades[0]["entry_time"][:10] if self.trades else now.strftime("%Y-%m-%d"),
                "end_date": now.strftime("%Y-%m-%d"),
                "last_updated": now.isoformat(),
                "mode": "live",
                "symbols": ["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT"],
                "strategy": "FVG Trend Follow Combo3 Optimized",
                "trades": self.trades,
                "positions": self.positions,
                "equity_curve": self.equity_snapshots,
                "final_equity": round(self.equity, 2),
            }
            os.makedirs(DATA_PATH.parent, exist_ok=True)
            # Atomic write: write to temp file then rename
            try:
                with tempfile.NamedTemporaryFile(
                    mode='w', dir=DATA_PATH.parent, delete=False, suffix='.tmp'
                ) as tmp:
                    json.dump(data, tmp, indent=2)
                    tmp_path = tmp.name
                os.replace(tmp_path, DATA_PATH)
            except Exception as e:
                logger.error(f"Failed to save: {e}")
                # Fallback: direct write
                with open(DATA_PATH, "w") as f:
                    json.dump(data, f, indent=2)

    def log_entry(self, symbol: str, side: str, price: float,
                  quantity: float, sl: float):
        """Record a new position opening."""
        now = datetime.now(HKT).isoformat()
        self.positions[symbol] = {
            "side": side,
            "entry_price": round(price, 2),
            "quantity": round(quantity, 6),
            "sl": round(sl, 2),
            "entry_time": now,
            "notional": round(price * quantity, 2),
        }
        self._save()
        logger.info(f"Logged entry: {symbol} {side} @ ${price:,.2f}")

    def log_exit(self, symbol: str, side: str, entry_price: float,
                 exit_price: float, quantity: float, pnl: float,
                 pnl_pct: float, exit_reason: str):
        """Record a position close."""
        now = datetime.now(HKT).isoformat()
        pos = self.positions.pop(symbol, {})

        trade = {
            "symbol": symbol,
            "side": side,
            "entry_time": pos.get("entry_time", now),
            "exit_time": now,
            "entry_price": round(entry_price, 2),
            "exit_price": round(exit_price, 2),
            "quantity": round(quantity, 6),
            "pnl": round(pnl, 2),
            "pnl_pct": round(pnl_pct, 2),
            "commission": round((entry_price + exit_price) * quantity * 0.0004, 2),  # 0.04% per side
            "net_pnl": round(pnl - (entry_price + exit_price) * quantity * 0.0004, 2),
            "exit_reason": exit_reason,
        }
        self.trades.append(trade)
        self.equity += trade["net_pnl"]

        # Add equity snapshot
        self.equity_snapshots.append({
            "time": now[:10],
            "value": round(self.equity, 2),
        })

        self._save()
        logger.info(f"Logged exit: {symbol} {side} PnL ${pnl:+,.2f}")

    def snapshot_equity(self):
        """Record current equity (called periodically)."""
        now = datetime.now(HKT)
        today = now.strftime("%Y-%m-%d")

        # Update today's snapshot (don't duplicate)
        if self.equity_snapshots and self.equity_snapshots[-1]["time"] == today:
            self.equity_snapshots[-1]["value"] = round(self.equity, 2)
        else:
            self.equity_snapshots.append({
                "time": today,
                "value": round(self.equity, 2),
            })

        self._save()

    def get_data(self) -> dict:
        """Get current state for dashboard."""
        now = datetime.now(HKT)
        return {
            "initial_capital": self.initial_capital,
            "equity": round(self.equity, 2),
            "start_date": self.trades[0]["entry_time"][:10] if self.trades else now.strftime("%Y-%m-%d"),
            "end_date": now.strftime("%Y-%m-%d"),
            "last_updated": now.isoformat(),
            "mode": "live",
            "symbols": ["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT"],
            "strategy": "FVG Trend Follow Combo3 Optimized",
            "trades": self.trades,
            "positions": self.positions,
            "equity_curve": self.equity_snapshots,
            "final_equity": round(self.equity, 2),
        }


# Global singleton
trade_logger = TradeLogger()
