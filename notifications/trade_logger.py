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
_DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DATA_PATH = _DATA_DIR / "live_portfolio.json"  # default, overridden by set_mode()


def set_mode(mode: str):
    """Set trading mode to use separate data files.
    Call BEFORE any trade_logger operations.
      mode='paper' → data/paper_portfolio.json
      mode='live'  → data/live_portfolio.json
    """
    global DATA_PATH
    if mode == "live":
        DATA_PATH = _DATA_DIR / "live_portfolio.json"
    else:
        DATA_PATH = _DATA_DIR / "paper_portfolio.json"
    logger.info(f"TradeLogger mode={mode}, path={DATA_PATH}")


class TradeLogger:
    """Write live trading state to JSON for dashboard consumption."""

    # MEXC taker fees (same as ExecutionEngineer)
    MEXC_TAKER_FEE = {
        "BTCUSDT": 0.0001, "ETHUSDT": 0.0001, "SOLUSDT": 0.0, "XRPUSDT": 0.0,
    }

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
                self.initial_capital = data.get("initial_capital", self.initial_capital)
                loaded_equity = data.get("equity", self.initial_capital)
                # Validate equity: must be positive and reasonable
                if loaded_equity <= 0:
                    logger.warning(f"Invalid equity in JSON: ${loaded_equity}, using initial_capital")
                    loaded_equity = self.initial_capital
                self.equity = loaded_equity
                self.trades = data.get("trades", [])
                self.positions = data.get("positions", {})
                self.equity_snapshots = data.get("equity_curve", [])
                logger.info(f"Loaded {len(self.trades)} existing trades, equity ${self.equity:,.2f}")
            except Exception as e:
                logger.error(f"Failed to load live data: {e}")

    def _calc_commission(self, symbol: str, entry_price: float, exit_price: float, quantity: float) -> float:
        """Calculate round-trip commission based on exchange."""
        from config.settings import settings
        if settings.exchange.is_mexc:
            fee_rate = self.MEXC_TAKER_FEE.get(symbol, 0.0001)
            return (entry_price + exit_price) * quantity * fee_rate
        return (entry_price + exit_price) * quantity * 0.0004  # Binance 0.04%

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
                "symbols": list(self.positions.keys()) if self.positions else ["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT"],
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
                logger.error(f"Atomic save failed: {e}")
                # Clean up failed temp file
                try:
                    os.unlink(tmp_path)
                except Exception:
                    pass
                # Retry atomic write once (do NOT fall back to non-atomic write)
                try:
                    with tempfile.NamedTemporaryFile(
                        mode='w', dir=DATA_PATH.parent, delete=False, suffix='.tmp'
                    ) as tmp2:
                        json.dump(data, tmp2, indent=2)
                        tmp2_path = tmp2.name
                    os.replace(tmp2_path, DATA_PATH)
                except Exception as e2:
                    logger.critical(f"Save retry also failed: {e2} — data NOT persisted!")
                    try:
                        os.unlink(tmp2_path)
                    except Exception:
                        pass

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

    def update_position(self, symbol: str, updates: dict):
        """Update a position's trailing stop / best_price without full re-log.
        Only saves if there are actual changes to persist."""
        if symbol not in self.positions:
            return
        changed = False
        for key in ("best_price", "trail_sl", "sl"):
            if key in updates and updates[key] != self.positions[symbol].get(key):
                self.positions[symbol][key] = updates[key]
                changed = True
        if changed:
            self._save()

    def force_close_position(self, symbol: str):
        """Force-remove a corrupted position (qty=0 etc.) without logging a trade."""
        if symbol in self.positions:
            self.positions.pop(symbol)
            self._save()
            logger.warning(f"Force-closed corrupted position: {symbol}")

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
            "commission": round(self._calc_commission(symbol, entry_price, exit_price, quantity), 2),
            "net_pnl": round(pnl - self._calc_commission(symbol, entry_price, exit_price, quantity), 2),
            "exit_reason": exit_reason,
        }
        self.trades.append(trade)
        # Cap trades list to prevent unbounded growth (keep last 2000)
        if len(self.trades) > 2000:
            self.trades = self.trades[-2000:]
        # NOTE: do NOT self.equity += net_pnl here.
        # self.equity is synced from exchange total (which already contains
        # the realized PnL as the position's unrealized became 0). Adding net_pnl
        # would double-count. The periodic equity sync in orchestrator refreshes
        # self.equity from the exchange every 60s.

        # Add equity snapshot (cap at 1000 entries)
        self.equity_snapshots.append({
            "time": now[:10],
            "value": round(self.equity, 2),
        })
        if len(self.equity_snapshots) > 1000:
            self.equity_snapshots = self.equity_snapshots[-1000:]

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

    def get_unrealized_pnl(self, current_prices: dict) -> float:
        """Calculate total unrealized PnL across all open positions."""
        total_upnl = 0.0
        for sym, pos in self.positions.items():
            cur_price = current_prices.get(sym, 0)
            if cur_price <= 0:
                continue
            qty = pos.get("quantity", 0)
            entry = pos.get("entry_price", 0)
            if pos.get("side") == "long":
                total_upnl += (cur_price - entry) * qty
            else:
                total_upnl += (entry - cur_price) * qty
        return total_upnl

    def get_mark_to_market_equity(self, current_prices: dict) -> float:
        """Mark-to-market equity.
        self.equity is synced from the exchange total (which already includes
        unrealized PnL of open positions), so this returns self.equity directly.
        Only used as a fallback when the exchange sync is stale — a best-effort
        estimate rather than a primary source of truth.
        """
        return self.equity

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


    def reload(self):
        """Re-load data from current DATA_PATH (call after set_mode)."""
        self.positions = {}
        self.trades = []
        self.equity_snapshots = []
        self._load()

# Global singleton
trade_logger = TradeLogger()
