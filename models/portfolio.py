"""投資組合模型"""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field


class Position(BaseModel):
    """單個持倉"""
    symbol: str
    side: str  # "long" / "short"
    entry_price: float
    current_price: float = 0.0
    quantity: float
    leverage: float = 1.0
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0
    opened_at: datetime = Field(default_factory=datetime.utcnow)

    @property
    def notional_value(self) -> float:
        return self.current_price * self.quantity

    @property
    def pnl_pct(self) -> float:
        if self.entry_price == 0:
            return 0.0
        direction = 1.0 if self.side == "long" else -1.0
        return direction * (self.current_price - self.entry_price) / self.entry_price * 100


class PortfolioState(BaseModel):
    """投資組合狀態"""
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    total_equity: float = 0.0
    available_balance: float = 0.0
    positions: list[Position] = Field(default_factory=list)
    total_unrealized_pnl: float = 0.0
    total_realized_pnl: float = 0.0
    margin_used: float = 0.0

    @property
    def position_count(self) -> int:
        return len(self.positions)

    @property
    def margin_ratio(self) -> float:
        if self.total_equity == 0:
            return 0.0
        return self.margin_used / self.total_equity
