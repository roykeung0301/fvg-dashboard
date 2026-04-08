"""訂單模型"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class OrderSide(str, Enum):
    BUY = "BUY"
    SELL = "SELL"


class OrderType(str, Enum):
    MARKET = "MARKET"
    LIMIT = "LIMIT"
    STOP_MARKET = "STOP_MARKET"
    STOP_LIMIT = "STOP_LIMIT"
    TAKE_PROFIT_MARKET = "TAKE_PROFIT_MARKET"


class OrderStatus(str, Enum):
    PENDING = "pending"
    SUBMITTED = "submitted"
    PARTIALLY_FILLED = "partially_filled"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"
    EXPIRED = "expired"


class Order(BaseModel):
    """交易訂單"""
    symbol: str
    side: OrderSide
    order_type: OrderType = OrderType.MARKET
    quantity: float
    price: Optional[float] = None          # Limit 單價格
    stop_price: Optional[float] = None     # 止損/止盈觸發價
    time_in_force: str = "GTC"          # GTC / IOC / FOK
    reduce_only: bool = False
    metadata: dict = Field(default_factory=dict)


class ExecutionReport(BaseModel):
    """訂單執行回報"""
    order_id: str
    symbol: str
    side: OrderSide
    order_type: OrderType
    status: OrderStatus
    quantity: float
    filled_quantity: float = 0.0
    avg_fill_price: float = 0.0
    commission: float = 0.0
    slippage_bps: float = 0.0  # 滑點 (基點)
    submitted_at: datetime = Field(default_factory=datetime.utcnow)
    filled_at: Optional[datetime] = None
    error_message: Optional[str] = None
