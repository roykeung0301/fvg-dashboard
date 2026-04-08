"""市場數據模型"""

from __future__ import annotations

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field


class OHLCV(BaseModel):
    """K 線數據"""
    symbol: str
    interval: str  # "1m", "5m", "1h", "1d" ...
    open_time: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
    close_time: datetime
    quote_volume: float = 0.0
    trades: int = 0


class OrderBookLevel(BaseModel):
    price: float
    quantity: float


class OrderBookSnapshot(BaseModel):
    """訂單簿快照"""
    symbol: str
    timestamp: datetime
    bids: list[OrderBookLevel] = Field(default_factory=list)
    asks: list[OrderBookLevel] = Field(default_factory=list)

    @property
    def best_bid(self) -> Optional[float]:
        return self.bids[0].price if self.bids else None

    @property
    def best_ask(self) -> Optional[float]:
        return self.asks[0].price if self.asks else None

    @property
    def spread(self) -> Optional[float]:
        if self.best_bid and self.best_ask:
            return self.best_ask - self.best_bid
        return None


class MarketSnapshot(BaseModel):
    """市場狀態快照"""
    symbol: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    price: float
    volume_24h: float = 0.0
    price_change_24h: float = 0.0  # 百分比
    high_24h: float = 0.0
    low_24h: float = 0.0
    funding_rate: Optional[float] = None  # 合約資金費率
    open_interest: Optional[float] = None  # 未平倉合約
    market_sentiment: Optional[str] = None  # "bullish" / "bearish" / "neutral"
    anomalies: list[str] = Field(default_factory=list)
