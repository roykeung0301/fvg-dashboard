"""交易信號模型"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class SignalType(str, Enum):
    BUY = "buy"
    SELL = "sell"
    HOLD = "hold"
    CLOSE = "close"  # 平倉


class SignalStrength(str, Enum):
    WEAK = "weak"
    MODERATE = "moderate"
    STRONG = "strong"


class Signal(BaseModel):
    """標準化交易信號"""
    symbol: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    signal_type: SignalType
    strength: SignalStrength = SignalStrength.MODERATE
    confidence: float = Field(ge=0.0, le=1.0, default=0.5)
    suggested_size: float = Field(ge=0.0, default=0.0)  # 建議倉位比例 (0~1)
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    timeframe: str = "1h"
    source_indicators: list[str] = Field(default_factory=list)  # 觸發的指標
    metadata: dict = Field(default_factory=dict)
