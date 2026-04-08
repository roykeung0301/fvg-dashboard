"""Agent 間通訊的消息模型"""

from __future__ import annotations

import uuid
from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class MessageType(str, Enum):
    """消息類型"""
    MARKET_UPDATE = "market_update"          # 市場分析師 → 全員
    STRATEGY_PROPOSAL = "strategy_proposal"  # 量化研究員 → 回測工程師
    BACKTEST_RESULT = "backtest_result"      # 回測工程師 → 量化研究員/風控
    RISK_ASSESSMENT = "risk_assessment"      # 風控 → 信號/執行
    RISK_ALERT = "risk_alert"               # 風控 → 全員
    TRADE_SIGNAL = "trade_signal"           # 信號工程師 → 執行工程師
    EXECUTION_REPORT = "execution_report"   # 執行工程師 → 全員
    ANOMALY_ALERT = "anomaly_alert"         # 市場分析師 → 風控
    COMMAND = "command"                     # 調度器 → 任意角色
    STATUS_REPORT = "status_report"         # 任意角色 → 調度器


class AgentMessage(BaseModel):
    """Agent 之間傳遞的標準消息"""
    id: str = Field(default_factory=lambda: uuid.uuid4().hex[:12])
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    sender: str
    receiver: str  # agent id 或 "broadcast"
    msg_type: MessageType
    payload: dict[str, Any] = Field(default_factory=dict)
    priority: int = Field(default=0, ge=0, le=10)  # 0=普通, 10=最緊急

    class Config:
        json_encoders = {datetime: lambda v: v.isoformat()}
