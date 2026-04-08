"""策略與回測模型"""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field


class StrategyConfig(BaseModel):
    """策略配置"""
    name: str
    version: str = "1.0"
    symbols: list[str] = Field(default_factory=list)
    timeframe: str = "1h"
    indicators: list[dict] = Field(default_factory=list)  # 使用的指標及參數
    entry_rules: list[str] = Field(default_factory=list)  # 進場條件描述
    exit_rules: list[str] = Field(default_factory=list)   # 出場條件描述
    parameters: dict = Field(default_factory=dict)         # 策略參數


class BacktestResult(BaseModel):
    """回測結果"""
    strategy_name: str
    start_date: datetime
    end_date: datetime
    initial_capital: float = 10000.0
    final_capital: float = 0.0
    total_return_pct: float = 0.0
    annualized_return_pct: float = 0.0
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    max_drawdown_pct: float = 0.0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    total_trades: int = 0
    avg_trade_duration: str = ""
    best_trade_pct: float = 0.0
    worst_trade_pct: float = 0.0
    equity_curve: list[float] = Field(default_factory=list)
    monthly_returns: list[dict] = Field(default_factory=list)
    passed: bool = False  # 是否通過品質門檻
    notes: list[str] = Field(default_factory=list)
