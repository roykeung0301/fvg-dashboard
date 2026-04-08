"""風險管理模型"""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field


class RiskLimits(BaseModel):
    """風險限額參數"""
    max_position_size: float = 0.1       # 單倉位最大比例 (佔總權益)
    max_total_exposure: float = 0.5      # 總曝險最大比例
    max_leverage: float = 3.0            # 最大槓桿
    max_drawdown_pct: float = 10.0       # 最大回撤百分比 → 觸發全部平倉
    stop_loss_pct: float = 2.0           # 單倉止損百分比
    take_profit_pct: float = 5.0         # 單倉止盈百分比
    max_correlated_positions: int = 3    # 高相關性幣種最大同時持倉
    max_daily_trades: int = 50           # 每日最大交易次數
    min_signal_confidence: float = 0.6   # 最低信號信心閾值

    # ── FVG Trend Follow 新增 ─────────────────
    # 每資產限額
    asset_configs: dict = Field(default_factory=lambda: {
        "BTCUSDT": {"max_position_pct": 0.40, "max_dd_pct": 30.0, "vol_class": "low"},
        "ETHUSDT": {"max_position_pct": 0.35, "max_dd_pct": 40.0, "vol_class": "medium"},
        "SOLUSDT": {"max_position_pct": 0.25, "max_dd_pct": 50.0, "vol_class": "high"},
    })
    risk_per_trade_pct: float = 2.0            # 每筆交易風險佔權益 %
    max_correlated_exposure: float = 0.70      # 加密貨幣最大總曝險 (高相關)


class RiskReport(BaseModel):
    """風險評估報告"""
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    portfolio_var: float = 0.0           # Value at Risk (95%)
    portfolio_cvar: float = 0.0          # Conditional VaR
    current_drawdown: float = 0.0        # 當前回撤 %
    max_drawdown: float = 0.0            # 歷史最大回撤 %
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    exposure_ratio: float = 0.0          # 當前曝險比
    risk_level: str = "normal"           # "low" / "normal" / "elevated" / "critical"
    warnings: list[str] = Field(default_factory=list)
    position_adjustments: list[dict] = Field(default_factory=list)  # 建議的倉位調整
    # 新增: 每資產風險摘要
    per_asset_risk: dict = Field(default_factory=dict)
