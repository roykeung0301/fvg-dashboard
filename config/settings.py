"""全局配置 — 從 .env 載入，含 FVG Trend Follow Combo3 策略參數"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


def _load_dotenv():
    """手動載入 .env 文件（不依賴 python-dotenv）"""
    env_path = Path(__file__).resolve().parent.parent / ".env"
    if not env_path.exists():
        return
    with open(env_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" not in line:
                continue
            key, _, value = line.partition("=")
            key = key.strip()
            value = value.strip()
            # 去掉引號
            if len(value) >= 2 and value[0] == value[-1] and value[0] in ('"', "'"):
                value = value[1:-1]
            os.environ.setdefault(key, value)


# 啟動時載入 .env
_load_dotenv()


@dataclass
class BinanceConfig:
    api_key: str = ""
    api_secret: str = ""
    testnet: bool = False  # 使用正式網（需 VPN）
    base_url: str = ""

    def __post_init__(self):
        # 從環境變數讀取
        if not self.api_key:
            self.api_key = os.getenv("BINANCE_API_KEY", "")
        if not self.api_secret:
            self.api_secret = os.getenv("BINANCE_SECRET_KEY",
                             os.getenv("BINANCE_API_SECRET", ""))
        if not self.base_url:
            self.base_url = (
                "https://testnet.binancefuture.com"
                if self.testnet
                else "https://fapi.binance.com"
            )


@dataclass
class TradingConfig:
    symbols: list[str] = field(default_factory=lambda: ["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT"])
    default_timeframe: str = "1h"
    initial_capital: float = 5000.0
    max_open_positions: int = 5
    trading_enabled: bool = False  # 預設關閉實盤交易
    news_monitoring_enabled: bool = False  # 新聞監控（暫時關閉）

    def __post_init__(self):
        # 允許從環境變數覆寫
        env_symbols = os.getenv("TRADING_SYMBOLS")
        if env_symbols:
            self.symbols = [s.strip() for s in env_symbols.split(",") if s.strip()]
        env_capital = os.getenv("INITIAL_CAPITAL")
        if env_capital:
            try:
                self.initial_capital = float(env_capital)
            except ValueError:
                pass


# ── FVG Trend Follow Combo3 策略參數 ──────────────────

@dataclass
class FVGCombo3Config:
    """Combo3 Optimized — WR 42.2%, Return +192%, PF 2.18"""
    daily_ema_fast: int = 10
    daily_ema_slow: int = 20
    fvg_min_size_pct: float = 0.10
    entry_min_score: int = 2
    sl_atr: float = 10.0
    fvg_max_age: int = 75
    max_active_fvgs: int = 30
    trail_start_atr: float = 99.0   # disabled — trailing stop hurts this strategy's returns
    trail_atr: float = 5.0          # trail distance (inactive when trail_start_atr=99)
    breakeven_atr: float = 99.0     # disabled
    cooldown: int = 12
    max_hold: int = 360
    atr_period: int = 14


# ── 每資產風險配置 ────────────────────────────────

@dataclass
class AssetRiskConfig:
    """單一資產的風險參數"""
    symbol: str = ""
    max_position_pct: float = 0.30  # 該資產最大倉位佔權益 %
    max_drawdown_pct: float = 30.0  # 該資產 MDD 限制
    volatility_class: str = "medium"  # "low" / "medium" / "high"


# 依據回測結果的每資產預設值
DEFAULT_ASSET_RISK_CONFIGS: dict[str, AssetRiskConfig] = {
    "BTCUSDT": AssetRiskConfig(
        symbol="BTCUSDT",
        max_position_pct=0.35,
        max_drawdown_pct=30.0,
        volatility_class="low",
    ),
    "ETHUSDT": AssetRiskConfig(
        symbol="ETHUSDT",
        max_position_pct=0.30,
        max_drawdown_pct=40.0,
        volatility_class="medium",
    ),
    "SOLUSDT": AssetRiskConfig(
        symbol="SOLUSDT",
        max_position_pct=0.20,
        max_drawdown_pct=50.0,
        volatility_class="high",
    ),
    "XRPUSDT": AssetRiskConfig(
        symbol="XRPUSDT",
        max_position_pct=0.20,
        max_drawdown_pct=45.0,
        volatility_class="medium",
    ),
}


@dataclass
class RiskConfig:
    """全局風險參數"""
    risk_per_trade_pct: float = 2.0   # 每筆交易風險 2% 權益
    max_correlated_exposure: float = 0.70  # 加密貨幣高相關，最大總曝險 70%
    max_daily_trades: int = 50
    min_signal_confidence: float = 0.6
    # 波動率倉位調整閾值 (ATR as % of price)
    vol_very_high_pct: float = 5.0    # > 5% → 倉位 -50%
    vol_high_pct: float = 3.0         # > 3% → 倉位 -25%
    vol_low_pct: float = 1.5          # < 1.5% → 倉位 +25%
    asset_configs: dict[str, AssetRiskConfig] = field(
        default_factory=lambda: dict(DEFAULT_ASSET_RISK_CONFIGS)
    )

    def __post_init__(self):
        env_risk = os.getenv("MAX_PORTFOLIO_RISK_PCT")
        if env_risk:
            try:
                self.risk_per_trade_pct = float(env_risk)
            except ValueError:
                pass


@dataclass
class Settings:
    binance: BinanceConfig = field(default_factory=BinanceConfig)
    trading: TradingConfig = field(default_factory=TradingConfig)
    strategy: FVGCombo3Config = field(default_factory=FVGCombo3Config)
    risk: RiskConfig = field(default_factory=RiskConfig)
    log_level: str = "INFO"
    data_dir: str = "data"


# 全局單例
settings = Settings()
