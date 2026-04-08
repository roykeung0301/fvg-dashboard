"""量化研究員 — 發現 alpha，設計策略"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

from agents.base_agent import BaseAgent
from models.messages import AgentMessage, MessageType
from models.strategy import StrategyConfig


class QuantResearcher(BaseAgent):
    """
    量化研究員：
    - 從市場數據中發現 alpha 因子
    - 設計交易策略並提交回測
    - 根據回測結果迭代優化
    """

    def __init__(self):
        super().__init__(agent_id="quant_researcher", name="量化研究員")
        self.strategies: dict[str, StrategyConfig] = {}
        self.alpha_factors: list[dict] = []

    async def on_start(self):
        self.state["active_strategy"] = None
        self.state["iteration"] = 0

    async def handle_message(self, message: AgentMessage):
        handlers = {
            MessageType.MARKET_UPDATE: self._on_market_update,
            MessageType.BACKTEST_RESULT: self._on_backtest_result,
            MessageType.COMMAND: self._on_command,
        }
        handler = handlers.get(message.msg_type)
        if handler:
            await handler(message)

    # ── 消息處理 ─────────────────────────────

    async def _on_market_update(self, msg: AgentMessage):
        """根據市場狀態評估是否需要調整策略"""
        market = msg.payload
        sentiment = market.get("market_sentiment", "neutral")
        anomalies = market.get("anomalies", [])

        if anomalies:
            self.logger.info(f"偵測到市場異常: {anomalies}，評估策略調整")

    async def _on_backtest_result(self, msg: AgentMessage):
        """根據回測結果決定下一步"""
        result = msg.payload
        strategy_name = result.get("strategy_name", "")

        if result.get("passed", False):
            self.logger.info(f"策略 [{strategy_name}] 通過回測，提交信號工程師")
            await self.send(
                "signal_engineer",
                MessageType.COMMAND,
                {"action": "deploy_strategy", "strategy": result},
            )
        else:
            self.logger.info(f"策略 [{strategy_name}] 未通過回測，進行迭代")
            self.state["iteration"] += 1

    async def _on_command(self, msg: AgentMessage):
        action = msg.payload.get("action")
        if action == "design_strategy":
            await self.design_strategy(msg.payload)

    # ── 核心邏輯 ─────────────────────────────

    async def design_strategy(self, params: dict) -> StrategyConfig:
        """設計一個新的交易策略"""
        symbols = params.get("symbols", ["BTCUSDT"])
        timeframe = params.get("timeframe", "1h")

        strategy = StrategyConfig(
            name=f"alpha_v{self.state.get('iteration', 0) + 1}",
            symbols=symbols,
            timeframe=timeframe,
            indicators=[
                {"name": "EMA", "params": {"fast": 12, "slow": 26}},
                {"name": "RSI", "params": {"period": 14}},
                {"name": "MACD", "params": {"fast": 12, "slow": 26, "signal": 9}},
                {"name": "BB", "params": {"period": 20, "std": 2}},
            ],
            entry_rules=[
                "EMA_fast > EMA_slow (趨勢向上)",
                "RSI < 70 (未超買)",
                "MACD histogram > 0 (動能正向)",
                "Price > BB_middle (高於均值)",
            ],
            exit_rules=[
                "EMA_fast < EMA_slow (趨勢反轉)",
                "RSI > 80 (超買)",
                "止損: -2%",
                "止盈: +5%",
            ],
            parameters={
                "ema_fast": 12,
                "ema_slow": 26,
                "rsi_period": 14,
                "rsi_overbought": 70,
                "rsi_oversold": 30,
                "bb_period": 20,
                "bb_std": 2,
            },
        )

        self.strategies[strategy.name] = strategy
        self.state["active_strategy"] = strategy.name

        # 提交給回測工程師
        await self.send(
            "backtest_engineer",
            MessageType.STRATEGY_PROPOSAL,
            strategy.model_dump(),
        )

        self.logger.info(f"策略 [{strategy.name}] 已設計並提交回測")
        return strategy

    def compute_alpha_factor(self, df: pd.DataFrame, factor_name: str) -> pd.Series:
        """計算 alpha 因子"""
        factors = {
            "momentum": lambda d: d["close"].pct_change(20),
            "mean_reversion": lambda d: -(d["close"] - d["close"].rolling(20).mean())
            / d["close"].rolling(20).std(),
            "volume_spike": lambda d: d["volume"] / d["volume"].rolling(20).mean() - 1,
            "volatility": lambda d: d["close"].pct_change().rolling(20).std(),
        }

        if factor_name not in factors:
            raise ValueError(f"未知因子: {factor_name}")

        result = factors[factor_name](df)
        self.alpha_factors.append({"name": factor_name, "ic": self._calc_ic(result, df)})
        return result

    @staticmethod
    def _calc_ic(factor: pd.Series, df: pd.DataFrame) -> float:
        """計算信息係數 (IC)"""
        forward_returns = df["close"].pct_change().shift(-1)
        valid = factor.notna() & forward_returns.notna()
        if valid.sum() < 30:
            return 0.0
        return float(factor[valid].corr(forward_returns[valid]))
