"""回測工程師 — 歷史回測，績效分析"""

from __future__ import annotations

from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd

from agents.base_agent import BaseAgent
from models.messages import AgentMessage, MessageType
from models.strategy import StrategyConfig, BacktestResult


class BacktestEngineer(BaseAgent):
    """
    回測工程師：
    - 接收策略定義，執行歷史回測
    - 計算績效指標 (Sharpe, Sortino, MDD, Win Rate...)
    - 檢測過擬合風險
    - 回傳回測報告
    """

    def __init__(self):
        super().__init__(agent_id="backtest_engineer", name="回測工程師")
        self.results_history: list[BacktestResult] = []
        self.data_manager = None  # 由 orchestrator 注入

    async def handle_message(self, message: AgentMessage):
        handlers = {
            MessageType.STRATEGY_PROPOSAL: self._on_strategy_proposal,
            MessageType.COMMAND: self._on_command,
        }
        handler = handlers.get(message.msg_type)
        if handler:
            await handler(message)

    async def _on_strategy_proposal(self, msg: AgentMessage):
        """接收策略，執行回測"""
        strategy = StrategyConfig(**msg.payload)
        self.logger.info(f"收到策略 [{strategy.name}]，開始回測")

        # 嘗試從 DataManager 獲取真實數據
        data = None
        if self.data_manager and strategy.symbols:
            symbol = strategy.symbols[0]
            data = self.data_manager.get_historical_klines(
                symbol, strategy.timeframe, n_bars=5000
            )
            if data is not None:
                self.logger.info(f"使用 TradingView 真實數據: {len(data)} 根 K 線")

        result = await self.run_backtest(strategy, data=data)

        # 回傳結果給量化研究員 & 風控
        payload = result.model_dump(mode="json")
        await self.send("quant_researcher", MessageType.BACKTEST_RESULT, payload)
        await self.send("risk_manager", MessageType.BACKTEST_RESULT, payload)

    async def _on_command(self, msg: AgentMessage):
        action = msg.payload.get("action")
        if action == "run_backtest":
            strategy = StrategyConfig(**msg.payload.get("strategy", {}))
            await self.run_backtest(strategy)

    # ── 核心回測引擎 ─────────────────────────

    async def run_backtest(
        self,
        strategy: StrategyConfig,
        data: Optional[pd.DataFrame] = None,
        initial_capital: float = 10000.0,
    ) -> BacktestResult:
        """執行向量化回測"""
        if data is None:
            data = self._generate_sample_data(strategy.symbols[0] if strategy.symbols else "BTCUSDT")

        signals = self._generate_signals(data, strategy)
        equity_curve, trades = self._simulate(data, signals, initial_capital)

        result = self._compute_metrics(strategy.name, data, equity_curve, trades, initial_capital)
        result = self._quality_check(result)
        self.results_history.append(result)
        self.logger.info(
            f"[{strategy.name}] 回測完成 — "
            f"收益: {result.total_return_pct:.2f}%, "
            f"Sharpe: {result.sharpe_ratio:.2f}, "
            f"MDD: {result.max_drawdown_pct:.2f}%, "
            f"勝率: {result.win_rate:.1f}%"
        )
        return result

    def _generate_signals(self, data: pd.DataFrame, strategy: StrategyConfig) -> pd.Series:
        """根據策略規則生成信號 (1=買, -1=賣, 0=持有)"""
        params = strategy.parameters
        signals = pd.Series(0, index=data.index)

        ema_fast = data["close"].ewm(span=params.get("ema_fast", 12)).mean()
        ema_slow = data["close"].ewm(span=params.get("ema_slow", 26)).mean()

        # RSI
        delta = data["close"].diff()
        gain = delta.where(delta > 0, 0.0).rolling(params.get("rsi_period", 14)).mean()
        loss = (-delta.where(delta < 0, 0.0)).rolling(params.get("rsi_period", 14)).mean()
        rs = gain / loss.replace(0, np.nan)
        rsi = 100 - (100 / (1 + rs))

        # 進場: EMA 金叉 + RSI 未超買
        buy_cond = (ema_fast > ema_slow) & (rsi < params.get("rsi_overbought", 70))
        sell_cond = (ema_fast < ema_slow) | (rsi > 80)

        signals[buy_cond] = 1
        signals[sell_cond] = -1
        return signals

    def _simulate(
        self, data: pd.DataFrame, signals: pd.Series, initial_capital: float
    ) -> tuple[list[float], list[dict]]:
        """模擬交易，返回權益曲線和交易記錄"""
        capital = initial_capital
        position = 0.0
        entry_price = 0.0
        equity_curve = []
        trades = []

        for i in range(len(data)):
            price = data["close"].iloc[i]

            if signals.iloc[i] == 1 and position == 0:
                position = capital * 0.95 / price  # 95% 倉位
                entry_price = price
                capital -= position * price

            elif signals.iloc[i] == -1 and position > 0:
                sell_value = position * price
                pnl = sell_value - (position * entry_price)
                trades.append({
                    "entry_price": entry_price,
                    "exit_price": price,
                    "pnl": pnl,
                    "pnl_pct": (price - entry_price) / entry_price * 100,
                    "entry_idx": None,
                    "exit_idx": i,
                })
                capital += sell_value
                position = 0.0

            equity = capital + position * price
            equity_curve.append(equity)

        return equity_curve, trades

    def _compute_metrics(
        self,
        name: str,
        data: pd.DataFrame,
        equity_curve: list[float],
        trades: list[dict],
        initial_capital: float,
    ) -> BacktestResult:
        """計算績效指標"""
        eq = np.array(equity_curve)
        returns = np.diff(eq) / eq[:-1] if len(eq) > 1 else np.array([])

        final_capital = eq[-1] if len(eq) > 0 else initial_capital
        total_return = (final_capital - initial_capital) / initial_capital * 100

        # Sharpe (年化，假設 hourly data)
        sharpe = 0.0
        if len(returns) > 1 and np.std(returns) > 0:
            sharpe = float(np.mean(returns) / np.std(returns) * np.sqrt(365 * 24))

        # Sortino
        downside = returns[returns < 0]
        sortino = 0.0
        if len(downside) > 0 and np.std(downside) > 0:
            sortino = float(np.mean(returns) / np.std(downside) * np.sqrt(365 * 24))

        # Max Drawdown
        peak = np.maximum.accumulate(eq) if len(eq) > 0 else np.array([initial_capital])
        drawdown = (peak - eq) / peak * 100 if len(eq) > 0 else np.array([0.0])
        max_dd = float(np.max(drawdown))

        # Win Rate & Profit Factor
        winning = [t for t in trades if t["pnl"] > 0]
        win_rate = len(winning) / len(trades) * 100 if trades else 0
        gross_profit = sum(t["pnl"] for t in winning)
        gross_loss = abs(sum(t["pnl"] for t in trades if t["pnl"] <= 0))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

        return BacktestResult(
            strategy_name=name,
            start_date=data.index[0] if isinstance(data.index[0], datetime) else datetime(2024, 1, 1),
            end_date=data.index[-1] if isinstance(data.index[-1], datetime) else datetime(2025, 1, 1),
            initial_capital=initial_capital,
            final_capital=float(final_capital),
            total_return_pct=float(total_return),
            sharpe_ratio=sharpe,
            sortino_ratio=sortino,
            max_drawdown_pct=max_dd,
            win_rate=win_rate,
            profit_factor=float(profit_factor),
            total_trades=len(trades),
            best_trade_pct=max((t["pnl_pct"] for t in trades), default=0),
            worst_trade_pct=min((t["pnl_pct"] for t in trades), default=0),
            equity_curve=equity_curve[-100:],  # 只保留最後 100 點
        )

    def _quality_check(self, result: BacktestResult) -> BacktestResult:
        """品質門檻檢查"""
        notes = []
        passed = True

        if result.total_trades < 30:
            notes.append("交易次數不足 30，統計顯著性不夠")
            passed = False
        if result.sharpe_ratio < 1.0:
            notes.append(f"Sharpe {result.sharpe_ratio:.2f} < 1.0，風險調整收益不佳")
            passed = False
        if result.max_drawdown_pct > 20:
            notes.append(f"最大回撤 {result.max_drawdown_pct:.1f}% > 20%，風險過高")
            passed = False
        if result.win_rate < 40:
            notes.append(f"勝率 {result.win_rate:.1f}% < 40%")
            passed = False
        if result.sharpe_ratio > 5.0:
            notes.append("Sharpe > 5.0，疑似過擬合")
            passed = False

        result.passed = passed
        result.notes = notes
        return result

    @staticmethod
    def _generate_sample_data(symbol: str, periods: int = 5000) -> pd.DataFrame:
        """生成模擬數據（開發/測試用）"""
        np.random.seed(42)
        dates = pd.date_range("2023-01-01", periods=periods, freq="h")
        price = 30000.0
        prices = []
        for _ in range(periods):
            price *= 1 + np.random.normal(0.00005, 0.005)
            prices.append(price)

        df = pd.DataFrame(
            {
                "open": prices,
                "high": [p * (1 + abs(np.random.normal(0, 0.003))) for p in prices],
                "low": [p * (1 - abs(np.random.normal(0, 0.003))) for p in prices],
                "close": prices,
                "volume": np.random.uniform(100, 10000, periods),
            },
            index=dates,
        )
        return df
