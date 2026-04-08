"""風險管理師 — 每資產風控 + 風險基礎倉位管理

核心職責：
1. 每筆交易風險 ≤ 2% 權益
2. 每資產持倉上限（BTC 40%, ETH 35%, SOL 25%）
3. 總加密貨幣曝險 ≤ 70%（高相關性）
4. 每資產獨立 MDD 追蹤
5. 異常市場自動降低曝險
"""

from __future__ import annotations

from typing import Optional, Dict

import numpy as np

from agents.base_agent import BaseAgent
from models.messages import AgentMessage, MessageType
from models.portfolio import PortfolioState, Position
from models.risk import RiskLimits, RiskReport
from config.settings import settings
from notifications.telegram_bot import notifier as tg
from notifications.trade_logger import trade_logger


class RiskManager(BaseAgent):
    """
    風險管理師：
    - 即時監控投資組合風險
    - 設定並執行風險限額（每資產獨立）
    - 風險基礎倉位計算
    - 在風險超標時發出警報 / 強制減倉
    """

    def __init__(self, limits: Optional[RiskLimits] = None):
        super().__init__(agent_id="risk_manager", name="風險管理師")
        self.limits = limits or RiskLimits()
        self.portfolio = PortfolioState()
        self.peak_equity: float = settings.trading.initial_capital
        self.daily_trade_count: int = 0
        self.return_history: list[float] = []

        # ── 每資產追蹤 ──
        self.asset_peak_equity: Dict[str, float] = {}
        self.asset_current_pnl: Dict[str, float] = {}
        self.active_positions: Dict[str, dict] = {}  # {symbol: {side, size_pct, entry_price}}

    async def on_start(self):
        self.state["risk_level"] = "normal"
        self.logger.info(
            f"風控啟動 | 每筆風險 {settings.risk.risk_per_trade_pct}% | "
            f"總曝險限制 {settings.risk.max_correlated_exposure:.0%} | "
            f"資產: {list(settings.risk.asset_configs.keys())}"
        )

    async def handle_message(self, message: AgentMessage):
        handlers = {
            MessageType.BACKTEST_RESULT: self._on_backtest_result,
            MessageType.TRADE_SIGNAL: self._on_trade_signal,
            MessageType.EXECUTION_REPORT: self._on_execution_report,
            MessageType.ANOMALY_ALERT: self._on_anomaly_alert,
            MessageType.COMMAND: self._on_command,
        }
        handler = handlers.get(message.msg_type)
        if handler:
            await handler(message)

    # ── 消息處理 ─────────────────────────────

    async def _on_backtest_result(self, msg: AgentMessage):
        """評估回測策略的風險"""
        result = msg.payload
        mdd = result.get("max_drawdown_pct", 0)
        sharpe = result.get("sharpe_ratio", 0)

        if mdd > self.limits.max_drawdown_pct:
            self.logger.warning(
                f"策略 [{result.get('strategy_name')}] MDD {mdd:.1f}% "
                f"超過限額 {self.limits.max_drawdown_pct}%"
            )

    async def _on_trade_signal(self, msg: AgentMessage):
        """檢查交易信號是否在風險限額內"""
        signal = msg.payload
        approval = self.validate_signal(signal)

        await self.send(
            "execution_engineer",
            MessageType.RISK_ASSESSMENT,
            {
                "signal": signal,
                "approved": approval["approved"],
                "adjusted_size": approval["adjusted_size"],
                "reason": approval.get("reason", ""),
            },
        )

    async def _on_execution_report(self, msg: AgentMessage):
        """更新交易計數和持倉追蹤"""
        report = msg.payload
        symbol = report.get("symbol", "")
        self.daily_trade_count += 1

        if report.get("action") == "close":
            self.active_positions.pop(symbol, None)
        elif report.get("status") == "filled":
            fill_price = report.get("avg_fill_price", 0)
            fill_qty = report.get("filled_quantity", 0)
            equity = trade_logger.equity or settings.trading.initial_capital
            notional = fill_price * fill_qty
            self.active_positions[symbol] = {
                "side": report.get("side", ""),
                "size_pct": notional / equity if equity > 0 else 0,
                "entry_price": fill_price,
            }

        if self.daily_trade_count >= self.limits.max_daily_trades:
            self.logger.warning("已達每日交易上限")
            await self.broadcast(
                MessageType.RISK_ALERT,
                {"alert": "daily_trade_limit", "count": self.daily_trade_count},
                priority=8,
            )

    async def _on_anomaly_alert(self, msg: AgentMessage):
        """收到市場異常或新聞事件，按嚴重程度分級處理"""
        payload = msg.payload
        action = payload.get("action", "")

        if action == "pause_all":
            # ── 極端事件：全面暫停交易 ──
            self.state["risk_level"] = "critical"
            self.state["paused_by_news"] = True
            self.logger.critical(f"極端新聞事件 — 全面暫停交易: {payload.get('title', '')}")

            # 通知執行工程師取消所有掛單
            await self.send(
                "execution_engineer",
                MessageType.RISK_ALERT,
                {"action": "cancel_all", "reason": "extreme_news"},
                priority=10,
            )
            # 廣播給所有 agent
            await self.broadcast(
                MessageType.RISK_ALERT,
                {"alert": "trading_paused", "reason": "extreme_news", "details": payload},
                priority=10,
            )

        elif action == "close_longs":
            # ── 重大負面：平多倉，保留空倉 ──
            self.state["risk_level"] = "elevated"
            self.state["block_longs"] = True
            self.logger.warning(f"重大負面新聞 — 平多倉: {payload.get('title', '')}")

            # 找出所有多倉並發送平倉指令
            longs_to_close = [
                sym for sym, pos in self.active_positions.items()
                if pos.get("side") == "long"
            ]
            for sym in longs_to_close:
                await self.send(
                    "execution_engineer",
                    MessageType.RISK_ALERT,
                    {"action": "close_position", "symbol": sym, "reason": "major_negative_news"},
                    priority=9,
                )
                self.logger.info(f"發送平倉指令: {sym} (多倉)")

            await self.broadcast(
                MessageType.RISK_ALERT,
                {"alert": "close_longs", "reason": "major_negative_news", "details": payload},
                priority=9,
            )

        else:
            # ── 一般市場異常 ──
            self.state["risk_level"] = "elevated"
            self.logger.warning(f"市場異常警報: {payload}")
            await self.broadcast(
                MessageType.RISK_ALERT,
                {"alert": "market_anomaly", "details": payload, "action": "reduce_exposure"},
                priority=9,
            )

    async def _on_command(self, msg: AgentMessage):
        action = msg.payload.get("action")
        if action == "update_portfolio":
            self.update_portfolio(msg.payload.get("portfolio", {}))
        elif action == "restore_positions":
            # 重啟後恢復持倉狀態
            restored = msg.payload.get("positions", {})
            equity = msg.payload.get("equity", settings.trading.initial_capital)
            self.peak_equity = max(self.peak_equity, equity)
            for sym, pos in restored.items():
                self.active_positions[sym] = {
                    "side": pos.get("side", ""),
                    "size_pct": pos.get("notional", 0) / equity if equity > 0 else 0,
                    "entry_price": pos.get("entry_price", 0),
                }
            if restored:
                exposure = sum(p.get("size_pct", 0) for p in self.active_positions.values())
                self.logger.info(
                    f"已恢復 {len(restored)} 個持倉 | "
                    f"權益 ${equity:,.2f} | 曝險 {exposure:.1%}"
                )
        elif action == "get_report":
            report = self.generate_risk_report()
            await self.send(msg.sender, MessageType.STATUS_REPORT, report.model_dump())
        elif action == "reset_daily":
            self.daily_trade_count = 0
        elif action == "pause_trading":
            self.state["risk_level"] = "critical"
            self.state["paused_by_user"] = True
            self.logger.warning("用戶手動暫停交易")
        elif action == "resume_trading":
            # 手動恢復交易（極端新聞暫停後）
            self.state["risk_level"] = "normal"
            self.state["paused_by_news"] = False
            self.state["paused_by_user"] = False
            self.state["block_longs"] = False
            self.logger.info("交易已手動恢復")
            if tg.enabled:
                await tg.send("✅ <b>交易已恢復</b>\n風險等級: normal")
        elif action == "unblock_longs":
            # 解除禁止做多
            self.state["block_longs"] = False
            self.logger.info("做多限制已解除")
            if tg.enabled:
                await tg.send("✅ <b>做多限制已解除</b>")

    # ── 核心邏輯 ─────────────────────────────

    def validate_signal(self, signal: dict) -> dict:
        """驗證信號是否符合風控限額，並計算最終倉位大小"""
        symbol = signal.get("symbol", "")
        signal_type = signal.get("signal_type", "")
        confidence = signal.get("confidence", 0)
        suggested_size = signal.get("suggested_size", 0)
        metadata = signal.get("metadata", {})

        # ── 平倉信號直接通過 ──
        if signal_type == "close":
            return {"approved": True, "adjusted_size": 0, "reason": "平倉信號"}

        # ── 信心度門檻 ──
        if confidence < self.limits.min_signal_confidence:
            return {
                "approved": False,
                "adjusted_size": 0,
                "reason": f"信心度 {confidence:.2f} < {self.limits.min_signal_confidence}",
            }

        # ── 每日交易次數 ──
        if self.daily_trade_count >= self.limits.max_daily_trades:
            return {"approved": False, "adjusted_size": 0, "reason": "已達每日交易上限"}

        # ── 風險等級檢查 ──
        if self.state.get("risk_level") == "critical":
            return {"approved": False, "adjusted_size": 0, "reason": "風險等級: critical — 拒絕所有交易"}

        # ── 新聞事件：禁止做多 ──
        if self.state.get("block_longs") and signal_type in ("buy", "long"):
            return {"approved": False, "adjusted_size": 0, "reason": "重大負面新聞 — 暫時禁止做多"}

        # ── 每資產持倉限額 ──
        asset_cfg = settings.risk.asset_configs.get(symbol)
        max_pos_pct = asset_cfg.max_position_pct if asset_cfg else 0.30
        adjusted_size = min(suggested_size, max_pos_pct)

        # ── 總曝險檢查 ──
        current_exposure = sum(p.get("size_pct", 0) for p in self.active_positions.values())
        if current_exposure + adjusted_size > settings.risk.max_correlated_exposure:
            available = max(0, settings.risk.max_correlated_exposure - current_exposure)
            if available <= 0.02:  # 少於 2% 不值得開倉
                return {
                    "approved": False,
                    "adjusted_size": 0,
                    "reason": f"總曝險 {current_exposure:.1%} + {adjusted_size:.1%} > {settings.risk.max_correlated_exposure:.0%} 限額",
                }
            adjusted_size = available
            self.logger.info(f"曝險限制: {symbol} 倉位縮減至 {adjusted_size:.1%}")

        # ── 已有該幣種持倉 ──
        if symbol in self.active_positions:
            return {
                "approved": False,
                "adjusted_size": 0,
                "reason": f"已持有 {symbol} ({self.active_positions[symbol]['side']})",
            }

        # ── 風險等級 elevated 時減半倉位 ──
        if self.state.get("risk_level") == "elevated":
            adjusted_size *= 0.5
            self.logger.info(f"風險提升: {symbol} 倉位減半至 {adjusted_size:.1%}")

        return {
            "approved": True,
            "adjusted_size": round(adjusted_size, 4),
            "reason": f"通過 | 倉位 {adjusted_size:.1%} | 曝險 {current_exposure + adjusted_size:.1%}",
        }

    def update_portfolio(self, portfolio_data: dict):
        """更新投資組合狀態"""
        self.portfolio = PortfolioState(**portfolio_data)
        equity = self.portfolio.total_equity
        if equity > self.peak_equity:
            self.peak_equity = equity

        # 回撤檢查
        if self.peak_equity > 0:
            drawdown = (self.peak_equity - equity) / self.peak_equity * 100
            if drawdown >= self.limits.max_drawdown_pct:
                self.state["risk_level"] = "critical"
                self.logger.critical(f"回撤 {drawdown:.1f}% 觸發最大回撤限額！")

    def generate_risk_report(self) -> RiskReport:
        """生成風險報告"""
        equity = self.portfolio.total_equity or settings.trading.initial_capital
        drawdown = 0.0
        if self.peak_equity > 0:
            drawdown = (self.peak_equity - equity) / self.peak_equity * 100

        # VaR 計算
        var_95 = 0.0
        cvar_95 = 0.0
        if len(self.return_history) >= 30:
            returns = np.array(self.return_history)
            var_95 = float(-np.percentile(returns, 5) * equity)
            tail = returns[returns <= np.percentile(returns, 5)]
            cvar_95 = float(-np.mean(tail) * equity) if len(tail) > 0 else var_95

        warnings = []
        if drawdown > self.limits.max_drawdown_pct * 0.8:
            warnings.append(f"回撤接近限額: {drawdown:.1f}%")

        current_exposure = sum(p.get("size_pct", 0) for p in self.active_positions.values())
        if current_exposure > settings.risk.max_correlated_exposure * 0.8:
            warnings.append(f"曝險接近限額: {current_exposure:.1%}")

        # 每資產風險摘要
        per_asset = {}
        for sym, pos in self.active_positions.items():
            cfg = settings.risk.asset_configs.get(sym)
            per_asset[sym] = {
                "side": pos.get("side"),
                "size_pct": pos.get("size_pct", 0),
                "max_allowed": cfg.max_position_pct if cfg else 0.30,
                "vol_class": cfg.volatility_class if cfg else "unknown",
            }

        return RiskReport(
            portfolio_var=var_95,
            portfolio_cvar=cvar_95,
            current_drawdown=drawdown,
            max_drawdown=drawdown,
            exposure_ratio=current_exposure,
            risk_level=self.state.get("risk_level", "normal"),
            warnings=warnings,
            per_asset_risk=per_asset,
        )
