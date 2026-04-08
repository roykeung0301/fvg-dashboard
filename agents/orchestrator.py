"""團隊調度器 — 協調 FVG Trend Follow 交易系統

工作流程:
1. 啟動所有 6 agent
2. 載入歷史數據到信號工程師
3. 部署 FVG Trend Follow Combo3 策略
4. 啟動市場監控 (BTC/ETH/SOL)
5. 信號 → 風控 → 執行 的自動化管線
"""

from __future__ import annotations

import asyncio
import logging
from typing import Optional, List

from agents.base_agent import MessageBus
from agents.quant_researcher import QuantResearcher
from agents.backtest_engineer import BacktestEngineer
from agents.risk_manager import RiskManager
from agents.signal_engineer import SignalEngineer
from agents.execution_engineer import ExecutionEngineer
from agents.market_analyst import MarketAnalyst
from data.data_manager import DataManager
from data.historical_fetcher import load_data
from models.messages import AgentMessage, MessageType
from models.risk import RiskLimits
from config.settings import settings
from notifications.telegram_bot import notifier as tg, run_periodic_reports
from notifications.trade_logger import trade_logger


logger = logging.getLogger("orchestrator")


class TeamOrchestrator:
    """
    團隊調度器：
    - 初始化並管理所有 agent
    - 透過 MessageBus 路由消息
    - 整合 DataManager 提供數據
    - 提供高層操作接口
    """

    def __init__(
        self,
        risk_limits: Optional[RiskLimits] = None,
        tv_username: Optional[str] = None,
        tv_password: Optional[str] = None,
        webhook_port: int = 8080,
    ):
        self.bus = MessageBus()
        self.data = DataManager(
            tv_username=tv_username,
            tv_password=tv_password,
            webhook_port=webhook_port,
        )

        # 初始化 6 個角色
        self.quant = QuantResearcher()
        self.backtest = BacktestEngineer()
        self.risk = RiskManager(limits=risk_limits)
        self.signal = SignalEngineer()
        self.execution = ExecutionEngineer()
        self.market = MarketAnalyst()

        # 注入 data_manager 到需要數據的 agent
        self.backtest.data_manager = self.data
        self.market.data_manager = self.data

        self._agents = [
            self.quant,
            self.backtest,
            self.risk,
            self.signal,
            self.execution,
            self.market,
        ]

        # 註冊到消息匯流排
        for agent in self._agents:
            self.bus.register(agent)

    async def start(self):
        """啟動所有 agent"""
        logger.info("=" * 60)
        logger.info("FVG Trend Follow 交易系統啟動")
        logger.info(f"  資產: {settings.trading.symbols}")
        logger.info(f"  資金: ${settings.trading.initial_capital:,.0f}")
        logger.info(f"  策略: Combo3 (EMA {settings.strategy.daily_ema_fast}/{settings.strategy.daily_ema_slow}, SL {settings.strategy.sl_atr} ATR)")
        logger.info("=" * 60)
        await self.bus.start_all()
        logger.info("所有 agent 已啟動")

    async def stop(self):
        """停止所有 agent"""
        await self.bus.stop_all()
        await self.data.close()
        logger.info("所有 agent 已停止")

    # ── FVG 交易專用接口 ─────────────────────

    async def start_trading_session(self, paper: bool = True):
        """
        啟動完整交易 session:
        1. 啟動所有 agent
        2. 載入歷史數據到信號工程師
        3. 部署 FVG 策略
        4. 啟動市場監控
        """
        await self.start()

        # 載入每個幣種的歷史 1h 數據
        for symbol in settings.trading.symbols:
            df = load_data(symbol, "1h")
            if df is not None and len(df) > 500:
                await self.bus.publish(
                    AgentMessage(
                        sender="orchestrator",
                        receiver="signal_engineer",
                        msg_type=MessageType.COMMAND,
                        payload={
                            "action": "load_historical",
                            "symbol": symbol,
                            "data": df,
                        },
                    )
                )
                logger.info(f"已載入 {symbol} 歷史數據: {len(df)} bars")
            else:
                logger.warning(f"{symbol} 無歷史數據，請先執行 run_cross_asset.py 下載")

        # ── 恢復持倉狀態 ──
        await self._restore_state(paper)

        # 部署 FVG Trend Follow 策略
        await self.bus.publish(
            AgentMessage(
                sender="orchestrator",
                receiver="signal_engineer",
                msg_type=MessageType.COMMAND,
                payload={
                    "action": "deploy_strategy",
                    "strategy": {
                        "strategy_name": "FVG Trend Follow Combo3",
                        "type": "fvg_trend",
                        "symbols": settings.trading.symbols,
                    },
                },
            )
        )

        # 啟動市場監控
        await self.start_monitoring()

        if not paper:
            if settings.binance.api_key:
                await self.enable_trading()
                logger.info("實盤交易已開啟")
            else:
                logger.warning("未設定 API Key，維持紙上交易模式")
        else:
            logger.info("紙上交易模式")

        # Telegram: startup notification + periodic reports + command polling
        if tg.enabled:
            tg.initial_capital = trade_logger.initial_capital
            tg.equity = trade_logger.equity
            tg.positions = {s: p for s, p in trade_logger.positions.items()}
            tg.orchestrator = self
            mode = "LIVE" if not paper else "PAPER"
            n_pos = len(trade_logger.positions)
            restored_info = ""
            if n_pos > 0 or trade_logger.equity != trade_logger.initial_capital:
                restored_info = (
                    f"\n🔄 Restored: {n_pos} positions, "
                    f"${trade_logger.equity:,.2f} equity"
                )
            from strategies.param_optimizer import get_current_params, load_optimizer_state
            opt_params = get_current_params()
            opt_state = load_optimizer_state()
            last_opt = opt_state.get("last_optimization", "never")
            await tg.send(
                f"🤖 <b>FVG Trading System Started</b>\n"
                f"━━━━━━━━━━━━━━━\n"
                f"📋 Mode: {mode}\n"
                f"💰 Capital: ${trade_logger.equity:,.2f}\n"
                f"📊 Symbols: {', '.join(settings.trading.symbols)}\n"
                f"⚙️ Strategy: FVG Trend Follow (Dynamic)\n"
                f"   score≥{opt_params.get('entry_min_score',2)} | "
                f"FVG≥{opt_params.get('fvg_min_size_pct',0.10)}% | "
                f"SL {opt_params.get('sl_atr',6.0)} ATR\n"
                f"🔧 Last optimized: {last_opt}"
                f"{restored_info}\n\n"
                f"Commands: /status /positions /pnl /pause /help"
            )
            await tg.start_polling()
            asyncio.create_task(run_periodic_reports(tg))
            logger.info("Telegram notifications enabled")

        # Periodic parameter re-optimization check (every 24h)
        asyncio.create_task(self._periodic_reoptimization())

    async def _restore_state(self, paper: bool):
        """重啟後恢復持倉狀態到所有 agent"""
        positions = {}
        equity = settings.trading.initial_capital

        if paper:
            # Paper trading: 從 live_portfolio.json 恢復
            if trade_logger.positions:
                positions = trade_logger.positions
                equity = trade_logger.equity
                logger.info(
                    f"從 JSON 恢復 {len(positions)} 個持倉 | "
                    f"權益 ${equity:,.2f}"
                )
            else:
                equity = trade_logger.equity  # 可能無持倉但有歷史權益
                if equity != trade_logger.initial_capital:
                    logger.info(f"從 JSON 恢復權益 ${equity:,.2f}（無持倉）")
                return
        else:
            # Live trading: 從 Binance API 查詢實際持倉
            try:
                account = await self.execution.get_account_info()
                if account:
                    equity = float(account.get("totalWalletBalance", equity))
                    # 查詢每個幣種的持倉
                    for pos_data in account.get("positions", []):
                        sym = pos_data.get("symbol", "")
                        amt = float(pos_data.get("positionAmt", 0))
                        if sym in settings.trading.symbols and amt != 0:
                            entry = float(pos_data.get("entryPrice", 0))
                            positions[sym] = {
                                "side": "long" if amt > 0 else "short",
                                "entry_price": entry,
                                "quantity": abs(amt),
                                "notional": abs(amt) * entry,
                                "sl": 0,  # 無法從 API 恢復 SL
                            }
                    logger.info(
                        f"從 Binance 恢復 {len(positions)} 個持倉 | "
                        f"權益 ${equity:,.2f}"
                    )
            except Exception as e:
                logger.warning(f"Binance 持倉查詢失敗: {e}，使用本地數據")
                positions = trade_logger.positions
                equity = trade_logger.equity

        if not positions:
            return

        # 通知 risk_manager 恢復持倉
        await self.bus.publish(
            AgentMessage(
                sender="orchestrator",
                receiver="risk_manager",
                msg_type=MessageType.COMMAND,
                payload={
                    "action": "restore_positions",
                    "positions": positions,
                    "equity": equity,
                },
            )
        )

        # 通知 signal_engineer 恢復持倉（避免重複開倉）
        await self.bus.publish(
            AgentMessage(
                sender="orchestrator",
                receiver="signal_engineer",
                msg_type=MessageType.COMMAND,
                payload={
                    "action": "restore_positions",
                    "positions": positions,
                },
            )
        )

    async def run_backtest(self, symbols: Optional[List[str]] = None):
        """在指定資產上執行 FVG Combo3 回測"""
        from strategies.fvg_trend import FVGTrendStrategy

        symbols = symbols or settings.trading.symbols
        cfg = settings.strategy

        strategy = FVGTrendStrategy(
            daily_ema_fast=cfg.daily_ema_fast,
            daily_ema_slow=cfg.daily_ema_slow,
            fvg_min_size_pct=cfg.fvg_min_size_pct,
            entry_min_score=cfg.entry_min_score,
            sl_atr=cfg.sl_atr,
            fvg_max_age=cfg.fvg_max_age,
            max_active_fvgs=cfg.max_active_fvgs,
            trail_start_atr=cfg.trail_start_atr,
            breakeven_atr=cfg.breakeven_atr,
            cooldown=cfg.cooldown,
            max_hold=cfg.max_hold,
        )

        results = {}
        for symbol in symbols:
            df = load_data(symbol, "1h")
            if df is None or len(df) < 1000:
                logger.warning(f"{symbol}: 數據不足")
                continue

            r = strategy.backtest(df, settings.trading.initial_capital, cfg.max_hold)
            m = r["metrics"]
            results[symbol] = m

            if "error" not in m:
                logger.info(
                    f"{symbol}: Ret {m['total_return_pct']:+.1f}% | "
                    f"WR {m['win_rate']:.1f}% | PF {m['profit_factor']:.2f} | "
                    f"Sharpe {m['sharpe_ratio']:.2f} | MDD {m['max_drawdown_pct']:.1f}% | "
                    f"Trades {m['total_trades']}"
                )

        return results

    async def screen_asset(self, symbol: str) -> dict:
        """篩選新資產是否適合 FVG 策略"""
        from strategies.fvg_trend import FVGTrendStrategy
        from data.historical_fetcher import ensure_data

        logger.info(f"篩選新資產: {symbol}")

        # 下載數據
        df = await ensure_data(symbol, "1h", "2021-04-01", "2026-04-01")
        if df is None or len(df) < 5000:
            return {"symbol": symbol, "viable": False, "reason": "數據不足"}

        # 回測
        cfg = settings.strategy
        strategy = FVGTrendStrategy(
            daily_ema_fast=cfg.daily_ema_fast,
            daily_ema_slow=cfg.daily_ema_slow,
            fvg_min_size_pct=cfg.fvg_min_size_pct,
            entry_min_score=cfg.entry_min_score,
            sl_atr=cfg.sl_atr,
            fvg_max_age=cfg.fvg_max_age,
            max_active_fvgs=cfg.max_active_fvgs,
            trail_start_atr=cfg.trail_start_atr,
            breakeven_atr=cfg.breakeven_atr,
            cooldown=cfg.cooldown,
            max_hold=cfg.max_hold,
        )

        r = strategy.backtest(df, settings.trading.initial_capital, cfg.max_hold)
        m = r["metrics"]

        if "error" in m:
            return {"symbol": symbol, "viable": False, "reason": m["error"]}

        # 判斷是否可行
        viable = (
            m["total_return_pct"] > 100
            and m["profit_factor"] > 1.3
            and m["total_trades"] >= 50
            and m["sharpe_ratio"] > 0.8
        )

        result = {
            "symbol": symbol,
            "viable": viable,
            "return_pct": m["total_return_pct"],
            "win_rate": m["win_rate"],
            "profit_factor": m["profit_factor"],
            "sharpe": m["sharpe_ratio"],
            "max_dd": m["max_drawdown_pct"],
            "trades": m["total_trades"],
            "recommendation": "建議加入" if viable else "不建議",
        }

        logger.info(
            f"篩選結果 {symbol}: {'PASS' if viable else 'FAIL'} | "
            f"Ret {m['total_return_pct']:+.1f}% | WR {m['win_rate']:.1f}% | "
            f"PF {m['profit_factor']:.2f} | Sharpe {m['sharpe_ratio']:.2f}"
        )

        return result

    # ── 原有接口（保留）─────────────────────

    async def design_and_backtest(
        self,
        symbols: Optional[List[str]] = None,
        timeframe: str = "1h",
        use_tv_data: bool = True,
    ):
        """完整流程：拉取數據 → 設計策略 → 回測 → 風控評估"""
        symbols = symbols or settings.trading.symbols
        logger.info(f"啟動策略設計流程: symbols={symbols}, timeframe={timeframe}")

        if use_tv_data:
            for symbol in symbols:
                df = self.data.get_historical_klines(symbol, timeframe, n_bars=5000)
                if df is not None:
                    logger.info(f"已載入 {symbol} 歷史數據: {len(df)} 根 K 線")

        await self.bus.publish(
            AgentMessage(
                sender="orchestrator",
                receiver="quant_researcher",
                msg_type=MessageType.COMMAND,
                payload={
                    "action": "design_strategy",
                    "symbols": symbols,
                    "timeframe": timeframe,
                },
            )
        )

    async def _periodic_reoptimization(self):
        """Check every 24h if parameters need re-optimization."""
        await asyncio.sleep(60)  # wait 1 min for data to load
        while True:
            try:
                await self.signal.check_reoptimization()
            except Exception as e:
                logger.error(f"Re-optimization check failed: {e}")
            await asyncio.sleep(86400)  # check every 24h

    async def get_market_snapshot(self, symbol: str = "BTCUSDT") -> dict:
        return await self.data.get_full_market_snapshot(symbol)

    async def start_monitoring(self):
        logger.info("啟動市場監控")
        await self.bus.publish(
            AgentMessage(
                sender="orchestrator",
                receiver="market_analyst",
                msg_type=MessageType.COMMAND,
                payload={"action": "start_monitoring"},
            )
        )

    async def start_webhook(self):
        logger.info("啟動 TradingView Webhook 伺服器")

        async def on_tv_alert(payload: dict):
            await self.bus.publish(
                AgentMessage(
                    sender="webhook",
                    receiver="broadcast",
                    msg_type=MessageType.MARKET_UPDATE,
                    payload=payload,
                    priority=5,
                )
            )

        self.data.webhook.on_alert(on_tv_alert)
        await self.data.webhook.start()

    async def enable_trading(self):
        logger.info("開啟實盤交易")
        await self.bus.publish(
            AgentMessage(
                sender="orchestrator",
                receiver="execution_engineer",
                msg_type=MessageType.COMMAND,
                payload={"action": "enable_trading"},
            )
        )

    async def disable_trading(self):
        logger.info("關閉實盤交易")
        await self.bus.publish(
            AgentMessage(
                sender="orchestrator",
                receiver="execution_engineer",
                msg_type=MessageType.COMMAND,
                payload={"action": "disable_trading"},
            )
        )

    async def emergency_stop(self):
        logger.warning("緊急停止")
        await self.bus.publish(
            AgentMessage(
                sender="orchestrator",
                receiver="execution_engineer",
                msg_type=MessageType.COMMAND,
                payload={"action": "disable_trading"},
                priority=10,
            )
        )
        await self.bus.publish(
            AgentMessage(
                sender="orchestrator",
                receiver="execution_engineer",
                msg_type=MessageType.COMMAND,
                payload={"action": "cancel_all"},
                priority=10,
            )
        )

    async def get_risk_report(self) -> dict:
        report = self.risk.generate_risk_report()
        return report.model_dump()

    def get_team_status(self) -> dict:
        status = {}
        for agent in self._agents:
            s = agent.get_status()
            # 信號工程師額外詳情
            if hasattr(agent, "get_status_detail"):
                s["detail"] = agent.get_status_detail()
            status[agent.agent_id] = s
        return status

    def get_message_log(self, last_n: int = 20) -> list:
        messages = self.bus.message_log[-last_n:]
        return [m.model_dump(mode="json") for m in messages]
