"""執行工程師 — 幣安 API，訂單執行"""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import time
import logging
from datetime import datetime
from typing import Optional
from urllib.parse import urlencode

import aiohttp

from agents.base_agent import BaseAgent
from config.settings import settings
from models.messages import AgentMessage, MessageType
from models.orders import (
    ExecutionReport,
    Order,
    OrderSide,
    OrderStatus,
    OrderType,
)
from models.signals import SignalType
from notifications.telegram_bot import notifier as tg
from notifications.trade_logger import trade_logger


class ExecutionEngineer(BaseAgent):
    """
    執行工程師：
    - 幣安 API 整合 (Spot / USDⓈ-M Futures)
    - 將交易信號轉化為訂單
    - 訂單生命週期管理
    - 執行品質追蹤
    """

    def __init__(self):
        super().__init__(agent_id="execution_engineer", name="執行工程師")
        self.pending_orders: dict[str, Order] = {}
        self.execution_reports: list[ExecutionReport] = []
        self._session: Optional[aiohttp.ClientSession] = None

    async def on_start(self):
        self._session = aiohttp.ClientSession()
        self.state["trading_enabled"] = settings.trading.trading_enabled

    async def on_stop(self):
        if self._session:
            await self._session.close()

    async def handle_message(self, message: AgentMessage):
        handlers = {
            MessageType.RISK_ASSESSMENT: self._on_risk_assessment,
            MessageType.RISK_ALERT: self._on_risk_alert,
            MessageType.COMMAND: self._on_command,
        }
        handler = handlers.get(message.msg_type)
        if handler:
            await handler(message)

    # ── 消息處理 ─────────────────────────────

    async def _on_risk_assessment(self, msg: AgentMessage):
        """風控審批結果 → 決定是否執行"""
        data = msg.payload
        if not data.get("approved"):
            self.logger.info(f"信號被風控拒絕: {data.get('reason')}")
            return

        signal = data["signal"]
        adjusted_size = data.get("adjusted_size", signal.get("suggested_size", 0))

        order = self._signal_to_order(signal, adjusted_size)
        if order:
            report = await self.execute_order(order)
            # 附帶信號資訊到 execution report（供 risk_manager 和 signal_engineer 使用）
            report_data = report.model_dump(mode="json")
            report_data["stop_loss"] = signal.get("stop_loss", 0)
            report_data["position_pct"] = order.metadata.get("position_pct", adjusted_size)
            await self.broadcast(
                MessageType.EXECUTION_REPORT,
                report_data,
            )

            # Log entry + Telegram notification
            if report.status == OrderStatus.FILLED:
                sl = signal.get("stop_loss", 0)
                side = "long" if order.side == OrderSide.BUY else "short"
                price = report.avg_fill_price
                qty = report.filled_quantity or order.quantity

                trade_logger.log_entry(
                    symbol=order.symbol, side=side, price=price,
                    quantity=qty, sl=sl,
                )

                if tg.enabled:
                    await tg.notify_entry(
                        symbol=order.symbol, side=side, price=price,
                        quantity=qty, sl=sl,
                        reason=signal.get("reason", "FVG signal"),
                    )

    async def _on_risk_alert(self, msg: AgentMessage):
        """風險警報 → 取消掛單 / 平倉"""
        action = msg.payload.get("action", "")
        if action == "reduce_exposure":
            self.logger.warning("風險警報: 取消所有掛單")
            await self.cancel_all_orders()
        elif action == "cancel_all":
            self.logger.warning(f"緊急取消所有掛單: {msg.payload.get('reason', '')}")
            await self.cancel_all_orders()
        elif action == "close_position":
            symbol = msg.payload.get("symbol", "")
            reason = msg.payload.get("reason", "risk_alert")
            self.logger.warning(f"風控平倉: {symbol} ({reason})")
            await self._close_position(symbol, reason)

    async def _close_position(self, symbol: str, reason: str = "risk"):
        """平掉指定幣種的持倉"""
        if not self.state.get("trading_enabled"):
            self.logger.info(f"[模擬] 平倉 {symbol} ({reason})")
            # Notify via Telegram
            if tg.enabled:
                await tg.send(
                    f"📤 <b>平倉 (模擬)</b>\n"
                    f"幣種: {symbol}\n"
                    f"原因: {reason}"
                )
            return

        # 查詢當前持倉
        try:
            params = {"symbol": symbol, "timestamp": str(int(time.time() * 1000))}
            positions = await self._signed_request("GET", "/fapi/v2/positionRisk", params)
            for pos in positions:
                if pos["symbol"] == symbol and float(pos["positionAmt"]) != 0:
                    amt = float(pos["positionAmt"])
                    side = OrderSide.SELL if amt > 0 else OrderSide.BUY
                    close_order = Order(
                        symbol=symbol,
                        side=side,
                        order_type=OrderType.MARKET,
                        quantity=abs(amt),
                        reduce_only=True,
                        metadata={"source": "risk_manager", "reason": reason},
                    )
                    report = await self.execute_order(close_order)
                    if report.status == OrderStatus.FILLED and tg.enabled:
                        await tg.send(
                            f"📤 <b>風控平倉</b>\n"
                            f"幣種: {symbol}\n"
                            f"數量: {abs(amt)}\n"
                            f"原因: {reason}"
                        )
        except Exception as e:
            self.logger.error(f"平倉 {symbol} 失敗: {e}")

    async def _on_command(self, msg: AgentMessage):
        action = msg.payload.get("action")
        if action == "enable_trading":
            self.state["trading_enabled"] = True
        elif action == "disable_trading":
            self.state["trading_enabled"] = False
        elif action == "cancel_all":
            await self.cancel_all_orders()

    # ── 訂單生成 ─────────────────────────────

    def _signal_to_order(self, signal: dict, size: float) -> Optional[Order]:
        """將信號轉化為訂單（size 是倉位百分比，需轉為實際數量）"""
        signal_type = signal.get("signal_type", "hold")

        if signal_type == SignalType.HOLD:
            return None

        side = OrderSide.BUY if signal_type == SignalType.BUY else OrderSide.SELL

        # size 是百分比（如 0.10 = 10%），轉換為實際幣量
        metadata = signal.get("metadata", {})
        entry_price = metadata.get("entry_price", 0)
        if entry_price <= 0:
            self.logger.error(f"信號缺少 entry_price: {signal.get('symbol')}")
            return None

        equity = metadata.get("equity", settings.trading.initial_capital)
        position_value = equity * size           # 倉位金額
        quantity = position_value / entry_price   # 實際幣量

        if quantity <= 0:
            self.logger.warning(f"計算倉位為 0: {signal.get('symbol')} size={size} price={entry_price}")
            return None

        return Order(
            symbol=signal["symbol"],
            side=side,
            order_type=OrderType.MARKET,
            quantity=round(quantity, 6),
            metadata={
                "source": "signal_engine",
                "confidence": signal.get("confidence", 0),
                "signal_price": entry_price,
                "position_pct": size,
            },
        )

    # ── 訂單執行 ─────────────────────────────

    async def execute_order(self, order: Order) -> ExecutionReport:
        """執行訂單（支援實盤與模擬）"""
        if not self.state.get("trading_enabled"):
            self.logger.info(f"[模擬] {order.side.value} {order.quantity} {order.symbol}")
            return self._simulate_execution(order)

        return await self._execute_binance(order)

    def _simulate_execution(self, order: Order) -> ExecutionReport:
        """模擬執行（不實際下單，使用信號攜帶的價格）"""
        order_id = f"SIM-{int(time.time() * 1000)}"
        # 從 order metadata 取得信號價格
        sim_price = order.metadata.get("signal_price", 0)
        report = ExecutionReport(
            order_id=order_id,
            symbol=order.symbol,
            side=order.side,
            order_type=order.order_type,
            status=OrderStatus.FILLED,
            quantity=order.quantity,
            filled_quantity=order.quantity,
            avg_fill_price=sim_price,
            commission=sim_price * order.quantity * 0.0004,  # 0.04% 手續費
            slippage_bps=1.0,
        )
        self.execution_reports.append(report)
        return report

    async def _execute_binance(self, order: Order) -> ExecutionReport:
        """透過幣安 API 實際下單"""
        order_id = f"ORD-{int(time.time() * 1000)}"

        try:
            params = {
                "symbol": order.symbol,
                "side": order.side.value,
                "type": order.order_type.value,
                "quantity": str(order.quantity),
                "timestamp": str(int(time.time() * 1000)),
            }

            if order.price and order.order_type == OrderType.LIMIT:
                params["price"] = str(order.price)
                params["timeInForce"] = order.time_in_force

            if order.reduce_only:
                params["reduceOnly"] = "true"

            result = await self._signed_request("POST", "/fapi/v1/order", params)

            status = OrderStatus.SUBMITTED
            if result.get("status") == "FILLED":
                status = OrderStatus.FILLED
            elif result.get("status") == "PARTIALLY_FILLED":
                status = OrderStatus.PARTIALLY_FILLED

            report = ExecutionReport(
                order_id=result.get("orderId", order_id),
                symbol=order.symbol,
                side=order.side,
                order_type=order.order_type,
                status=status,
                quantity=order.quantity,
                filled_quantity=float(result.get("executedQty", 0)),
                avg_fill_price=float(result.get("avgPrice", 0)),
                commission=float(result.get("commission", 0)),
            )

        except Exception as e:
            self.logger.error(f"下單失敗: {e}")
            report = ExecutionReport(
                order_id=order_id,
                symbol=order.symbol,
                side=order.side,
                order_type=order.order_type,
                status=OrderStatus.REJECTED,
                quantity=order.quantity,
                error_message=str(e),
            )

        self.execution_reports.append(report)
        return report

    # ── 幣安 API 層 ─────────────────────────

    async def _signed_request(self, method: str, path: str, params: dict) -> dict:
        """發送帶簽名的幣安 API 請求"""
        query_string = urlencode(params)
        signature = hmac.new(
            settings.binance.api_secret.encode(),
            query_string.encode(),
            hashlib.sha256,
        ).hexdigest()
        params["signature"] = signature

        url = f"{settings.binance.base_url}{path}"
        headers = {"X-MBX-APIKEY": settings.binance.api_key}

        async with self._session.request(method, url, params=params, headers=headers) as resp:
            data = await resp.json()
            if resp.status != 200:
                raise RuntimeError(f"Binance API error {resp.status}: {data}")
            return data

    async def cancel_all_orders(self, symbol: Optional[str] = None):
        """取消所有掛單"""
        if not self.state.get("trading_enabled"):
            self.logger.info("[模擬] 取消所有掛單")
            self.pending_orders.clear()
            return

        symbols = [symbol] if symbol else settings.trading.symbols
        for sym in symbols:
            try:
                params = {
                    "symbol": sym,
                    "timestamp": str(int(time.time() * 1000)),
                }
                await self._signed_request("DELETE", "/fapi/v1/allOpenOrders", params)
                self.logger.info(f"已取消 {sym} 所有掛單")
            except Exception as e:
                self.logger.error(f"取消 {sym} 掛單失敗: {e}")

    async def get_account_info(self) -> dict:
        """查詢帳戶資訊"""
        params = {"timestamp": str(int(time.time() * 1000))}
        return await self._signed_request("GET", "/fapi/v2/account", params)
