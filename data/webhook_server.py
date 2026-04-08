"""TradingView Webhook 接收器 — 接收 Pine Script Alert"""

from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime
from typing import Optional, Callable, Awaitable, List

from aiohttp import web

logger = logging.getLogger("data.webhook")


class WebhookServer:
    """
    輕量 HTTP 伺服器，接收 TradingView Alert Webhooks。

    TradingView Essential 支援最多 20 個 alerts，可以發送 JSON 到 webhook URL。

    使用方式:
    1. 啟動此伺服器 (本地需搭配 ngrok 暴露)
    2. 在 TradingView 設定 Alert → Webhook URL
    3. Alert message 使用 JSON 格式

    TradingView Alert Message 範例 (Pine Script):
    {
        "symbol": "{{ticker}}",
        "exchange": "{{exchange}}",
        "action": "{{strategy.order.action}}",
        "price": {{close}},
        "volume": {{volume}},
        "time": "{{time}}",
        "indicator": "自訂指標名稱",
        "message": "{{strategy.order.comment}}"
    }
    """

    def __init__(self, host: str = "0.0.0.0", port: int = 8080):
        self.host = host
        self.port = port
        self._app = web.Application()
        self._app.router.add_post("/webhook", self._handle_webhook)
        self._app.router.add_get("/health", self._handle_health)
        self._callbacks: List[Callable[[dict], Awaitable[None]]] = []
        self._alert_history: List[dict] = []
        self._runner: Optional[web.AppRunner] = None

    def on_alert(self, callback: Callable[[dict], Awaitable[None]]):
        """
        註冊 alert 回調。

        callback 會收到解析後的 JSON payload:
        {
            "symbol": "BTCUSDT",
            "action": "buy",
            "price": 65000.0,
            ...
            "received_at": "2024-01-01T00:00:00"
        }
        """
        self._callbacks.append(callback)

    async def start(self):
        """啟動 webhook 伺服器"""
        self._runner = web.AppRunner(self._app)
        await self._runner.setup()
        site = web.TCPSite(self._runner, self.host, self.port)
        await site.start()
        logger.info(f"Webhook 伺服器啟動: http://{self.host}:{self.port}/webhook")
        logger.info("提示: 使用 ngrok http 8080 暴露到公網供 TradingView 使用")

    async def stop(self):
        """停止伺服器"""
        if self._runner:
            await self._runner.cleanup()
            logger.info("Webhook 伺服器已停止")

    async def _handle_webhook(self, request: web.Request) -> web.Response:
        """處理 TradingView 發來的 webhook"""
        try:
            # TradingView 可能送 JSON 或純文字
            content_type = request.content_type
            if "json" in content_type:
                payload = await request.json()
            else:
                text = await request.text()
                try:
                    payload = json.loads(text)
                except json.JSONDecodeError:
                    payload = {"raw_message": text}

            payload["received_at"] = datetime.utcnow().isoformat()
            payload["source"] = "tradingview_webhook"

            logger.info(f"收到 Alert: {payload.get('symbol', 'unknown')} — {payload.get('action', 'N/A')}")

            self._alert_history.append(payload)
            # 保留最近 500 條
            if len(self._alert_history) > 500:
                self._alert_history = self._alert_history[-500:]

            # 觸發所有回調
            for callback in self._callbacks:
                try:
                    await callback(payload)
                except Exception as e:
                    logger.error(f"Webhook callback 錯誤: {e}")

            return web.json_response({"status": "ok"})

        except Exception as e:
            logger.error(f"Webhook 處理失敗: {e}")
            return web.json_response({"status": "error", "message": str(e)}, status=400)

    async def _handle_health(self, request: web.Request) -> web.Response:
        return web.json_response({
            "status": "healthy",
            "alerts_received": len(self._alert_history),
        })

    @property
    def alert_history(self) -> List[dict]:
        return self._alert_history
