"""市場分析師 — 即時行情，市場新聞分析，異常檢測"""

from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime
from collections import deque
from typing import Optional

import aiohttp
import numpy as np

from agents.base_agent import BaseAgent
from config.settings import settings
from models.messages import AgentMessage, MessageType
from models.market import MarketSnapshot
from data.news_provider import news_provider
from notifications.telegram_bot import notifier as tg


class MarketAnalyst(BaseAgent):
    """
    市場分析師：
    - 透過幣安 WebSocket 接收即時行情
    - 異常交易量 / 價格檢測
    - 市場情緒評估
    - 廣播市場狀態給全團隊
    """

    def __init__(self):
        super().__init__(agent_id="market_analyst", name="市場分析師")
        self.price_buffer: dict[str, deque] = {}  # symbol → 最近 N 筆價格
        self.volume_buffer: dict[str, deque] = {}
        self.snapshots: dict[str, MarketSnapshot] = {}
        self._ws_task: Optional[asyncio.Task] = None
        self._news_task: Optional[asyncio.Task] = None
        self._buffer_size = 200
        self.data_manager = None  # 由 orchestrator 注入
        self.last_news_alerts: list = []  # recent news for dedup
        self._last_ticker_broadcast: dict[str, float] = {}  # throttle ticker to 1/min

    async def on_start(self):
        self.state["monitoring"] = True
        for symbol in settings.trading.symbols:
            self.price_buffer[symbol] = deque(maxlen=self._buffer_size)
            self.volume_buffer[symbol] = deque(maxlen=self._buffer_size)

    async def on_stop(self):
        self.state["monitoring"] = False
        if self._ws_task:
            self._ws_task.cancel()
        if self._news_task:
            self._news_task.cancel()
        await news_provider.close()

    async def handle_message(self, message: AgentMessage):
        handlers = {
            MessageType.COMMAND: self._on_command,
        }
        handler = handlers.get(message.msg_type)
        if handler:
            await handler(message)

    async def _on_command(self, msg: AgentMessage):
        action = msg.payload.get("action")
        if action == "start_monitoring":
            await self.start_monitoring()
        elif action == "stop_monitoring":
            self.state["monitoring"] = False
        elif action == "get_snapshot":
            symbol = msg.payload.get("symbol", "BTCUSDT")
            snapshot = self.snapshots.get(symbol)
            if snapshot:
                await self.send(msg.sender, MessageType.MARKET_UPDATE, snapshot.model_dump(mode="json"))
        elif action == "get_full_snapshot":
            symbol = msg.payload.get("symbol", "BTCUSDT")
            snapshot = await self.get_enriched_snapshot(symbol)
            await self.send(msg.sender, MessageType.MARKET_UPDATE, snapshot)

    async def get_enriched_snapshot(self, symbol: str = "BTCUSDT") -> dict:
        """
        使用所有數據源生成豐富的市場快照:
        - 即時價格 (WebSocket)
        - 資金費率 + OI (Binance)
        - 多空比 + Taker 比 (Binance)
        - Fear & Greed (alternative.me)
        - 全球市場總覽 (CoinGecko)
        """
        if not self.data_manager:
            return self.snapshots.get(symbol, MarketSnapshot(symbol=symbol, price=0)).model_dump(mode="json")

        full_snapshot = await self.data_manager.get_full_market_snapshot(symbol)

        # 合併即時數據
        if symbol in self.snapshots:
            ws_data = self.snapshots[symbol]
            full_snapshot["price"] = ws_data.price
            full_snapshot["volume_24h"] = ws_data.volume_24h
            full_snapshot["price_change_24h"] = ws_data.price_change_24h
            full_snapshot["anomalies"] = ws_data.anomalies

        return full_snapshot

    # ── 即時行情監控 ─────────────────────────

    async def start_monitoring(self):
        """啟動 WebSocket 行情監控（+ 可選新聞監控）"""
        self._ws_task = asyncio.create_task(self._ws_listener())
        if settings.trading.news_monitoring_enabled:
            self._news_task = asyncio.create_task(self._news_monitor_loop())
            self.logger.info("WebSocket 行情監控 + 新聞監控已啟動")
        else:
            self.logger.info("WebSocket 行情監控已啟動（新聞監控已關閉）")

    async def _ws_listener(self):
        """連接幣安 WebSocket 接收 1h K 線 + ticker 行情"""
        symbols = [s.lower() for s in settings.trading.symbols]
        # kline@1h 用於信號計算，ticker 用於即時價格監控
        kline_streams = [f"{s}@kline_1h" for s in symbols]
        ticker_streams = [f"{s}@ticker" for s in symbols]
        all_streams = "/".join(kline_streams + ticker_streams)
        url = f"wss://stream.binance.com:9443/stream?streams={all_streams}"

        if settings.binance.testnet:
            url = f"wss://testnet.binance.vision/stream?streams={all_streams}"

        while self.state.get("monitoring"):
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.ws_connect(url, heartbeat=30) as ws:
                        self.logger.info(f"已連接 WebSocket: {len(kline_streams)} kline + {len(ticker_streams)} ticker streams")
                        async for msg in ws:
                            if msg.type == aiohttp.WSMsgType.TEXT:
                                data = json.loads(msg.data)
                                stream = data.get("stream", "")
                                if "@kline_" in stream:
                                    await self._process_kline(data.get("data", {}))
                                elif "@ticker" in stream:
                                    await self._process_ticker(data)
                            elif msg.type == aiohttp.WSMsgType.ERROR:
                                break
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"WebSocket 斷線: {e}，5 秒後重連")
                await asyncio.sleep(5)

    async def _process_kline(self, data: dict):
        """處理 1h K 線數據 → 推送給信號工程師"""
        kline = data.get("k", {})
        symbol = kline.get("s", "")
        if not symbol:
            return

        is_closed = kline.get("x", False)  # K 線是否已收盤

        candle = {
            "open": float(kline.get("o", 0)),
            "high": float(kline.get("h", 0)),
            "low": float(kline.get("l", 0)),
            "close": float(kline.get("c", 0)),
            "volume": float(kline.get("v", 0)),
            "timestamp": kline.get("t", 0),
        }

        # 只在 K 線收盤時推送完整信號計算
        if is_closed:
            close_time = datetime.utcfromtimestamp(kline.get("T", 0) / 1000)
            await self.broadcast(
                MessageType.MARKET_UPDATE,
                {
                    "symbol": symbol,
                    "candle": candle,
                    "candle_close_time": close_time.isoformat(),
                    "price": candle["close"],
                    "volume_24h": candle["volume"],
                },
            )
            self.logger.info(
                f"1h K 線收盤 | {symbol} | "
                f"O:{candle['open']:.2f} H:{candle['high']:.2f} "
                f"L:{candle['low']:.2f} C:{candle['close']:.2f}"
            )

    async def _process_ticker(self, data: dict):
        """處理 ticker 數據（僅用於異常檢測，節流至每分鐘一次）"""
        stream_data = data.get("data", data)
        symbol = stream_data.get("s", "")
        if not symbol:
            return

        # 節流：每個幣種每 60 秒處理一次 ticker
        import time
        now = time.time()
        last = self._last_ticker_broadcast.get(symbol, 0)
        if now - last < 60:
            return
        self._last_ticker_broadcast[symbol] = now

        price = float(stream_data.get("c", 0))     # 最新價
        volume = float(stream_data.get("v", 0))     # 24h 成交量
        change_pct = float(stream_data.get("P", 0)) # 24h 漲跌幅
        high = float(stream_data.get("h", 0))
        low = float(stream_data.get("l", 0))

        # 更新緩衝區
        if symbol in self.price_buffer:
            self.price_buffer[symbol].append(price)
            self.volume_buffer[symbol].append(volume)

        # 異常檢測
        anomalies = self._detect_anomalies(symbol, price, volume)

        # Update Telegram price cache for position reports
        from notifications.telegram_bot import notifier as tg
        if tg.enabled:
            tg.last_prices[symbol] = price

        # 更新快照（不廣播 MARKET_UPDATE，kline 負責信號）
        snapshot = MarketSnapshot(
            symbol=symbol,
            price=price,
            volume_24h=volume,
            price_change_24h=change_pct,
            high_24h=high,
            low_24h=low,
            market_sentiment=self._assess_sentiment(change_pct),
            anomalies=anomalies,
        )
        self.snapshots[symbol] = snapshot

        # 異常警報
        if anomalies:
            await self.send(
                "risk_manager",
                MessageType.ANOMALY_ALERT,
                {"symbol": symbol, "anomalies": anomalies, "price": price},
                priority=7,
            )

    # ── 異常檢測 ─────────────────────────────

    def _detect_anomalies(self, symbol: str, price: float, volume: float) -> list[str]:
        """基於統計的異常檢測"""
        anomalies = []
        prices = self.price_buffer.get(symbol)
        volumes = self.volume_buffer.get(symbol)

        if not prices or len(prices) < 30:
            return anomalies

        price_arr = np.array(prices)
        vol_arr = np.array(volumes)

        # 價格 Z-score 異常
        price_mean = np.mean(price_arr)
        price_std = np.std(price_arr)
        if price_std > 0:
            z_score = abs(price - price_mean) / price_std
            if z_score > 3:
                anomalies.append(f"price_zscore_{z_score:.1f}")

        # 價格急漲急跌 (5 分鐘內)
        if len(prices) >= 5:
            recent_change = (price - price_arr[-5]) / price_arr[-5] * 100
            if abs(recent_change) > 2:  # 2% 短期波動
                direction = "spike" if recent_change > 0 else "dump"
                anomalies.append(f"price_{direction}_{recent_change:.1f}pct")

        # 成交量異常
        if len(volumes) >= 20:
            vol_mean = np.mean(vol_arr[-20:])
            if vol_mean > 0 and volume > vol_mean * 3:
                anomalies.append(f"volume_surge_{volume / vol_mean:.1f}x")

        return anomalies

    @staticmethod
    def _assess_sentiment(change_pct: float) -> str:
        """簡易市場情緒評估"""
        if change_pct > 3:
            return "bullish"
        elif change_pct < -3:
            return "bearish"
        return "neutral"

    # ── 新聞監控 ─────────────────────────────

    async def _news_monitor_loop(self):
        """每 5 分鐘檢查新聞，按嚴重程度分級處理"""
        await asyncio.sleep(10)  # initial delay
        while self.state.get("monitoring"):
            try:
                news = await news_provider.fetch_latest_news(limit=20)
                for article in news:
                    severity = article.get("severity", 0)
                    if severity >= 2:
                        await self._handle_news_event(article)
                    elif severity == 0 and article.get("sentiment") == "negative":
                        # Log minor negative, no action
                        self.logger.info(f"Minor negative news: {article['title'][:80]}")
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"News monitor error: {e}")

            await asyncio.sleep(300)  # 5 minutes

    async def _handle_news_event(self, article: dict):
        """Handle significant news events with graded response."""
        severity = article["severity"]
        title = article["title"]
        coins = article.get("relevant_coins", [])
        coins_str = ", ".join(coins) if coins else "crypto"

        if severity == 3:
            # ── EXTREME: Pause all trading ──
            self.logger.critical(f"EXTREME NEWS EVENT: {title}")

            # Notify risk manager to go critical
            await self.send(
                "risk_manager",
                MessageType.ANOMALY_ALERT,
                {
                    "type": "extreme_news",
                    "severity": 3,
                    "title": title,
                    "coins": coins,
                    "action": "pause_all",
                    "anomalies": [f"extreme_news: {title[:60]}"],
                },
                priority=10,
            )

            # Telegram alert
            if tg.enabled:
                await tg.send(
                    f"🚨 <b>EXTREME NEWS — TRADING PAUSED</b>\n"
                    f"━━━━━━━━━━━━━━━\n"
                    f"📰 {title}\n"
                    f"🪙 Coins: {coins_str}\n"
                    f"⚡ Action: ALL trading paused\n"
                    f"👉 Use /status to check, manual restart needed\n"
                    f"🔗 {article.get('url', '')}"
                )

        elif severity == 2:
            # ── MAJOR NEGATIVE: Close longs, keep shorts ──
            self.logger.warning(f"MAJOR NEGATIVE NEWS: {title}")

            await self.send(
                "risk_manager",
                MessageType.ANOMALY_ALERT,
                {
                    "type": "major_negative_news",
                    "severity": 2,
                    "title": title,
                    "coins": coins,
                    "action": "close_longs",
                    "anomalies": [f"major_news: {title[:60]}"],
                },
                priority=8,
            )

            if tg.enabled:
                await tg.send(
                    f"⚠️ <b>MAJOR NEGATIVE NEWS</b>\n"
                    f"━━━━━━━━━━━━━━━\n"
                    f"📰 {title}\n"
                    f"🪙 Coins: {coins_str}\n"
                    f"⚡ Action: Closing long positions, shorts OK\n"
                    f"🔗 {article.get('url', '')}"
                )

    # ── 手動行情獲取 ─────────────────────────

    async def fetch_ticker(self, symbol: str) -> Optional[dict]:
        """透過 REST API 獲取即時行情"""
        base = settings.binance.base_url.replace("fapi", "api")
        url = f"https://api.binance.com/api/v3/ticker/24hr?symbol={symbol}"
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url) as resp:
                    if resp.status == 200:
                        return await resp.json()
        except Exception as e:
            self.logger.error(f"獲取 {symbol} 行情失敗: {e}")
        return None

    async def fetch_klines(
        self, symbol: str, interval: str = "1h", limit: int = 100
    ) -> list[list]:
        """獲取 K 線數據"""
        url = f"https://api.binance.com/api/v3/klines?symbol={symbol}&interval={interval}&limit={limit}"
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url) as resp:
                    if resp.status == 200:
                        return await resp.json()
        except Exception as e:
            self.logger.error(f"獲取 {symbol} K線失敗: {e}")
        return []
