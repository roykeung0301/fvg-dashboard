"""DataManager — 統一數據管理，整合所有數據源"""

from __future__ import annotations

import asyncio
import logging
from typing import Optional, Dict, List

import pandas as pd

from data.tradingview_provider import TradingViewProvider
from data.binance_provider import BinanceProvider
from data.sentiment_provider import SentimentProvider
from data.webhook_server import WebhookServer

logger = logging.getLogger("data.manager")


class DataManager:
    """
    統一數據管理器，整合:
    - TradingView (tvdatafeed) — 歷史 K 線
    - Binance API — 資金費率、OI、多空比
    - Sentiment API — Fear & Greed, CoinGecko
    - TradingView Webhooks — 即時 alert

    提供:
    - 統一的數據獲取接口
    - 多數據源合併
    - 完整市場分析快照
    """

    def __init__(
        self,
        tv_username: Optional[str] = None,
        tv_password: Optional[str] = None,
        webhook_port: int = 8080,
    ):
        self.tv = TradingViewProvider(username=tv_username, password=tv_password)
        self.binance = BinanceProvider()
        self.sentiment = SentimentProvider()
        self.webhook = WebhookServer(port=webhook_port)

    async def close(self):
        """清理所有連接"""
        await self.binance.close()
        await self.sentiment.close()
        await self.webhook.stop()

    # ── 歷史 K 線 (主要用 TradingView) ──────

    def get_historical_klines(
        self,
        symbol: str,
        interval: str = "1h",
        n_bars: int = 5000,
    ) -> Optional[pd.DataFrame]:
        """
        拉取歷史 K 線。優先用 tvdatafeed，失敗時 fallback 到幣安。
        """
        try:
            df = self.tv.get_history(symbol, interval, n_bars)
            if df is not None and not df.empty:
                return df
        except Exception as e:
            logger.warning(f"tvdatafeed 不可用: {e}")

        logger.info("Fallback 到 Binance API 拉取 K 線")
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # 如果在 async 環境中，返回 None 讓調用者用 await
                return None
            return loop.run_until_complete(
                self.binance.get_klines(symbol, interval, min(n_bars, 1000))
            )
        except Exception as e:
            logger.warning(f"Binance fallback 也失敗: {e}")
            return None

    def get_multi_timeframe(
        self,
        symbol: str,
        intervals: Optional[List[str]] = None,
    ) -> Dict[str, pd.DataFrame]:
        """拉取多時間框架數據"""
        return self.tv.get_multi_timeframe(symbol, intervals)

    # ── 完整市場快照 ────────────────────────

    async def get_full_market_snapshot(self, symbol: str = "BTCUSDT") -> Dict:
        """
        聚合所有數據源，生成完整市場快照。

        返回:
        {
            "symbol": "BTCUSDT",
            "funding_rate": {...},
            "open_interest": {...},
            "long_short_ratio": {...},
            "taker_ratio": {...},
            "order_book": {...},
            "fear_greed": {...},
            "global_market": {...},
        }
        """
        # 並行拉取所有數據
        results = await asyncio.gather(
            self.binance.get_funding_rate(symbol),
            self.binance.get_open_interest(symbol),
            self._safe(self.binance.get_long_short_ratio(symbol, limit=1)),
            self._safe(self.binance.get_taker_buy_sell_ratio(symbol, limit=1)),
            self.binance.get_order_book(symbol, limit=20),
            self.sentiment.get_fear_greed(),
            self.sentiment.get_global_market(),
            return_exceptions=True,
        )

        snapshot = {"symbol": symbol}
        keys = [
            "funding_rate", "open_interest", "long_short_ratio",
            "taker_ratio", "order_book", "fear_greed", "global_market",
        ]
        for key, result in zip(keys, results):
            if isinstance(result, Exception):
                logger.error(f"獲取 {key} 失敗: {result}")
                snapshot[key] = None
            elif isinstance(result, pd.DataFrame):
                snapshot[key] = result.iloc[-1].to_dict() if not result.empty else None
            else:
                snapshot[key] = result

        return snapshot

    # ── 回測用完整數據包 ────────────────────

    def get_backtest_data(
        self,
        symbol: str = "BTCUSDT",
        interval: str = "1h",
        n_bars: int = 5000,
    ) -> Dict[str, pd.DataFrame]:
        """
        為回測工程師打包完整數據:
        - K 線 (來自 TradingView)
        - 資金費率歷史
        - OI 歷史
        - 多空比歷史
        """
        data = {}

        # K 線
        klines = self.get_historical_klines(symbol, interval, n_bars)
        if klines is not None:
            data["klines"] = klines

        # 以下為 async 數據，同步包裝
        loop = asyncio.get_event_loop()

        async def _fetch_extras():
            return await asyncio.gather(
                self.binance.get_funding_rate_history(symbol, limit=500),
                self.binance.get_open_interest_history(symbol, period="1h", limit=200),
                self.binance.get_long_short_ratio(symbol, period="1h", limit=200),
                return_exceptions=True,
            )

        results = loop.run_until_complete(_fetch_extras())

        names = ["funding_rate", "open_interest", "long_short_ratio"]
        for name, result in zip(names, results):
            if isinstance(result, pd.DataFrame) and not result.empty:
                data[name] = result

        return data

    @staticmethod
    async def _safe(coro):
        """安全執行 coroutine，失敗時回傳 None"""
        try:
            return await coro
        except Exception as e:
            logger.error(f"Safe exec failed: {e}")
            return None
