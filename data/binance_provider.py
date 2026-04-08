"""幣安數據提供者 — 資金費率、未平倉合約、清算、深度"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Optional, Dict, List

import aiohttp
import pandas as pd

logger = logging.getLogger("data.binance")

# 幣安公開 API (不需 API Key)
# 如果 api.binance.com 被限制，可改用以下備用域名:
#   api1.binance.com / api2.binance.com / api3.binance.com
BASE_SPOT = "https://api1.binance.com"
BASE_FUTURES = "https://fapi.binance.com"


class BinanceProvider:
    """
    幣安公開數據提供者 (全部免費，不需 API Key)。

    提供:
    - 資金費率 (Funding Rate) — 多空情緒指標
    - 未平倉合約 (Open Interest) — 市場參與度
    - 多空持倉比 — 大戶/散戶方向
    - 爆倉/清算數據
    - 訂單簿深度
    - 歷史 K 線 (備用)
    """

    def __init__(self):
        self._session: Optional[aiohttp.ClientSession] = None

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession()
        return self._session

    async def close(self):
        if self._session and not self._session.closed:
            await self._session.close()

    # ── 資金費率 ─────────────────────────────

    async def get_funding_rate(self, symbol: str = "BTCUSDT") -> Optional[Dict]:
        """
        取得當前資金費率。
        正值 = 多方付空方 (市場偏多)
        負值 = 空方付多方 (市場偏空)
        """
        data = await self._request(f"{BASE_FUTURES}/fapi/v1/premiumIndex", {"symbol": symbol})
        if data:
            return {
                "symbol": data["symbol"],
                "funding_rate": float(data["lastFundingRate"]),
                "mark_price": float(data["markPrice"]),
                "index_price": float(data["indexPrice"]),
                "next_funding_time": datetime.fromtimestamp(data["nextFundingTime"] / 1000),
            }
        return None

    async def get_funding_rate_history(
        self, symbol: str = "BTCUSDT", limit: int = 100
    ) -> pd.DataFrame:
        """取得歷史資金費率"""
        data = await self._request(
            f"{BASE_FUTURES}/fapi/v1/fundingRate",
            {"symbol": symbol, "limit": limit},
        )
        if not data:
            return pd.DataFrame()

        df = pd.DataFrame(data)
        df["fundingTime"] = pd.to_datetime(df["fundingTime"], unit="ms")
        df["fundingRate"] = df["fundingRate"].astype(float)
        df = df.set_index("fundingTime")
        return df

    # ── 未平倉合約 ───────────────────────────

    async def get_open_interest(self, symbol: str = "BTCUSDT") -> Optional[Dict]:
        """取得當前未平倉合約量"""
        data = await self._request(f"{BASE_FUTURES}/fapi/v1/openInterest", {"symbol": symbol})
        if data:
            return {
                "symbol": data["symbol"],
                "open_interest": float(data["openInterest"]),
                "timestamp": datetime.utcnow(),
            }
        return None

    async def get_open_interest_history(
        self, symbol: str = "BTCUSDT", period: str = "1h", limit: int = 200
    ) -> pd.DataFrame:
        """
        取得歷史未平倉合約。
        period: "5m", "15m", "30m", "1h", "2h", "4h", "6h", "12h", "1d"
        """
        data = await self._request(
            f"{BASE_FUTURES}/futures/data/openInterestHist",
            {"symbol": symbol, "period": period, "limit": limit},
        )
        if not data:
            return pd.DataFrame()

        df = pd.DataFrame(data)
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        for col in ["sumOpenInterest", "sumOpenInterestValue"]:
            if col in df.columns:
                df[col] = df[col].astype(float)
        df = df.set_index("timestamp")
        return df

    # ── 多空持倉比 ───────────────────────────

    async def get_long_short_ratio(
        self, symbol: str = "BTCUSDT", period: str = "1h", limit: int = 100
    ) -> pd.DataFrame:
        """
        取得大戶多空持倉比。
        > 1 = 多方主導, < 1 = 空方主導
        """
        data = await self._request(
            f"{BASE_FUTURES}/futures/data/topLongShortAccountRatio",
            {"symbol": symbol, "period": period, "limit": limit},
        )
        if not data:
            return pd.DataFrame()

        df = pd.DataFrame(data)
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        for col in ["longShortRatio", "longAccount", "shortAccount"]:
            if col in df.columns:
                df[col] = df[col].astype(float)
        df = df.set_index("timestamp")
        return df

    async def get_global_long_short_ratio(
        self, symbol: str = "BTCUSDT", period: str = "1h", limit: int = 100
    ) -> pd.DataFrame:
        """全市場多空持倉比 (非僅大戶)"""
        data = await self._request(
            f"{BASE_FUTURES}/futures/data/globalLongShortAccountRatio",
            {"symbol": symbol, "period": period, "limit": limit},
        )
        if not data:
            return pd.DataFrame()

        df = pd.DataFrame(data)
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        for col in ["longShortRatio", "longAccount", "shortAccount"]:
            if col in df.columns:
                df[col] = df[col].astype(float)
        df = df.set_index("timestamp")
        return df

    # ── Taker 買賣比 ─────────────────────────

    async def get_taker_buy_sell_ratio(
        self, symbol: str = "BTCUSDT", period: str = "1h", limit: int = 100
    ) -> pd.DataFrame:
        """
        Taker 買賣量比。
        > 1 = 主動買入多 (看漲), < 1 = 主動賣出多 (看跌)
        """
        data = await self._request(
            f"{BASE_FUTURES}/futures/data/takerlongshortRatio",
            {"symbol": symbol, "period": period, "limit": limit},
        )
        if not data:
            return pd.DataFrame()

        df = pd.DataFrame(data)
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        for col in ["buySellRatio", "buyVol", "sellVol"]:
            if col in df.columns:
                df[col] = df[col].astype(float)
        df = df.set_index("timestamp")
        return df

    # ── 訂單簿深度 ───────────────────────────

    async def get_order_book(self, symbol: str = "BTCUSDT", limit: int = 20) -> Optional[Dict]:
        """
        取得訂單簿。
        可分析買賣牆、支撐/阻力位。
        """
        data = await self._request(
            f"{BASE_SPOT}/api/v3/depth",
            {"symbol": symbol, "limit": limit},
        )
        if not data:
            return None

        bids = [[float(p), float(q)] for p, q in data.get("bids", [])]
        asks = [[float(p), float(q)] for p, q in data.get("asks", [])]

        bid_volume = sum(q for _, q in bids)
        ask_volume = sum(q for _, q in asks)

        return {
            "symbol": symbol,
            "bids": bids,
            "asks": asks,
            "bid_volume": bid_volume,
            "ask_volume": ask_volume,
            "bid_ask_ratio": bid_volume / ask_volume if ask_volume > 0 else 0,
            "spread": asks[0][0] - bids[0][0] if bids and asks else 0,
        }

    # ── 歷史 K 線 (備用) ────────────────────

    async def get_klines(
        self,
        symbol: str = "BTCUSDT",
        interval: str = "1h",
        limit: int = 500,
    ) -> pd.DataFrame:
        """從幣安拉取 K 線 (作為 tvdatafeed 的備用)"""
        data = await self._request(
            f"{BASE_SPOT}/api/v3/klines",
            {"symbol": symbol, "interval": interval, "limit": limit},
        )
        if not data:
            return pd.DataFrame()

        df = pd.DataFrame(
            data,
            columns=[
                "open_time", "open", "high", "low", "close", "volume",
                "close_time", "quote_volume", "trades",
                "taker_buy_volume", "taker_buy_quote_volume", "ignore",
            ],
        )
        df["open_time"] = pd.to_datetime(df["open_time"], unit="ms")
        for col in ["open", "high", "low", "close", "volume", "quote_volume"]:
            df[col] = df[col].astype(float)
        df = df.set_index("open_time")
        df.index.name = "datetime"
        return df[["open", "high", "low", "close", "volume"]]

    # ── HTTP 請求 ────────────────────────────

    async def _request(self, url: str, params: Optional[Dict] = None) -> any:
        session = await self._get_session()
        try:
            async with session.get(url, params=params) as resp:
                if resp.status == 200:
                    return await resp.json()
                else:
                    text = await resp.text()
                    logger.error(f"API error {resp.status}: {text}")
                    return None
        except Exception as e:
            logger.error(f"Request failed: {url} — {e}")
            return None
