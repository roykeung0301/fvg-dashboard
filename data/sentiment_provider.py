"""市場情緒數據提供者 — Fear & Greed Index, CoinGecko"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Optional, Dict, List

import aiohttp
import pandas as pd

logger = logging.getLogger("data.sentiment")


class SentimentProvider:
    """
    免費市場情緒數據:
    - Fear & Greed Index (alternative.me)
    - CoinGecko 市場總覽
    - CoinGecko 趨勢幣種
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

    # ── Fear & Greed Index ───────────────────

    async def get_fear_greed(self, limit: int = 1) -> Optional[Dict]:
        """
        取得 Crypto Fear & Greed Index。

        0-24: Extreme Fear (極度恐懼) — 可能是買入機會
        25-49: Fear (恐懼)
        50-74: Greed (貪婪)
        75-100: Extreme Greed (極度貪婪) — 可能是賣出信號

        Returns:
            {"value": 72, "classification": "Greed", "timestamp": ...}
        """
        data = await self._request(
            "https://api.alternative.me/fng/",
            {"limit": limit, "format": "json"},
        )
        if not data or "data" not in data:
            return None

        entry = data["data"][0]
        return {
            "value": int(entry["value"]),
            "classification": entry["value_classification"],
            "timestamp": datetime.fromtimestamp(int(entry["timestamp"])),
        }

    async def get_fear_greed_history(self, days: int = 30) -> pd.DataFrame:
        """取得歷史 Fear & Greed 數據"""
        data = await self._request(
            "https://api.alternative.me/fng/",
            {"limit": days, "format": "json"},
        )
        if not data or "data" not in data:
            return pd.DataFrame()

        records = []
        for entry in data["data"]:
            records.append({
                "timestamp": datetime.fromtimestamp(int(entry["timestamp"])),
                "value": int(entry["value"]),
                "classification": entry["value_classification"],
            })

        df = pd.DataFrame(records)
        df = df.set_index("timestamp").sort_index()
        return df

    # ── CoinGecko ────────────────────────────

    async def get_global_market(self) -> Optional[Dict]:
        """
        CoinGecko 全球市場總覽。

        Returns:
            {
                "total_market_cap_usd": ...,
                "total_volume_24h": ...,
                "btc_dominance": ...,
                "eth_dominance": ...,
                "active_cryptocurrencies": ...,
                "market_cap_change_24h_pct": ...,
            }
        """
        data = await self._request("https://api.coingecko.com/api/v3/global")
        if not data or "data" not in data:
            return None

        d = data["data"]
        return {
            "total_market_cap_usd": d.get("total_market_cap", {}).get("usd", 0),
            "total_volume_24h": d.get("total_volume", {}).get("usd", 0),
            "btc_dominance": d.get("market_cap_percentage", {}).get("btc", 0),
            "eth_dominance": d.get("market_cap_percentage", {}).get("eth", 0),
            "active_cryptocurrencies": d.get("active_cryptocurrencies", 0),
            "market_cap_change_24h_pct": d.get("market_cap_change_percentage_24h_usd", 0),
        }

    async def get_trending(self) -> List[Dict]:
        """
        CoinGecko 趨勢幣種 (過去 24h 搜尋量最高)。
        可用於發現市場熱點。
        """
        data = await self._request("https://api.coingecko.com/api/v3/search/trending")
        if not data or "coins" not in data:
            return []

        trending = []
        for item in data["coins"]:
            coin = item.get("item", {})
            trending.append({
                "id": coin.get("id"),
                "symbol": coin.get("symbol", "").upper(),
                "name": coin.get("name"),
                "market_cap_rank": coin.get("market_cap_rank"),
                "price_btc": coin.get("price_btc", 0),
            })
        return trending

    async def get_coin_data(self, coin_id: str = "bitcoin") -> Optional[Dict]:
        """取得單一幣種詳細數據"""
        data = await self._request(
            f"https://api.coingecko.com/api/v3/coins/{coin_id}",
            {
                "localization": "false",
                "tickers": "false",
                "community_data": "true",
                "developer_data": "false",
            },
        )
        if not data:
            return None

        market = data.get("market_data", {})
        sentiment = data.get("sentiment_votes_up_percentage", 0)

        return {
            "id": data.get("id"),
            "symbol": data.get("symbol", "").upper(),
            "name": data.get("name"),
            "current_price": market.get("current_price", {}).get("usd", 0),
            "market_cap": market.get("market_cap", {}).get("usd", 0),
            "market_cap_rank": market.get("market_cap_rank"),
            "total_volume": market.get("total_volume", {}).get("usd", 0),
            "price_change_24h_pct": market.get("price_change_percentage_24h", 0),
            "price_change_7d_pct": market.get("price_change_percentage_7d", 0),
            "price_change_30d_pct": market.get("price_change_percentage_30d", 0),
            "ath": market.get("ath", {}).get("usd", 0),
            "ath_change_pct": market.get("ath_change_percentage", {}).get("usd", 0),
            "sentiment_up_pct": sentiment,
        }

    # ── HTTP 請求 ────────────────────────────

    async def _request(self, url: str, params: Optional[Dict] = None):
        session = await self._get_session()
        try:
            async with session.get(url, params=params) as resp:
                if resp.status == 200:
                    return await resp.json()
                elif resp.status == 429:
                    logger.warning(f"Rate limited: {url}")
                    return None
                else:
                    logger.error(f"API error {resp.status}: {url}")
                    return None
        except Exception as e:
            logger.error(f"Request failed: {url} — {e}")
            return None
