"""
Crypto News Provider — 免費新聞監控 + 關鍵字嚴重程度分級

來源 (全部免費，無需 API key):
- CoinTelegraph RSS (via rss2json)
- CoinDesk RSS (via rss2json)
- CoinGecko trending (備用)

事件分級:
- Level 0 (中性/正面): 忽略
- Level 1 (一般負面): 記錄，策略自行處理
- Level 2 (重大負面): 平多倉、保留空倉、通知用戶
- Level 3 (極端事件): 全面暫停、緊急通知
"""
from __future__ import annotations

import logging
import re
from datetime import datetime, timezone, timedelta
from typing import Optional

import aiohttp

logger = logging.getLogger("data.news")

HKT = timezone(timedelta(hours=8))

RSS2JSON = "https://api.rss2json.com/v1/api.json"
RSS_FEEDS = [
    "https://cointelegraph.com/rss",
    "https://coindesk.com/arc/outboundfeeds/rss/",
]

# Keywords for severity classification
EXTREME_KEYWORDS = [
    "exchange hacked", "protocol hacked", "hacked for",
    "exploit", "funds stolen", "rug pull", "rugpull",
    "depeg", "de-peg", "insolvency", "insolvent", "bankrupt",
    "withdrawal halt", "withdrawals suspended", "withdrawals frozen",
    "flash crash", "circuit breaker",
    "sec charges", "sec sues", "ceo arrested", "fraud charges",
]

MAJOR_NEGATIVE_KEYWORDS = [
    "crypto ban", "bitcoin ban", "banned crypto", "crackdown on crypto",
    "lawsuit against", "sec lawsuit", "subpoena",
    "sell-off", "selloff", "major dump", "price crash",
    "market crash", "crypto crash", "bitcoin crash",
    "plunge", "plummet",
    "whale sell", "whale dump", "mass liquidation",
    "billion liquidated", "million liquidated",
    "security breach",
    "delisted", "delisting", "trading suspended",
    "etf denied", "etf rejected",
]

POSITIVE_KEYWORDS = [
    "etf approved", "etf approval", "adoption", "partnership",
    "bullish", "rally", "surge", "all-time high", "ath",
    "institutional", "accumulation", "whale buy",
    "upgrade", "milestone", "record",
]

# Symbols we care about
WATCHED_COINS = ["btc", "bitcoin", "eth", "ethereum", "sol", "solana", "xrp", "ripple", "crypto", "binance"]


class NewsProvider:
    """Fetch and classify crypto news from free RSS sources."""

    def __init__(self):
        self._session: Optional[aiohttp.ClientSession] = None
        self._seen_ids: set = set()

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession()
        return self._session

    async def close(self):
        if self._session and not self._session.closed:
            await self._session.close()

    async def fetch_latest_news(self, limit: int = 20) -> list[dict]:
        """Fetch latest crypto news from RSS feeds."""
        news = []

        for feed_url in RSS_FEEDS:
            articles = await self._fetch_rss(feed_url, limit=limit)
            news.extend(articles)

        # Fallback: CoinGecko trending
        if not news:
            cg_news = await self._fetch_coingecko_trending()
            news.extend(cg_news)

        # Sort by publish date (newest first), deduplicate
        return news[:limit]

    async def _fetch_rss(self, feed_url: str, limit: int = 10) -> list[dict]:
        """Fetch and parse RSS feed via rss2json (free, no key)."""
        session = await self._get_session()
        params = {"rss_url": feed_url}

        try:
            async with session.get(
                RSS2JSON, params=params,
                timeout=aiohttp.ClientTimeout(total=15),
            ) as resp:
                if resp.status != 200:
                    logger.warning(f"RSS fetch error {resp.status}: {feed_url}")
                    return []
                data = await resp.json()
        except Exception as e:
            logger.error(f"RSS fetch failed ({feed_url}): {e}")
            return []

        if data.get("status") != "ok":
            logger.warning(f"RSS parse error: {data.get('message', '')}")
            return []

        feed_name = data.get("feed", {}).get("title", "Unknown")
        results = []

        for item in data.get("items", [])[:limit]:
            article_id = item.get("guid", item.get("link", ""))
            if article_id in self._seen_ids:
                continue
            self._seen_ids.add(article_id)

            title = item.get("title", "")
            categories = " ".join(item.get("categories", []))
            text_for_analysis = f"{title} {categories}"

            severity = self._classify_severity(text_for_analysis)
            sentiment = self._assess_sentiment(text_for_analysis)
            coins = self._extract_coins(text_for_analysis)

            results.append({
                "id": article_id,
                "title": title,
                "source": feed_name,
                "url": item.get("link", ""),
                "published": item.get("pubDate", datetime.now(HKT).isoformat()),
                "sentiment": sentiment,
                "severity": severity,
                "relevant_coins": coins,
            })

        return results

    async def _fetch_coingecko_trending(self) -> list[dict]:
        """Fallback: CoinGecko trending as basic market sentiment."""
        session = await self._get_session()
        try:
            async with session.get(
                "https://api.coingecko.com/api/v3/search/trending",
                timeout=aiohttp.ClientTimeout(total=15),
            ) as resp:
                if resp.status != 200:
                    return []
                data = await resp.json()
        except Exception:
            return []

        results = []
        for item in data.get("coins", [])[:5]:
            coin = item.get("item", {})
            results.append({
                "id": f"cg-{coin.get('id', '')}",
                "title": f"Trending: {coin.get('name', '')} ({coin.get('symbol', '').upper()}) - Rank #{coin.get('market_cap_rank', '?')}",
                "source": "CoinGecko",
                "url": "",
                "published": datetime.now(HKT).isoformat(),
                "sentiment": "neutral",
                "severity": 0,
                "relevant_coins": [coin.get("symbol", "").lower()],
            })
        return results

    # Words that cancel negative signals (context suggests positive/neutral)
    NEGATION_WORDS = [
        "lifted", "removed", "ends", "ended", "reversed",
        "recovers", "recovery", "rebound", "bounces",
        "game", "gaming", "launches", "launch",
        "reveals", "shows", "report says",
        "no ", "not ", "unlikely", "denies",
        "repayment", "returns funds",
    ]

    def _classify_severity(self, text: str) -> int:
        """
        Classify news severity with double-check to reduce false positives:
        0 = neutral/positive
        1 = minor negative
        2 = major negative (close longs)
        3 = extreme (pause all trading)
        """
        text_lower = text.lower()

        # Positive context cancels negative
        for kw in POSITIVE_KEYWORDS:
            if self._word_match(kw, text_lower):
                return 0

        # Negation check — if title has softening words, skip
        if self._has_negation(text_lower):
            return 0

        # Check extreme first — must be relevant
        extreme_hits = [kw for kw in EXTREME_KEYWORDS if self._word_match(kw, text_lower)]
        if extreme_hits and self._is_relevant(text_lower):
            return 3

        # Major negative — must be relevant
        major_hits = [kw for kw in MAJOR_NEGATIVE_KEYWORDS if self._word_match(kw, text_lower)]
        if major_hits and self._is_relevant(text_lower):
            return 2

        return 0

    def _has_negation(self, text: str) -> bool:
        """Check if text contains words that negate the negative sentiment."""
        for word in self.NEGATION_WORDS:
            if word in text:
                return True
        return False

    @staticmethod
    def _word_match(keyword: str, text: str) -> bool:
        """Check keyword exists as whole word(s), not as substring."""
        return bool(re.search(r'\b' + re.escape(keyword) + r'\b', text))

    def _assess_sentiment(self, text: str) -> str:
        """Keyword-based sentiment assessment."""
        text_lower = text.lower()
        neg_count = sum(1 for kw in MAJOR_NEGATIVE_KEYWORDS + EXTREME_KEYWORDS if self._word_match(kw, text_lower))
        pos_count = sum(1 for kw in POSITIVE_KEYWORDS if self._word_match(kw, text_lower))

        if neg_count > pos_count:
            return "negative"
        elif pos_count > neg_count:
            return "positive"
        return "neutral"

    def _is_relevant(self, title_lower: str) -> bool:
        """Check if news is relevant to our traded coins."""
        for coin in WATCHED_COINS:
            if coin in title_lower:
                return True
        return False

    def _extract_coins(self, text: str) -> list[str]:
        """Extract mentioned coin symbols from text."""
        text_lower = text.lower()
        found = []
        coin_map = {
            "bitcoin": "BTC", "btc": "BTC",
            "ethereum": "ETH", "eth": "ETH",
            "solana": "SOL", "sol": "SOL",
            "xrp": "XRP", "ripple": "XRP",
        }
        for keyword, symbol in coin_map.items():
            if keyword in text_lower and symbol not in found:
                found.append(symbol)
        return found


# Global singleton
news_provider = NewsProvider()
