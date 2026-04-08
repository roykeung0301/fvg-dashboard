"""TradingView 數據提供者 — 透過 tvdatafeed 拉取歷史數據"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Optional, Dict

import pandas as pd

logger = logging.getLogger("data.tradingview")

try:
    from tvDatafeed import TvDatafeed, Interval
    TV_AVAILABLE = True

    INTERVAL_MAP: Dict[str, "Interval"] = {
        "1m": Interval.in_1_minute,
        "3m": Interval.in_3_minute,
        "5m": Interval.in_5_minute,
        "15m": Interval.in_15_minute,
        "30m": Interval.in_30_minute,
        "45m": Interval.in_45_minute,
        "1h": Interval.in_1_hour,
        "2h": Interval.in_2_hour,
        "3h": Interval.in_3_hour,
        "4h": Interval.in_4_hour,
        "1d": Interval.in_daily,
        "1w": Interval.in_weekly,
        "1M": Interval.in_monthly,
    }
except ImportError:
    TV_AVAILABLE = False
    INTERVAL_MAP = {}
    logger.warning(
        "tvdatafeed 未安裝。請執行: pip install tvdatafeed "
        "或 pip install git+https://github.com/StreamAlpha/tvdatafeed.git"
    )

# 常用加密貨幣交易所 mapping
EXCHANGE_MAP: Dict[str, str] = {
    "BTCUSDT": "BINANCE",
    "ETHUSDT": "BINANCE",
    "BNBUSDT": "BINANCE",
    "SOLUSDT": "BINANCE",
    "XRPUSDT": "BINANCE",
    "DOGEUSDT": "BINANCE",
    "ADAUSDT": "BINANCE",
    "AVAXUSDT": "BINANCE",
    "DOTUSDT": "BINANCE",
    "LINKUSDT": "BINANCE",
}


class TradingViewProvider:
    """
    透過 tvdatafeed 從 TradingView 拉取歷史 K 線數據。

    支援功能:
    - 多時間框架 (1m ~ 1M)
    - 多交易所數據
    - 本地快取避免重複請求
    - 批量拉取多幣種
    """

    def __init__(
        self,
        username: Optional[str] = None,
        password: Optional[str] = None,
    ):
        """
        初始化 TradingView 連接。
        不帶帳密也可使用（匿名登入），但有速率限制。
        帶帳密可獲得更穩定的連接。
        """
        self._username = username
        self._password = password
        self._tv: Optional[TvDatafeed] = None
        self._cache: Dict[str, pd.DataFrame] = {}

    def _connect(self):
        """建立 / 重用連接"""
        if not TV_AVAILABLE:
            raise RuntimeError(
                "tvdatafeed 未安裝。請執行: "
                "pip install git+https://github.com/StreamAlpha/tvdatafeed.git"
            )
        if self._tv is None:
            if self._username and self._password:
                self._tv = TvDatafeed(self._username, self._password)
                logger.info("TradingView 已登入連接")
            else:
                self._tv = TvDatafeed()
                logger.info("TradingView 匿名連接")
        return self._tv

    def get_history(
        self,
        symbol: str,
        interval: str = "1h",
        n_bars: int = 5000,
        exchange: Optional[str] = None,
        fut_contract: Optional[int] = None,
    ) -> Optional[pd.DataFrame]:
        """
        拉取歷史 K 線數據。

        Args:
            symbol: 交易對 (e.g. "BTCUSDT")
            interval: K 線週期 ("1m", "5m", "15m", "1h", "4h", "1d" ...)
            n_bars: 拉取的 K 線數量
            exchange: 交易所 (預設自動對應)
            fut_contract: 合約月份 (None=現貨)

        Returns:
            DataFrame with columns: open, high, low, close, volume
        """
        cache_key = f"{symbol}_{interval}_{n_bars}_{exchange}"
        if cache_key in self._cache:
            logger.debug(f"使用快取: {cache_key}")
            return self._cache[cache_key]

        tv = self._connect()

        if exchange is None:
            exchange = EXCHANGE_MAP.get(symbol, "BINANCE")

        tv_interval = INTERVAL_MAP.get(interval)
        if tv_interval is None:
            logger.error(f"不支援的時間框架: {interval}")
            return None

        try:
            logger.info(f"拉取 {exchange}:{symbol} {interval} x{n_bars}")
            df = tv.get_hist(
                symbol=symbol,
                exchange=exchange,
                interval=tv_interval,
                n_bars=n_bars,
                fut_contract=fut_contract,
            )

            if df is None or df.empty:
                logger.warning(f"未取得數據: {symbol}")
                return None

            # 標準化欄位名稱
            df = self._normalize(df, symbol)
            self._cache[cache_key] = df

            logger.info(
                f"取得 {len(df)} 根 K 線: "
                f"{df.index[0]} → {df.index[-1]}"
            )
            return df

        except Exception as e:
            logger.error(f"拉取 {symbol} 失敗: {e}")
            return None

    def get_multiple(
        self,
        symbols: list,
        interval: str = "1h",
        n_bars: int = 5000,
    ) -> Dict[str, pd.DataFrame]:
        """批量拉取多個幣種的歷史數據"""
        results = {}
        for symbol in symbols:
            df = self.get_history(symbol, interval, n_bars)
            if df is not None:
                results[symbol] = df
        return results

    def get_multi_timeframe(
        self,
        symbol: str,
        intervals: Optional[list] = None,
        n_bars: int = 1000,
    ) -> Dict[str, pd.DataFrame]:
        """
        拉取同一幣種的多時間框架數據。
        用於多時框分析 (MTF)。
        """
        if intervals is None:
            intervals = ["15m", "1h", "4h", "1d"]

        results = {}
        for interval in intervals:
            df = self.get_history(symbol, interval, n_bars)
            if df is not None:
                results[interval] = df
        return results

    def search_symbol(self, query: str, exchange: str = "BINANCE") -> list:
        """搜尋 TradingView 上的交易對"""
        tv = self._connect()
        try:
            results = tv.search_symbol(query, exchange)
            return results if results else []
        except Exception as e:
            logger.error(f"搜尋失敗: {e}")
            return []

    def clear_cache(self):
        """清除本地快取"""
        self._cache.clear()
        logger.info("快取已清除")

    @staticmethod
    def _normalize(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """標準化 DataFrame 格式"""
        # tvdatafeed 回傳的欄位: symbol, open, high, low, close, volume
        col_map = {
            "open": "open",
            "high": "high",
            "low": "low",
            "close": "close",
            "volume": "volume",
        }
        df = df.rename(columns=col_map)

        # 確保必要欄位存在
        for col in ["open", "high", "low", "close", "volume"]:
            if col not in df.columns:
                df[col] = 0.0

        # 移除 tvdatafeed 加的 symbol 欄位
        if "symbol" in df.columns:
            df = df.drop(columns=["symbol"])

        # 確保 index 是 datetime
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)

        df.index.name = "datetime"
        return df[["open", "high", "low", "close", "volume"]]
