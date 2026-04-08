"""歷史數據下載器 — 從 Binance 分頁��取大量 K 線數據"""

from __future__ import annotations

import asyncio
import logging
import os
from datetime import datetime
from typing import Optional

import aiohttp
import pandas as pd

logger = logging.getLogger("data.fetcher")

BASE_URL = "https://api1.binance.com"


async def fetch_klines_page(
    session: aiohttp.ClientSession,
    symbol: str,
    interval: str,
    start_time: int,
    end_time: Optional[int] = None,
    limit: int = 1000,
) -> list:
    """拉取單頁 K 線"""
    params = {
        "symbol": symbol,
        "interval": interval,
        "startTime": start_time,
        "limit": limit,
    }
    if end_time:
        params["endTime"] = end_time

    async with session.get(f"{BASE_URL}/api/v3/klines", params=params) as resp:
        if resp.status == 200:
            return await resp.json()
        else:
            text = await resp.text()
            logger.error(f"API error {resp.status}: {text}")
            return []


async def fetch_full_history(
    symbol: str = "BTCUSDT",
    interval: str = "1h",
    start_date: str = "2021-04-01",
    end_date: str = "2026-04-01",
) -> pd.DataFrame:
    """
    分頁拉取完整歷史數據。

    Args:
        symbol: 交易對
        interval: K 線週期
        start_date: 起始日期 (YYYY-MM-DD)
        end_date: 結束日期 (YYYY-MM-DD)

    Returns:
        DataFrame [datetime, open, high, low, close, volume, ...]
    """
    start_ts = int(datetime.strptime(start_date, "%Y-%m-%d").timestamp() * 1000)
    end_ts = int(datetime.strptime(end_date, "%Y-%m-%d").timestamp() * 1000)

    all_data = []
    current_ts = start_ts
    page = 0

    async with aiohttp.ClientSession() as session:
        while current_ts < end_ts:
            page += 1
            data = await fetch_klines_page(session, symbol, interval, current_ts, end_ts)

            if not data:
                break

            all_data.extend(data)
            # 下一頁從最後一根 K 線的 close_time + 1 開始
            current_ts = data[-1][6] + 1

            if page % 10 == 0:
                logger.info(f"已拉取 {len(all_data)} 根 K 線 (page {page})")

            # 速率控制
            await asyncio.sleep(0.1)

    if not all_data:
        logger.warning(f"未取得 {symbol} 數據")
        return pd.DataFrame()

    df = pd.DataFrame(
        all_data,
        columns=[
            "open_time", "open", "high", "low", "close", "volume",
            "close_time", "quote_volume", "trades",
            "taker_buy_volume", "taker_buy_quote_volume", "ignore",
        ],
    )

    df["datetime"] = pd.to_datetime(df["open_time"], unit="ms")
    for col in ["open", "high", "low", "close", "volume", "quote_volume",
                "taker_buy_volume", "taker_buy_quote_volume"]:
        df[col] = df[col].astype(float)
    df["trades"] = df["trades"].astype(int)
    df = df.set_index("datetime")
    df = df.drop(columns=["open_time", "close_time", "ignore"])

    # 去重
    df = df[~df.index.duplicated(keep="first")]

    logger.info(
        f"完成: {symbol} {interval} | "
        f"{len(df)} 根 K 線 | "
        f"{df.index[0]} → {df.index[-1]}"
    )
    return df


def save_data(df: pd.DataFrame, symbol: str, interval: str, data_dir: str = "data"):
    """存檔為 CSV"""
    os.makedirs(data_dir, exist_ok=True)
    path = os.path.join(data_dir, f"{symbol}_{interval}.csv")
    df.to_csv(path)
    logger.info(f"已儲存: {path} ({len(df)} rows)")
    return path


def load_data(symbol: str, interval: str, data_dir: str = "data") -> Optional[pd.DataFrame]:
    """從 CSV 讀取 (避免重複下載)"""
    path = os.path.join(data_dir, f"{symbol}_{interval}.csv")
    if os.path.exists(path):
        df = pd.read_csv(path, index_col="datetime", parse_dates=True)
        logger.info(f"從快取載��: {path} ({len(df)} rows)")
        return df
    return None


async def ensure_data(
    symbol: str = "BTCUSDT",
    interval: str = "1h",
    start_date: str = "2021-04-01",
    end_date: str = "2026-04-01",
    data_dir: str = "data",
) -> pd.DataFrame:
    """確保數據存在 (有快取用快取，否則下載)"""
    df = load_data(symbol, interval, data_dir)
    if df is not None and len(df) > 100:
        return df

    df = await fetch_full_history(symbol, interval, start_date, end_date)
    if not df.empty:
        save_data(df, symbol, interval, data_dir)
    return df
