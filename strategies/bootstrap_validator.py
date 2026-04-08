"""
IS/OOS Block Bootstrap 驗證器

方法:
1. 把數據切成 70% In-Sample (IS) / 30% Out-of-Sample (OOS)
2. 在 IS 上優化策略，在 OOS 上驗證
3. Block Bootstrap: 把 OOS 隨機抽取連續 block，重組成新序列
4. 重複 N 次，統計績效分佈
5. 如果 OOS 績效在 Bootstrap 分佈中的排名夠高 → 策略有效

意義:
- 避免過擬合 (overfitting)
- 驗證策略在「未見數據」上是否仍然有效
- Block 保留了時序自相關性 (比簡單隨機打亂更真實)
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Callable

import numpy as np
import pandas as pd

logger = logging.getLogger("bootstrap")


class BlockBootstrapValidator:
    """IS/OOS Block Bootstrap 分析"""

    def __init__(
        self,
        is_ratio: float = 0.7,         # In-Sample 佔比
        block_size: int = 10,           # Block 大小 (K 線數)
        n_iterations: int = 1000,       # Bootstrap 迭代次數 (原始 3 萬次太慢，先用 1K)
        confidence_level: float = 0.95, # 信心水平
    ):
        self.is_ratio = is_ratio
        self.block_size = block_size
        self.n_iterations = n_iterations
        self.confidence_level = confidence_level

    def validate(
        self,
        df: pd.DataFrame,
        strategy_backtest_fn: Callable,
        initial_capital: float = 2000.0,
    ) -> Dict:
        """
        執行完整的 IS/OOS Block Bootstrap 驗證。

        Args:
            df: 完整歷史數據
            strategy_backtest_fn: 策略回測函數 (接受 df, initial_capital)
            initial_capital: 初始資金

        Returns:
            {
                "is_metrics": dict,     # In-Sample 績效
                "oos_metrics": dict,    # Out-of-Sample 績效
                "bootstrap_distribution": dict,  # Bootstrap 分佈統計
                "p_value": float,       # OOS 績效的 p-value
                "is_significant": bool, # 是否統計顯著
            }
        """
        # 切割 IS/OOS
        split_idx = int(len(df) * self.is_ratio)
        df_is = df.iloc[:split_idx].copy()
        df_oos = df.iloc[split_idx:].copy()

        logger.info(f"IS: {len(df_is)} bars ({df_is.index[0]} → {df_is.index[-1]})")
        logger.info(f"OOS: {len(df_oos)} bars ({df_oos.index[0]} → {df_oos.index[-1]})")

        # IS 回測
        logger.info("回測 In-Sample...")
        is_result = strategy_backtest_fn(df_is, initial_capital)
        is_metrics = is_result["metrics"]

        # OOS 回測
        logger.info("回測 Out-of-Sample...")
        oos_result = strategy_backtest_fn(df_oos, initial_capital)
        oos_metrics = oos_result["metrics"]
        oos_return = oos_metrics.get("total_return_pct", 0)

        # Block Bootstrap on OOS
        logger.info(f"Block Bootstrap ({self.n_iterations} 次, block={self.block_size})...")
        bootstrap_returns = self._block_bootstrap(
            df_oos, strategy_backtest_fn, initial_capital
        )

        # 統計分析
        distribution = self._analyze_distribution(bootstrap_returns, oos_return)

        # p-value: OOS 回報在 bootstrap 分佈中的百分位
        p_value = np.mean(np.array(bootstrap_returns) >= oos_return)
        is_significant = p_value <= (1 - self.confidence_level)

        logger.info(f"IS Return: {is_metrics.get('total_return_pct', 0):.2f}%")
        logger.info(f"OOS Return: {oos_return:.2f}%")
        logger.info(f"Bootstrap Mean: {distribution['mean']:.2f}%")
        logger.info(f"p-value: {p_value:.4f} {'✓ 顯著' if is_significant else '✗ 不顯著'}")

        return {
            "is_metrics": is_metrics,
            "oos_metrics": oos_metrics,
            "is_period": f"{df_is.index[0]} → {df_is.index[-1]}",
            "oos_period": f"{df_oos.index[0]} → {df_oos.index[-1]}",
            "bootstrap_distribution": distribution,
            "bootstrap_returns": bootstrap_returns,
            "p_value": round(p_value, 4),
            "is_significant": is_significant,
        }

    def _block_bootstrap(
        self,
        df_oos: pd.DataFrame,
        backtest_fn: Callable,
        initial_capital: float,
    ) -> List[float]:
        """
        Block Bootstrap 抽樣。

        把 OOS 數據切成 blocks，隨機有放回地抽取 blocks
        組成新的序列，跑回測，記錄回報。
        """
        n = len(df_oos)
        n_blocks = max(1, n // self.block_size)
        returns = []

        for iteration in range(self.n_iterations):
            if (iteration + 1) % 100 == 0:
                logger.info(f"  Bootstrap iteration {iteration + 1}/{self.n_iterations}")

            # 隨機抽取 block 起點
            block_starts = np.random.randint(0, n - self.block_size + 1, size=n_blocks)

            # 組合成新數據
            blocks = []
            for start in block_starts:
                block = df_oos.iloc[start:start + self.block_size].copy()
                blocks.append(block)

            if not blocks:
                continue

            bootstrap_df = pd.concat(blocks, ignore_index=True)
            # 重建 datetime index
            bootstrap_df.index = pd.date_range(
                start=df_oos.index[0],
                periods=len(bootstrap_df),
                freq=pd.infer_freq(df_oos.index[:10]) or "h",
            )

            try:
                result = backtest_fn(bootstrap_df, initial_capital)
                ret = result["metrics"].get("total_return_pct", 0)
                returns.append(ret)
            except Exception:
                continue

        return returns

    @staticmethod
    def _analyze_distribution(returns: List[float], actual_return: float) -> Dict:
        """分析 bootstrap 回報分佈"""
        arr = np.array(returns)
        if len(arr) == 0:
            return {"error": "no valid bootstrap results"}

        percentile_rank = np.mean(arr <= actual_return) * 100

        return {
            "n_samples": len(arr),
            "mean": round(float(np.mean(arr)), 2),
            "std": round(float(np.std(arr)), 2),
            "median": round(float(np.median(arr)), 2),
            "min": round(float(np.min(arr)), 2),
            "max": round(float(np.max(arr)), 2),
            "pct_5": round(float(np.percentile(arr, 5)), 2),
            "pct_25": round(float(np.percentile(arr, 25)), 2),
            "pct_75": round(float(np.percentile(arr, 75)), 2),
            "pct_95": round(float(np.percentile(arr, 95)), 2),
            "actual_return": round(actual_return, 2),
            "actual_percentile_rank": round(percentile_rank, 1),
            "pct_profitable": round(float(np.mean(arr > 0) * 100), 1),
        }
