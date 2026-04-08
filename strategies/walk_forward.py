"""
Walk-Forward Validation (Anchored)

Rolling Walk-Forward Analysis to detect overfitting:
1. Split data into N folds (e.g., 5 folds of ~1 year each)
2. For each fold: train on all previous folds, test on current fold
3. Report per-fold performance and overall consistency

Anchored approach:
- Fold 1: no training data -> skip train, only test
- Fold 2: train on fold 1, test on fold 2
- Fold 3: train on folds 1-2, test on fold 3
- ...

Overfitting risk assessment:
- LOW:    consistency >= 0.8 AND IS/OOS ratio < 3
- MEDIUM: consistency >= 0.6 OR  IS/OOS ratio < 5
- HIGH:   everything else
"""

from __future__ import annotations

import logging
from typing import Callable, Dict, List, Optional

import pandas as pd

logger = logging.getLogger("walk_forward")


class WalkForwardValidator:
    """Anchored Walk-Forward Analysis"""

    def __init__(
        self,
        n_folds: int = 5,
        min_train_bars: int = 5000,
    ):
        self.n_folds = n_folds
        self.min_train_bars = min_train_bars

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def validate(
        self,
        df: pd.DataFrame,
        backtest_fn: Callable[[pd.DataFrame, float], dict],
        initial_capital: float = 2000.0,
    ) -> dict:
        """Run anchored walk-forward analysis.

        Parameters
        ----------
        df : pd.DataFrame
            Full OHLCV dataframe (must be sorted by time ascending).
        backtest_fn : callable
            ``fn(df, initial_capital) -> dict`` where
            ``result["metrics"]`` contains at least:
            total_return_pct, win_rate, profit_factor,
            sharpe_ratio, max_drawdown_pct, total_trades.
        initial_capital : float
            Starting capital passed to *backtest_fn*.

        Returns
        -------
        dict
            ``{"folds": [...], "summary": {...}}``
        """
        folds_data = self._split_folds(df)
        fold_results: List[dict] = []

        for i, test_df in enumerate(folds_data):
            fold_num = i + 1

            # ---- test period label ----
            test_start = str(test_df.index[0])
            test_end = str(test_df.index[-1])

            # ---- train data = all previous folds combined ----
            if i == 0:
                train_df = None
                train_start = train_end = "N/A"
            else:
                train_df = pd.concat(folds_data[:i])
                train_start = str(train_df.index[0])
                train_end = str(train_df.index[-1])

            logger.info(
                "Fold %d/%d  train=[%s -> %s] (%s bars)  test=[%s -> %s] (%d bars)",
                fold_num,
                self.n_folds,
                train_start,
                train_end,
                len(train_df) if train_df is not None else 0,
                test_start,
                test_end,
                len(test_df),
            )

            # ---- run backtests ----
            train_metrics: Optional[dict] = None
            if train_df is not None and len(train_df) >= self.min_train_bars:
                train_result = backtest_fn(train_df, initial_capital)
                train_metrics = train_result.get("metrics", {})

            test_result = backtest_fn(test_df, initial_capital)
            test_metrics = test_result.get("metrics", {})

            fold_results.append(
                {
                    "fold": fold_num,
                    "train_period": f"{train_start} -> {train_end}",
                    "test_period": f"{test_start} -> {test_end}",
                    "train_bars": len(train_df) if train_df is not None else 0,
                    "test_bars": len(test_df),
                    "train_metrics": train_metrics,
                    "test_metrics": test_metrics,
                }
            )

        summary = self._compute_summary(fold_results)

        return {"folds": fold_results, "summary": summary}

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _split_folds(self, df: pd.DataFrame) -> List[pd.DataFrame]:
        """Split dataframe into *n_folds* roughly equal chunks."""
        total = len(df)
        fold_size = total // self.n_folds
        folds: List[pd.DataFrame] = []

        for i in range(self.n_folds):
            start = i * fold_size
            # last fold absorbs remainder
            end = total if i == self.n_folds - 1 else (i + 1) * fold_size
            folds.append(df.iloc[start:end].copy())

        return folds

    def _compute_summary(self, fold_results: List[dict]) -> dict:
        """Aggregate fold-level metrics into a summary."""
        test_returns: List[float] = []
        test_wrs: List[float] = []
        test_pfs: List[float] = []
        train_returns: List[float] = []
        profitable_count = 0

        for fr in fold_results:
            tm = fr["test_metrics"]
            if tm is None:
                continue

            ret = tm.get("total_return_pct", 0.0)
            test_returns.append(ret)
            test_wrs.append(tm.get("win_rate", 0.0))
            test_pfs.append(tm.get("profit_factor", 0.0))

            if ret > 0:
                profitable_count += 1

            # collect train returns where available
            trm = fr["train_metrics"]
            if trm is not None:
                train_returns.append(trm.get("total_return_pct", 0.0))

        total_folds = len(test_returns)
        avg_test_return = _safe_mean(test_returns)
        avg_test_wr = _safe_mean(test_wrs)
        avg_test_pf = _safe_mean(test_pfs)
        consistency = profitable_count / total_folds if total_folds > 0 else 0.0

        # IS vs OOS ratio (how much better is in-sample vs out-of-sample)
        avg_train_return = _safe_mean(train_returns)
        if avg_test_return != 0:
            is_vs_oos_ratio = abs(avg_train_return / avg_test_return)
        else:
            is_vs_oos_ratio = float("inf") if avg_train_return != 0 else 1.0

        overfitting_risk = self._assess_risk(consistency, is_vs_oos_ratio)

        logger.info(
            "Walk-Forward summary: consistency=%.2f  IS/OOS=%.2f  risk=%s",
            consistency,
            is_vs_oos_ratio,
            overfitting_risk,
        )

        return {
            "avg_test_return": round(avg_test_return, 4),
            "avg_test_wr": round(avg_test_wr, 4),
            "avg_test_pf": round(avg_test_pf, 4),
            "profitable_folds": profitable_count,
            "total_folds": total_folds,
            "consistency_score": round(consistency, 4),
            "is_vs_oos_ratio": round(is_vs_oos_ratio, 4),
            "overfitting_risk": overfitting_risk,
        }

    @staticmethod
    def _assess_risk(consistency: float, is_vs_oos_ratio: float) -> str:
        """Classify overfitting risk level."""
        if consistency >= 0.8 and is_vs_oos_ratio < 3:
            return "LOW"
        if consistency >= 0.6 or is_vs_oos_ratio < 5:
            return "MEDIUM"
        return "HIGH"


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _safe_mean(values: List[float]) -> float:
    """Return mean of *values*, or 0.0 if empty."""
    if not values:
        return 0.0
    return sum(values) / len(values)
