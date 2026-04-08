"""策略基礎類"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional, List, Dict

import numpy as np
import pandas as pd


@dataclass
class TradeRecord:
    """單筆交易記錄"""
    entry_time: pd.Timestamp
    exit_time: pd.Timestamp
    side: str  # "long" / "short"
    entry_price: float
    exit_price: float
    quantity: float
    pnl: float
    pnl_pct: float
    commission: float
    exit_reason: str  # "tp" / "sl" / "signal" / "timeout"


class BaseStrategy(ABC):
    """
    策略基礎類。

    子類只需實作:
    - name: 策略名稱
    - generate_signals(df): 返回帶信號的 DataFrame
    """

    name: str = "base"
    description: str = ""

    # ── 通用參數 ─────────────────────────────
    commission_rate: float = 0.0004  # 幣安 Maker 0.02%, Taker 0.04% → 來回約 0.04%
    slippage_rate: float = 0.0001   # 滑點估計 0.01%
    max_position_pct: float = 0.95  # 最大倉位 (佔總資金比例)

    @abstractmethod
    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        生成交易信號。

        必須在 df 中加入以下欄位:
        - signal: 1=開多, -1=開空, 0=無信號
        - stop_loss: 止損價 (optional)
        - take_profit: 止盈價 (optional)

        Returns: 帶信號的 DataFrame
        """
        ...

    def backtest(
        self,
        df: pd.DataFrame,
        initial_capital: float = 2000.0,
        max_holding_bars: int = 24,  # Day trading: 最多持倉 N 根 K 線
    ) -> Dict:
        """
        執行回測。

        Args:
            df: 帶信號的 OHLCV DataFrame
            initial_capital: 初始資金
            max_holding_bars: 最大持倉時間 (K 線數)

        Returns:
            {
                "trades": List[TradeRecord],
                "equity_curve": pd.Series,
                "metrics": dict,
            }
        """
        df = self.generate_signals(df.copy())
        trades: List[TradeRecord] = []
        equity = initial_capital
        equity_curve = []
        position = None  # {side, entry_price, quantity, entry_idx, sl, tp}

        for i in range(len(df)):
            row = df.iloc[i]
            price = row["close"]

            # ── 持倉中：檢查出場 ──
            if position is not None:
                bars_held = i - position["entry_idx"]
                exit_price = None
                exit_reason = None

                # 止損
                if position["side"] == "long":
                    if row["low"] <= position["sl"]:
                        exit_price = position["sl"]
                        exit_reason = "sl"
                    elif row["high"] >= position["tp"]:
                        exit_price = position["tp"]
                        exit_reason = "tp"
                else:  # short
                    if row["high"] >= position["sl"]:
                        exit_price = position["sl"]
                        exit_reason = "sl"
                    elif row["low"] <= position["tp"]:
                        exit_price = position["tp"]
                        exit_reason = "tp"

                # 超時平倉
                if exit_price is None and bars_held >= max_holding_bars:
                    exit_price = price
                    exit_reason = "timeout"

                # 反向信號平倉
                if exit_price is None and row.get("signal", 0) != 0:
                    sig = row["signal"]
                    if (position["side"] == "long" and sig == -1) or \
                       (position["side"] == "short" and sig == 1):
                        exit_price = price
                        exit_reason = "signal"

                # 執行出場
                if exit_price is not None:
                    trade = self._close_position(position, exit_price, exit_reason, df.index[i])
                    trades.append(trade)
                    equity += trade.pnl - trade.commission
                    position = None

            # ── 無持倉：檢查進場 ──
            if position is None and row.get("signal", 0) != 0:
                signal = row["signal"]
                side = "long" if signal == 1 else "short"
                quantity = (equity * self.max_position_pct) / price

                sl = row.get("stop_loss", 0)
                tp = row.get("take_profit", 0)

                # 確保 SL/TP 有效
                if sl == 0 or pd.isna(sl):
                    sl = price * (0.985 if side == "long" else 1.015)
                if tp == 0 or pd.isna(tp):
                    tp = price * (1.005 if side == "long" else 0.995)

                position = {
                    "side": side,
                    "entry_price": price,
                    "quantity": quantity,
                    "entry_idx": i,
                    "entry_time": df.index[i],
                    "sl": sl,
                    "tp": tp,
                }

            # 記錄權益
            mark_to_market = equity
            if position is not None:
                if position["side"] == "long":
                    unrealized = (price - position["entry_price"]) * position["quantity"]
                else:
                    unrealized = (position["entry_price"] - price) * position["quantity"]
                mark_to_market = equity + unrealized
            equity_curve.append(mark_to_market)

        # 強制平倉最後一筆
        if position is not None:
            trade = self._close_position(position, df.iloc[-1]["close"], "end", df.index[-1])
            trades.append(trade)
            equity += trade.pnl - trade.commission

        equity_series = pd.Series(equity_curve, index=df.index)
        metrics = self._compute_metrics(trades, equity_series, initial_capital)

        return {
            "trades": trades,
            "equity_curve": equity_series,
            "metrics": metrics,
        }

    def _close_position(
        self, position: dict, exit_price: float, reason: str, exit_time
    ) -> TradeRecord:
        """平倉計算"""
        side = position["side"]
        entry = position["entry_price"]
        qty = position["quantity"]

        if side == "long":
            pnl = (exit_price - entry) * qty
        else:
            pnl = (entry - exit_price) * qty

        pnl_pct = pnl / (entry * qty) * 100
        commission = (entry * qty + exit_price * qty) * self.commission_rate
        # 加上滑點
        slippage_cost = (entry * qty + exit_price * qty) * self.slippage_rate
        commission += slippage_cost

        return TradeRecord(
            entry_time=position["entry_time"],
            exit_time=exit_time,
            side=side,
            entry_price=entry,
            exit_price=exit_price,
            quantity=qty,
            pnl=pnl,
            pnl_pct=pnl_pct,
            commission=commission,
            exit_reason=reason,
        )

    @staticmethod
    def _compute_metrics(
        trades: List[TradeRecord],
        equity: pd.Series,
        initial_capital: float,
    ) -> Dict:
        """計算完整績效指標"""
        if not trades:
            return {"error": "no trades"}

        net_pnls = [t.pnl - t.commission for t in trades]
        wins = [p for p in net_pnls if p > 0]
        losses = [p for p in net_pnls if p <= 0]

        total_pnl = sum(net_pnls)
        win_rate = len(wins) / len(trades) * 100

        avg_win = np.mean(wins) if wins else 0
        avg_loss = abs(np.mean(losses)) if losses else 0
        risk_reward = avg_win / avg_loss if avg_loss > 0 else float("inf")

        gross_profit = sum(wins)
        gross_loss = abs(sum(losses))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

        # 回撤
        eq_arr = equity.values
        peak = np.maximum.accumulate(eq_arr)
        drawdown = (peak - eq_arr) / peak * 100
        max_dd = float(np.max(drawdown))

        # 年化收益
        total_days = (equity.index[-1] - equity.index[0]).days
        total_years = total_days / 365.25 if total_days > 0 else 1
        total_return_pct = (eq_arr[-1] - initial_capital) / initial_capital * 100
        annual_return = ((eq_arr[-1] / initial_capital) ** (1 / total_years) - 1) * 100

        # Sharpe (日收益)
        daily_equity = equity.resample("D").last().dropna()
        daily_returns = daily_equity.pct_change().dropna()
        sharpe = 0.0
        if len(daily_returns) > 30 and daily_returns.std() > 0:
            sharpe = float(daily_returns.mean() / daily_returns.std() * np.sqrt(365))

        # Sortino
        downside = daily_returns[daily_returns < 0]
        sortino = 0.0
        if len(downside) > 0 and downside.std() > 0:
            sortino = float(daily_returns.mean() / downside.std() * np.sqrt(365))

        # Calmar
        calmar = annual_return / max_dd if max_dd > 0 else float("inf")

        # 交易頻率
        trades_per_day = len(trades) / total_days if total_days > 0 else 0

        # 連勝/連敗
        streaks = []
        current = 0
        for p in net_pnls:
            if p > 0:
                current = current + 1 if current > 0 else 1
            else:
                current = current - 1 if current < 0 else -1
            streaks.append(current)
        max_win_streak = max(streaks) if streaks else 0
        max_lose_streak = abs(min(streaks)) if streaks else 0

        # 按出場原因分類
        exit_reasons = {}
        for t in trades:
            exit_reasons[t.exit_reason] = exit_reasons.get(t.exit_reason, 0) + 1

        return {
            "total_trades": len(trades),
            "win_rate": round(win_rate, 2),
            "total_pnl": round(total_pnl, 2),
            "total_return_pct": round(total_return_pct, 2),
            "annual_return_pct": round(annual_return, 2),
            "max_drawdown_pct": round(max_dd, 2),
            "sharpe_ratio": round(sharpe, 2),
            "sortino_ratio": round(sortino, 2),
            "calmar_ratio": round(calmar, 2),
            "profit_factor": round(profit_factor, 2),
            "risk_reward_ratio": round(risk_reward, 2),
            "avg_win": round(avg_win, 2),
            "avg_loss": round(avg_loss, 2),
            "best_trade": round(max(net_pnls), 2),
            "worst_trade": round(min(net_pnls), 2),
            "trades_per_day": round(trades_per_day, 2),
            "max_win_streak": max_win_streak,
            "max_lose_streak": max_lose_streak,
            "total_commission": round(sum(t.commission for t in trades), 2),
            "final_equity": round(eq_arr[-1], 2),
            "initial_capital": initial_capital,
            "period_days": total_days,
            "exit_reasons": exit_reasons,
        }
