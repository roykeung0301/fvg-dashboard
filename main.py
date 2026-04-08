"""
FVG Trend Follow 加密貨幣交易系統 — 主入口

使用方式:
    python3 main.py                         # 在 BTC/ETH/SOL/XRP 上回測 Combo3
    python3 main.py --paper                 # 紙上交易模式（需歷史數據）
    python3 main.py --live                  # 實盤交易（需 .env API Key）
    python3 main.py --screen XRPUSDT       # 篩選新幣種是否適合策略
    python3 main.py --status                # 顯示團隊狀態和風險報告
    python3 main.py --snapshot BTCUSDT      # 取得完整市場快照
    python3 main.py --monitor               # 即時市場監控

環境變數 (.env):
    BINANCE_API_KEY     幣安 API Key
    BINANCE_SECRET_KEY  幣安 Secret Key
    INITIAL_CAPITAL     初始資金 (預設 2000)
    TRADING_SYMBOLS     交易對 (預設 BTCUSDT,ETHUSDT,SOLUSDT,XRPUSDT)
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys

from agents.orchestrator import TeamOrchestrator
from models.risk import RiskLimits
from config.settings import settings


def setup_logging(level: str = "INFO"):
    logging.basicConfig(
        level=getattr(logging, level),
        format="%(asctime)s | %(name)-20s | %(levelname)-7s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def create_team() -> TeamOrchestrator:
    """建立團隊（使用 settings 配置）"""
    risk_limits = RiskLimits(
        max_position_size=0.40,
        max_total_exposure=settings.risk.max_correlated_exposure,
        max_leverage=1.0,  # 現貨不用槓桿
        max_drawdown_pct=40.0,
        stop_loss_pct=settings.strategy.sl_atr * 1.5,  # 動態 SL
        max_daily_trades=settings.risk.max_daily_trades,
        min_signal_confidence=settings.risk.min_signal_confidence,
        risk_per_trade_pct=settings.risk.risk_per_trade_pct,
    )

    return TeamOrchestrator(
        risk_limits=risk_limits,
        tv_username=os.getenv("TV_USERNAME"),
        tv_password=os.getenv("TV_PASSWORD"),
    )


async def run_backtest(team: TeamOrchestrator, symbols=None):
    """在 BTC/ETH/SOL 上執行 FVG Combo3 回測"""
    print("\n" + "=" * 70)
    print("  FVG Trend Follow Combo3 — 回測報告")
    print(f"  資金: ${settings.trading.initial_capital:,.0f}")
    print(f"  策略: EMA {settings.strategy.daily_ema_fast}/{settings.strategy.daily_ema_slow} | SL {settings.strategy.sl_atr} ATR")
    print("=" * 70)

    results = await team.run_backtest(symbols)

    if results:
        print("\n" + "-" * 70)
        print(f"  {'資產':10s} | {'回報':>10s} | {'WR':>6s} | {'PF':>6s} | {'Sharpe':>7s} | {'MDD':>6s} | {'交易數':>6s}")
        print(f"  {'-'*10}-+-{'-'*10}-+-{'-'*6}-+-{'-'*6}-+-{'-'*7}-+-{'-'*6}-+-{'-'*6}")
        for sym, m in results.items():
            print(
                f"  {sym:10s} | {m['total_return_pct']:>+9.1f}% | "
                f"{m['win_rate']:>5.1f}% | {m['profit_factor']:>5.2f} | "
                f"{m['sharpe_ratio']:>6.2f} | {m['max_drawdown_pct']:>5.1f}% | {m['total_trades']:>6d}"
            )

        profitable = sum(1 for m in results.values() if m["total_return_pct"] > 0)
        print(f"\n  結果: {profitable}/{len(results)} 資產盈利")

    print()


async def run_paper(team: TeamOrchestrator):
    """紙上交易模式"""
    await team.start_trading_session(paper=True)

    print("\n紙上交易模式已啟動")
    print(f"交易對: {settings.trading.symbols}")
    print(f"資金: ${settings.trading.initial_capital:,.0f}")
    print("按 Ctrl+C 停止...\n")

    try:
        while True:
            await asyncio.sleep(60)
            # 定期輸出心跳
            status = team.get_team_status()
            active = sum(1 for s in status.values() if s["running"])
            sig_detail = status.get("signal_engineer", {}).get("detail", {})
            positions = sig_detail.get("positions", {})
            pos_str = ", ".join(f"{s}:{d}" for s, d in positions.items()) if positions else "無持倉"
            print(f"  [heartbeat] {active}/{len(status)} agents | {pos_str}")
    except KeyboardInterrupt:
        print("\n正在停止...")
        await team.stop()


async def run_live(team: TeamOrchestrator):
    """實盤交易模式"""
    if not settings.binance.api_key:
        print("ERROR: 未設定 BINANCE_API_KEY")
        print("請編輯 .env 文件填入 API Key")
        return

    print("\n" + "=" * 50)
    print("  WARNING: 即將開啟實盤交易")
    print(f"  資產: {settings.trading.symbols}")
    print(f"  資金: ${settings.trading.initial_capital:,.0f}")
    print(f"  每筆風險: {settings.risk.risk_per_trade_pct}%")
    print("=" * 50)
    confirm = input("確認開啟？(yes/no): ").strip().lower()
    if confirm != "yes":
        print("已取消")
        return

    await team.start_trading_session(paper=False)

    print("\n實盤交易已啟動，按 Ctrl+C 停止...")

    try:
        while True:
            await asyncio.sleep(60)
            status = team.get_team_status()
            active = sum(1 for s in status.values() if s["running"])
            risk = await team.get_risk_report()
            print(
                f"  [heartbeat] agents={active} | "
                f"risk={risk['risk_level']} | "
                f"dd={risk['current_drawdown']:.1f}% | "
                f"exposure={risk['exposure_ratio']:.1%}"
            )
    except KeyboardInterrupt:
        print("\n緊急停止...")
        await team.emergency_stop()
        await team.stop()


async def run_screen(team: TeamOrchestrator, symbol: str):
    """篩選新幣種"""
    print(f"\n篩選 {symbol} 是否適合 FVG Trend Follow 策略...")
    result = await team.screen_asset(symbol)

    print("\n" + "=" * 50)
    print(f"  {symbol} 篩選結果")
    print("=" * 50)
    if result.get("viable") is None:
        print(f"  失敗: {result.get('reason', 'unknown')}")
    else:
        print(f"  回報:    {result.get('return_pct', 0):+.1f}%")
        print(f"  勝率:    {result.get('win_rate', 0):.1f}%")
        print(f"  利潤因子: {result.get('profit_factor', 0):.2f}")
        print(f"  Sharpe:  {result.get('sharpe', 0):.2f}")
        print(f"  MDD:     {result.get('max_dd', 0):.1f}%")
        print(f"  交易數:  {result.get('trades', 0)}")
        print(f"\n  建議:    {'✅ ' + result['recommendation'] if result['viable'] else '❌ ' + result['recommendation']}")


async def run_status(team: TeamOrchestrator):
    """顯示系統狀態"""
    await team.start()
    await asyncio.sleep(1)

    print("\n" + "=" * 60)
    print("  系統狀態")
    print("=" * 60)

    status = team.get_team_status()
    for agent_id, info in status.items():
        print(f"  {info['name']:10s} | running={info['running']} | inbox={info['inbox_size']}")
        if "detail" in info:
            d = info["detail"]
            if "positions" in d:
                print(f"             | positions={d['positions']} | signals={d.get('signals_generated', 0)}")

    risk = await team.get_risk_report()
    print(f"\n  風險等級:   {risk['risk_level']}")
    print(f"  當前回撤:   {risk['current_drawdown']:.1f}%")
    print(f"  曝險比:     {risk['exposure_ratio']:.1%}")
    if risk.get("per_asset_risk"):
        print(f"  每資產風險: {risk['per_asset_risk']}")
    if risk["warnings"]:
        print(f"  警告:       {risk['warnings']}")

    await team.stop()


async def run_snapshot(team: TeamOrchestrator, symbol: str):
    """取得完整市場快照"""
    print(f"\n取得 {symbol} 完整市場快照...")
    snapshot = await team.get_market_snapshot(symbol)

    print("\n" + "=" * 60)
    print(f"  {symbol} 市場快照")
    print("=" * 60)

    if snapshot.get("funding_rate"):
        fr = snapshot["funding_rate"]
        print(f"  資金費率:    {fr.get('funding_rate', 0):.6f}")
        print(f"  標記價格:    {fr.get('mark_price', 0):,.2f}")

    if snapshot.get("open_interest"):
        oi = snapshot["open_interest"]
        print(f"  未平倉合約:  {oi.get('open_interest', 0):,.2f}")

    if snapshot.get("order_book"):
        ob = snapshot["order_book"]
        print(f"  買賣比:      {ob.get('bid_ask_ratio', 0):.3f}")

    if snapshot.get("fear_greed"):
        fg = snapshot["fear_greed"]
        print(f"  恐懼貪婪指數: {fg.get('value', 0)} ({fg.get('classification', '')})")

    await team.data.close()


def main():
    parser = argparse.ArgumentParser(
        description="FVG Trend Follow 加密貨幣交易系統",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("--backtest", action="store_true", help="回測 BTC/ETH/SOL (預設模式)")
    parser.add_argument("--paper", action="store_true", help="紙上交易模式")
    parser.add_argument("--live", action="store_true", help="實盤交易（需 .env API Key）")
    parser.add_argument("--screen", type=str, metavar="SYMBOL", help="篩選新幣種 (例: XRPUSDT)")
    parser.add_argument("--status", action="store_true", help="顯示系統狀態")
    parser.add_argument("--snapshot", type=str, metavar="SYMBOL", help="取得市場快照")
    parser.add_argument("--monitor", action="store_true", help="即時市場監控")
    parser.add_argument("--webhook", action="store_true", help="同時啟動 Webhook")
    parser.add_argument("--symbols", type=str, help="覆寫交易對 (逗號分隔)")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING"])
    args = parser.parse_args()

    setup_logging(args.log_level)
    team = create_team()

    symbols = None
    if args.symbols:
        symbols = [s.strip() for s in args.symbols.split(",")]

    if args.screen:
        asyncio.run(run_screen(team, args.screen))
    elif args.paper:
        asyncio.run(run_paper(team))
    elif args.live:
        asyncio.run(run_live(team))
    elif args.status:
        asyncio.run(run_status(team))
    elif args.snapshot:
        asyncio.run(run_snapshot(team, args.snapshot))
    elif args.monitor:
        async def _monitor():
            await team.start()
            await team.start_monitoring()
            if args.webhook:
                await team.start_webhook()
            print("監控模式，按 Ctrl+C 停止...")
            try:
                while True:
                    await asyncio.sleep(60)
            except KeyboardInterrupt:
                await team.stop()
        asyncio.run(_monitor())
    else:
        # 預設: 回測
        asyncio.run(run_backtest(team, symbols))


if __name__ == "__main__":
    main()
