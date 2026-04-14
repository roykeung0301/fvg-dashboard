"""
VPN + Trading 健康監控

每 5 分鐘檢查：
1. VPN 連線狀態（utun 接口 + Binance API ping）
2. 交易程序是否在運行
3. 異常時自動處理：
   - VPN 斷線 → 重啟 Surfshark → 等待連線 → 重啟交易
   - 交易程序崩潰 → 自動重啟

Usage:
  python3 vpn_watchdog.py           # 前台運行
  python3 vpn_watchdog.py --daemon  # 背景運行
"""
from __future__ import annotations

import asyncio
import logging
import os
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path

# Setup
BASE_DIR = Path(__file__).resolve().parent
LOG_DIR = BASE_DIR / "logs"
LOG_DIR.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.FileHandler(LOG_DIR / "vpn_watchdog.log"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger("watchdog")

HKT = timezone(timedelta(hours=8))

# Config
CHECK_INTERVAL = 300          # 5 minutes
EXCHANGE_PING_URL = os.getenv("EXCHANGE_PING_URL", "https://contract.mexc.com/api/v1/contract/ping")
EXCHANGE_PING_TIMEOUT = 10    # seconds
VPN_RECONNECT_WAIT = 30      # seconds to wait after reconnecting VPN
VPN_MAX_RETRIES = 3           # max VPN reconnect attempts
TRADING_SCRIPT = "main.py"
TRADING_ARGS = ["--live"]


def check_vpn_interface() -> bool:
    """Check if VPN tunnel interface (utun) exists."""
    try:
        result = subprocess.run(
            ["scutil", "--nwi"],
            capture_output=True, text=True, timeout=5,
        )
        # Look for utun interface with VPN server
        return "VPN server" in result.stdout
    except Exception as e:
        logger.error(f"VPN interface check failed: {e}")
        return False


def check_exchange_api() -> bool:
    """Ping exchange API to verify connectivity."""
    try:
        result = subprocess.run(
            ["curl", "-s", "--max-time", str(EXCHANGE_PING_TIMEOUT),
             "-o", "/dev/null", "-w", "%{http_code}", EXCHANGE_PING_URL],
            capture_output=True, text=True, timeout=EXCHANGE_PING_TIMEOUT + 5,
        )
        return result.stdout.strip() == "200"
    except Exception as e:
        logger.error(f"Exchange API check failed: {e}")
        return False


def check_vpn_health() -> dict:
    """Exchange reachability check.
    Note: MEXC is reachable from user's region without VPN, so VPN interface
    check was removed. Only the API ping matters for trading health.
    """
    api_ok = check_exchange_api()
    status = "healthy" if api_ok else "api_unreachable"
    # Keep 'vpn_interface' key for backward-compat with log formatting; report observed state.
    vpn_up = check_vpn_interface()
    return {"vpn_interface": vpn_up, "binance_api": api_ok, "status": status}


def restart_surfshark() -> bool:
    """Restart Surfshark VPN app (reconnects to last server)."""
    logger.info("Restarting Surfshark VPN...")
    try:
        # Quit Surfshark
        subprocess.run(
            ["osascript", "-e", 'tell application "Surfshark" to quit'],
            timeout=10, capture_output=True,
        )
        time.sleep(5)

        # Relaunch Surfshark (it auto-connects to last server)
        subprocess.run(
            ["open", "-a", "Surfshark"],
            timeout=10, capture_output=True,
        )
        logger.info(f"Surfshark relaunched, waiting {VPN_RECONNECT_WAIT}s for connection...")
        time.sleep(VPN_RECONNECT_WAIT)

        return check_vpn_interface()
    except Exception as e:
        logger.error(f"Surfshark restart failed: {e}")
        return False


def get_trading_pid() -> int | None:
    """Find running trading process PID."""
    try:
        # Use ps + grep instead of pgrep for reliability
        result = subprocess.run(
            ["bash", "-c", f"ps aux | grep '[Pp]ython.*{TRADING_SCRIPT}' | grep -v grep | awk '{{print $2}}'"],
            capture_output=True, text=True, timeout=5,
        )
        pids = result.stdout.strip().split("\n")
        pids = [int(p) for p in pids if p.strip()]
        pids = [p for p in pids if p != os.getpid()]
        return pids[0] if pids else None
    except Exception:
        return None


def stop_trading() -> bool:
    """Gracefully stop the trading process."""
    pid = get_trading_pid()
    if pid is None:
        logger.info("Trading process not running")
        return True

    logger.info(f"Stopping trading process (PID {pid})...")
    try:
        os.kill(pid, signal.SIGTERM)
        # Wait up to 10 seconds for graceful shutdown
        for _ in range(10):
            time.sleep(1)
            if get_trading_pid() is None:
                logger.info("Trading process stopped")
                return True
        # Force kill
        os.kill(pid, signal.SIGKILL)
        time.sleep(1)
        return get_trading_pid() is None
    except ProcessLookupError:
        return True
    except Exception as e:
        logger.error(f"Failed to stop trading: {e}")
        return False


def start_trading() -> bool:
    """Start the trading process (auto-confirms live mode)."""
    if get_trading_pid() is not None:
        logger.info("Trading process already running")
        return True

    logger.info("Starting trading process...")
    try:
        # Remove stale lock file from previous crash
        lock_file = BASE_DIR / ".bot_live.lock"
        if lock_file.exists():
            lock_file.unlink(missing_ok=True)
            logger.info("Removed stale lock file")

        log_file = open(LOG_DIR / "trading_output.log", "a")
        # Pipe "yes" to auto-confirm live trading prompt
        yes_pipe = subprocess.Popen(["echo", "yes"], stdout=subprocess.PIPE)
        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        subprocess.Popen(
            [sys.executable, str(BASE_DIR / TRADING_SCRIPT)] + TRADING_ARGS,
            cwd=str(BASE_DIR),
            stdin=yes_pipe.stdout,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            env=env,
        )
        yes_pipe.stdout.close()
        time.sleep(8)  # Live mode needs more startup time
        pid = get_trading_pid()
        if pid:
            logger.info(f"Trading started (PID {pid})")
            return True
        else:
            logger.error("Trading process failed to start")
            return False
    except Exception as e:
        logger.error(f"Failed to start trading: {e}")
        return False


async def send_telegram(message: str):
    """Send Telegram notification."""
    try:
        # Load .env for token
        env_path = BASE_DIR / ".env"
        token = ""
        chat_id = ""
        if env_path.exists():
            for line in env_path.read_text().splitlines():
                if line.startswith("TELEGRAM_BOT_TOKEN="):
                    token = line.split("=", 1)[1].strip().strip("'\"")
                elif line.startswith("TELEGRAM_CHAT_ID="):
                    chat_id = line.split("=", 1)[1].strip().strip("'\"")

        if not token or not chat_id:
            return

        import aiohttp
        async with aiohttp.ClientSession() as session:
            await session.post(
                f"https://api.telegram.org/bot{token}/sendMessage",
                json={"chat_id": chat_id, "text": message, "parse_mode": "HTML"},
                timeout=aiohttp.ClientTimeout(total=10),
            )
    except Exception:
        pass  # Don't fail watchdog if Telegram is down


def run_health_check():
    """Run a single health check cycle."""
    now = datetime.now(HKT).strftime("%Y-%m-%d %H:%M:%S")
    health = check_vpn_health()
    trading_pid = get_trading_pid()

    logger.info(
        f"Health check | VPN: {'✓' if health['vpn_interface'] else '✗'} | "
        f"API: {'✓' if health['binance_api'] else '✗'} | "
        f"Trading: {'PID ' + str(trading_pid) if trading_pid else '✗ DOWN'}"
    )

    actions_taken = []

    # ── Exchange API unreachable ──
    # Don't kill trading — brief API outages are common and the bot retries.
    # Only alert; let the bot's own error handling decide when to stop.
    if health["status"] != "healthy":
        logger.warning(f"Exchange API unreachable (VPN interface: "
                       f"{'up' if health['vpn_interface'] else 'down'})")
        actions_taken.append("exchange API unreachable (no action taken)")

    # ── Trading process down ──
    if trading_pid is None:
        logger.warning("Trading process not running, restarting...")
        if start_trading():
            actions_taken.append("trading restarted")
        else:
            actions_taken.append("trading restart FAILED")

    # ── Send Telegram if any actions taken ──
    if actions_taken:
        emoji = "✅" if "FAILED" not in str(actions_taken) else "🚨"
        msg = (
            f"{emoji} <b>Watchdog Alert</b>\n"
            f"━━━━━━━━━━━━━━━\n"
            f"🕐 {now} HKT\n"
        )
        for a in actions_taken:
            msg += f"• {a}\n"

        try:
            asyncio.run(send_telegram(msg))
        except Exception:
            pass

    return health["status"] == "healthy" and trading_pid is not None


def main():
    daemon = "--daemon" in sys.argv

    logger.info("=" * 50)
    logger.info("VPN + Trading Watchdog started")
    logger.info(f"Check interval: {CHECK_INTERVAL}s ({CHECK_INTERVAL // 60}min)")
    logger.info(f"Mode: {'daemon' if daemon else 'foreground'}")
    logger.info("=" * 50)

    # Initial check
    run_health_check()

    # Main loop
    try:
        while True:
            time.sleep(CHECK_INTERVAL)
            run_health_check()
    except KeyboardInterrupt:
        logger.info("Watchdog stopped by user")


if __name__ == "__main__":
    main()
