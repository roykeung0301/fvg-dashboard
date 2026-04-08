#!/bin/bash
# Auto-update dashboard: build + commit + push
# Run by cron daily

cd "/Users/keungkachun/Desktop/Binance Trading"
export PATH="/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:$PATH"

LOG="logs/dashboard_update.log"
mkdir -p logs

echo "$(date): Starting dashboard update..." >> "$LOG"

# Build new dashboard with latest data
python3 build_dashboard.py >> "$LOG" 2>&1

if [ $? -eq 0 ]; then
    git add docs/index.html
    git diff --cached --quiet
    if [ $? -ne 0 ]; then
        git commit -m "Update dashboard $(date +%Y-%m-%d)"
        git push origin main >> "$LOG" 2>&1
        echo "$(date): Dashboard updated and pushed." >> "$LOG"
    else
        echo "$(date): No changes, skipping push." >> "$LOG"
    fi
else
    echo "$(date): Build failed!" >> "$LOG"
fi
