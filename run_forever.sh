#!/bin/bash
# Single-process launcher: trading bot + MCP server (HTTP/SSE on 127.0.0.1:8765).
# Restarts on crash with exponential backoff; does NOT restart after a clean exit
# (e.g. MCP emergency_stop) or a configuration error (exit 78).

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

mkdir -p logs

PYTHON="${PYTHON:-$SCRIPT_DIR/.venv/bin/python}"
[ -x "$PYTHON" ] || PYTHON="python3"

delay=15
while true; do
    started=$(date +%s)
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting trade-bot + MCP server..."
    "$PYTHON" launch.py --forever --http --port 8765 "$@"
    code=$?
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Exited with code $code"

    case $code in
        0)  echo "Clean exit — not restarting."; exit 0 ;;
        78) echo "Configuration error — fix .env / settings, not restarting."; exit 78 ;;
    esac

    # Healthy run (>10 min) resets the backoff; quick crashes double it (max 10 min) so a
    # persistent failure cannot hammer the IG login endpoint and get the account locked.
    if [ $(( $(date +%s) - started )) -gt 600 ]; then delay=15; else delay=$(( delay * 2 > 600 ? 600 : delay * 2 )); fi
    echo "Restarting in ${delay}s..."
    sleep "$delay"
done
