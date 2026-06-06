#!/bin/bash
# Single-process launcher: trading bot + MCP server (HTTP/SSE on :8765).
# Restarts automatically on crash.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

mkdir -p logs

while true; do
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting trade-bot + MCP server..."
    python3 launch.py --forever --http --port 8765
    EXIT_CODE=$?
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Exited with code $EXIT_CODE — restarting in 15s..."
    sleep 15
done
