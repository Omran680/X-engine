#!/usr/bin/env python3
"""Watchdog probe: exit 0 if the trading loop wrote its heartbeat recently, 1 otherwise.

The loop writes the heartbeat on every iteration — also while the market is closed
(polling every 60-300 s) — so a stale file means the process is hung, not just idle.

    python healthcheck.py [max_age_seconds]     # default 600
"""
import json
import sys
import time

from trade_bot.core.config import HEARTBEAT_FILE

max_age = float(sys.argv[1]) if len(sys.argv) > 1 else 600.0
try:
    with open(HEARTBEAT_FILE) as f:
        hb = json.load(f)
    age = time.time() - hb["ts"]
except (OSError, ValueError, KeyError) as e:
    print(f"UNHEALTHY: no readable heartbeat ({e})")
    sys.exit(1)

if age > max_age:
    print(f"UNHEALTHY: heartbeat {age:.0f}s old (max {max_age:.0f}s)")
    sys.exit(1)
print(f"OK: age={age:.0f}s market={hb.get('market_status')} step={hb.get('step')} dry_run={hb.get('dry_run')}")
