#!/bin/bash
# Deploy X-engine from this machine to the VPS (rsync, no git on the server).
#   ./deploy/deploy.sh            # sync code + deps, restart the service if it is enabled
# Isolated from scanApp: own user (xengine), own dir (/opt/x-engine), own systemd unit, no Docker.
# NOTE: excludes are ANCHORED (/models, /logs) — an unanchored "models" would also drop the
# source package trade_bot/models/.
set -euo pipefail
HOST="${XENGINE_HOST:-root@169.58.39.39}"
cd "$(dirname "${BASH_SOURCE[0]}")/.."

rsync -rlt --delete \
  --exclude /.git --exclude /.venv --exclude /logs --exclude /models --exclude /.env \
  --exclude /tests --exclude __pycache__ --exclude '*.pyc' --exclude .pytest_cache --exclude .DS_Store \
  ./ "$HOST:/opt/x-engine/"

ssh "$HOST" 'set -e
chown -R xengine:xengine /opt/x-engine
cd /opt/x-engine
[ -d .venv ] || sudo -u xengine python3 -m venv .venv
sudo -u xengine .venv/bin/pip install -q -r requirements.txt
sudo -u xengine .venv/bin/python -c "import trade_bot.agents, trade_bot.execution, mcp_server.server" >/dev/null
cp deploy/x-engine.service deploy/x-engine-health.service deploy/x-engine-health.timer /etc/systemd/system/
systemctl daemon-reload
if systemctl is-enabled --quiet x-engine 2>/dev/null; then systemctl restart x-engine && echo "service restarted"; else echo "service not enabled yet (run set-ig-credentials.sh, then systemctl enable --now x-engine)"; fi'
