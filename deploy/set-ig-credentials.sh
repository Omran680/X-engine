#!/bin/bash
# Run ON THE SERVER as root:  /opt/x-engine/deploy/set-ig-credentials.sh
# Prompts for the IG credentials (hidden input) and writes /opt/x-engine/.env (mode 600, owner xengine).
# Nothing is echoed, logged or passed on a command line.
set -euo pipefail
ENV_FILE=/opt/x-engine/.env
read -r  -p  "IG username: "        IG_USERNAME
read -r  -s -p "IG password: "      IG_PASSWORD; echo
read -r  -s -p "IG API key: "       IG_API_KEY;  echo
read -r  -s -p "Groq API key (optional, Enter to skip): " GROQ_API_KEY; echo
umask 077
{
  echo "IG_USERNAME=$IG_USERNAME"
  echo "IG_PASSWORD=$IG_PASSWORD"
  echo "IG_API_KEY=$IG_API_KEY"
  echo "IG_ACC_TYPE=DEMO"          # switching to LIVE is a deliberate manual edit (+ IG_ALLOW_LIVE=1)
  [ -n "$GROQ_API_KEY" ] && echo "GROQ_API_KEY=$GROQ_API_KEY"
} > "$ENV_FILE"
chown xengine:xengine "$ENV_FILE"; chmod 600 "$ENV_FILE"
echo "Wrote $ENV_FILE (DEMO). Start with: systemctl restart x-engine && journalctl -u x-engine -f"
