#!/bin/bash
# launchd entry point:  scripts/run.sh morning|evening|train
# Secrets (SEC_USER_AGENT, optional DANELFIN_API_KEY) arrive via the launchd
# plist environment — they are never stored in this repo.
set -u
ROOT="${GRAVITY_ROOT:-$HOME/gravity}"
cd "$ROOT" || exit 1
export GRAVITY_ROOT="$ROOT"
export PATH="/usr/bin:/bin:/usr/sbin:/sbin:$PATH"
MODE="${1:-morning}"

# Only run on NYSE trading days (train is allowed any day).
if [ "$MODE" != "train" ]; then
  ./.venv/bin/python -c "from gravity.util import now_et,is_trading_day;import sys;sys.exit(0 if is_trading_day(now_et().date()) else 3)" 2>/dev/null
  [ $? -eq 3 ] && { echo "$(date) not a trading day — skipping $MODE"; exit 0; }
fi

# One run at a time; clear a lock older than 3 hours (crashed run).
LOCK="$ROOT/data/run.lock"
if ! mkdir "$LOCK" 2>/dev/null; then
  if [ -n "$(find "$LOCK" -maxdepth 0 -mmin +180 2>/dev/null)" ]; then
    rmdir "$LOCK" && mkdir "$LOCK" || exit 1
  else
    echo "$(date) another GRAVITY run is in progress — skipping $MODE"; exit 0
  fi
fi
trap 'rmdir "$LOCK" 2>/dev/null' EXIT

echo "=== gravity $MODE $(date) ==="
./.venv/bin/python -m gravity.cli "$@"
rc=$?
echo "=== gravity $MODE done rc=$rc $(date) ==="
exit $rc
