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
PY="./.venv/bin/python"

# Only run morning/evening on NYSE trading days; train only when no trading
# session is near (weekends/holidays, or outside 06:00–16:30 ET).
if [ "$MODE" = "train" ]; then
  $PY -c "from gravity.util import now_et,is_trading_day;from datetime import time as t;n=now_et();import sys;sys.exit(3 if is_trading_day(n.date()) and t(6,0)<=n.time()<=t(16,30) else 0)" 2>/dev/null
  [ $? -eq 3 ] && { echo "$(date) trading hours — skipping train"; exit 0; }
else
  $PY -c "from gravity.util import now_et,is_trading_day;import sys;sys.exit(0 if is_trading_day(now_et().date()) else 3)" 2>/dev/null
  [ $? -eq 3 ] && { echo "$(date) not a trading day — skipping $MODE"; exit 0; }
fi

# One run at a time. The lock holds the owner's PID; a lock whose PID is
# dead is stale. Morning/evening WAIT (up to 20 min) rather than skip.
LOCK="$ROOT/data/run.lock"
acquire() {
  if mkdir "$LOCK" 2>/dev/null; then echo $$ > "$LOCK/pid"; return 0; fi
  local owner; owner=$(cat "$LOCK/pid" 2>/dev/null || echo "")
  if [ -z "$owner" ] || ! kill -0 "$owner" 2>/dev/null; then
    rm -rf "$LOCK" && mkdir "$LOCK" 2>/dev/null && echo $$ > "$LOCK/pid" && return 0
  fi
  return 1
}
# live: cheap exit outside the regular session, and never queue behind a big run
if [ "$MODE" = "live" ]; then
  $PY -c "from gravity.util import market_phase;import sys;sys.exit(0 if market_phase()=='open' else 3)" 2>/dev/null
  [ $? -eq 3 ] && exit 0
fi
waited=0
until acquire; do
  if [ "$MODE" = "train" ] || [ "$MODE" = "live" ] || [ $waited -ge 1200 ]; then
    echo "$(date) another GRAVITY run holds the lock — giving up on $MODE"; exit 0
  fi
  sleep 30; waited=$((waited + 30))
done
trap '[ "$(cat "$LOCK/pid" 2>/dev/null)" = "$$" ] && rm -rf "$LOCK"' EXIT

echo "=== gravity $MODE $(date) ==="
# caffeinate keeps the Mac awake (idle + system sleep) for the whole run.
/usr/bin/caffeinate -i -s $PY -m gravity.cli "$@"
rc=$?
echo "=== gravity $MODE done rc=$rc $(date) ==="
exit $rc
