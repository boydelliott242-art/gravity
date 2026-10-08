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

notify() {  # visible failure: a macOS notification + the log line
  echo "$(date) $1"
  /usr/bin/osascript -e "display notification \"$1\" with title \"GRAVITY\"" >/dev/null 2>&1 || true
}

# A job launchd fires late (the Mac was asleep at its slot) must not run in
# the wrong window: an evening build fired the next morning would overwrite
# the morning's #1, and a morning build after the bell would be late anyway.
# GRAVITY_FORCE=1 overrides (manual runs).
if [ "${GRAVITY_FORCE:-0}" != "1" ] && { [ "$MODE" = "evening" ] || [ "$MODE" = "morning" ]; }; then
  $PY -c "from gravity.util import now_et;from datetime import time as t;import sys;n=now_et().time();m=sys.argv[1];sys.exit(3 if (m=='evening' and n<t(16,0)) or (m=='morning' and n>=t(9,30)) else 0)" "$MODE" 2>/dev/null
  [ $? -eq 3 ] && { echo "$(date) $MODE fired outside its window (Mac asleep at the scheduled time?) — skipping"; exit 0; }
fi

# One run at a time. The lock holds "PID MODE"; it is stale when that PID is
# dead or is no longer a GRAVITY run (PIDs get reused after a reboot).
# Morning/evening WAIT (up to 20 min) — and stop a weekly retrain that is
# still running, because the daily pick matters more (the model file is
# replaced atomically, so the old model stays in place).
LOCK="$ROOT/data/run.lock"
owner_alive() {
  local pid="$1"
  [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null && ps -p "$pid" -o command= 2>/dev/null | grep -q "run.sh"
}
acquire() {
  if mkdir "$LOCK" 2>/dev/null; then echo "$$ $MODE" > "$LOCK/pid"; return 0; fi
  local owner omode; read -r owner omode < "$LOCK/pid" 2>/dev/null || owner=""
  if ! owner_alive "$owner"; then
    rm -rf "$LOCK" && mkdir "$LOCK" 2>/dev/null && echo "$$ $MODE" > "$LOCK/pid" && return 0
  fi
  if [ "${omode:-}" = "train" ] && { [ "$MODE" = "morning" ] || [ "$MODE" = "evening" ]; }; then
    local pg; pg=$(ps -o pgid= -p "$owner" 2>/dev/null | tr -d ' ')
    notify "stopping a retrain that was still running so the $MODE run can publish"
    [ -n "$pg" ] && kill -TERM -- "-$pg" 2>/dev/null || kill -TERM "$owner" 2>/dev/null
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
    [ "$MODE" = "live" ] || notify "another GRAVITY run held the lock — gave up on $MODE"
    exit 0
  fi
  sleep 30; waited=$((waited + 30))
done
trap '[ "$(cut -d" " -f1 "$LOCK/pid" 2>/dev/null)" = "$$" ] && rm -rf "$LOCK"' EXIT

echo "=== gravity $MODE $(date) ==="
# caffeinate keeps the Mac awake (idle + system sleep) for the whole run.
/usr/bin/caffeinate -i -s $PY -m gravity.cli "$@"
rc=$?
echo "=== gravity $MODE done rc=$rc $(date) ==="
[ $rc -ne 0 ] && [ "$MODE" != "live" ] && notify "$MODE run failed (rc=$rc) — see data/logs"
exit $rc
