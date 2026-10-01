#!/bin/bash
# Installs the three GRAVITY LaunchAgents (times are the Mac's local time,
# Central): morning 07:20 + 08:05 (pre-market), evening 17:10 (after FINRA
# posts the day's short volume), train Saturday 09:00. launchd runs a
# missed job as soon as the Mac wakes, and scorecard logic refuses to count
# any pick published after the 9:30 ET open.
set -eu
ROOT="$HOME/gravity"
LA="$HOME/Library/LaunchAgents"
mkdir -p "$LA" "$ROOT/data/logs"
# Reuse the SEC contact string the user already configured for Convexity.
SEC_UA="${SEC_USER_AGENT:-$(/usr/libexec/PlistBuddy -c 'Print :EnvironmentVariables:SEC_USER_AGENT' "$LA/com.convexity.serve.plist" 2>/dev/null || true)}"
DAN="${DANELFIN_API_KEY:-}"

cal() { # $1=hour $2=minute  → weekday Mon-Fri dict entries
  for d in 1 2 3 4 5; do
    printf '    <dict><key>Weekday</key><integer>%s</integer><key>Hour</key><integer>%s</integer><key>Minute</key><integer>%s</integer></dict>\n' "$d" "$1" "$2"
  done
}

write_plist() { # label mode calendar-xml
  cat > "$LA/$1.plist" <<PL
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key><string>$1</string>
  <key>ProgramArguments</key>
  <array><string>/bin/bash</string><string>$ROOT/scripts/run.sh</string><string>$2</string></array>
  <key>StartCalendarInterval</key>
  <array>
$3
  </array>
  <key>EnvironmentVariables</key>
  <dict>
    <key>GRAVITY_ROOT</key><string>$ROOT</string>
    <key>SEC_USER_AGENT</key><string>$SEC_UA</string>
    <key>DANELFIN_API_KEY</key><string>$DAN</string>
  </dict>
  <key>StandardOutPath</key><string>$ROOT/data/logs/$2.log</string>
  <key>StandardErrorPath</key><string>$ROOT/data/logs/$2.log</string>
  <key>ProcessType</key><string>Background</string>
</dict>
</plist>
PL
  launchctl unload "$LA/$1.plist" 2>/dev/null || true
  launchctl load "$LA/$1.plist"
  echo "installed $1"
}

write_plist com.gravity.morning morning "$(cal 7 20; cal 8 5)"
write_plist com.gravity.evening evening "$(cal 17 10)"
# live tape: every 15 minutes (run.sh exits immediately outside 9:30–16:00 ET)
write_interval() { # label mode seconds
  cat > "$LA/$1.plist" <<PL
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key><string>$1</string>
  <key>ProgramArguments</key>
  <array><string>/bin/bash</string><string>$ROOT/scripts/run.sh</string><string>$2</string></array>
  <key>StartInterval</key><integer>$3</integer>
  <key>EnvironmentVariables</key>
  <dict>
    <key>GRAVITY_ROOT</key><string>$ROOT</string>
    <key>SEC_USER_AGENT</key><string>$SEC_UA</string>
  </dict>
  <key>StandardOutPath</key><string>$ROOT/data/logs/$2.log</string>
  <key>StandardErrorPath</key><string>$ROOT/data/logs/$2.log</string>
  <key>ProcessType</key><string>Background</string>
</dict>
</plist>
PL
  launchctl unload "$LA/$1.plist" 2>/dev/null || true
  launchctl load "$LA/$1.plist"
  echo "installed $1"
}
write_interval com.gravity.live live 900
write_plist com.gravity.train train '    <dict><key>Weekday</key><integer>6</integer><key>Hour</key><integer>9</integer><key>Minute</key><integer>0</integer></dict>'
