#!/bin/sh
# One-off background job managed by launchd (survives the Claude session).
# usage: bg.sh <name> <logfile> <command...>   — plist lives in data/jobs (not LaunchAgents: never auto-starts at login)
set -e
name="$1"; logf="$2"; shift 2
label="com.gravity.job.$name"
dir="$HOME/gravity/data/jobs"; mkdir -p "$dir"
plist="$dir/$label.plist"
launchctl bootout "gui/$(id -u)/$label" 2>/dev/null || true
args=""
for a in /usr/bin/caffeinate -i "$@"; do args="$args<string>$a</string>"; done
cat > "$plist" <<P
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
<key>Label</key><string>$label</string>
<key>ProgramArguments</key><array>$args</array>
<key>WorkingDirectory</key><string>$HOME/gravity</string>
<key>StandardOutPath</key><string>$logf</string>
<key>StandardErrorPath</key><string>$logf</string>
<key>RunAtLoad</key><true/>
<key>KeepAlive</key><false/>
<key>ProcessType</key><string>Interactive</string>
</dict></plist>
P
launchctl bootstrap "gui/$(id -u)" "$plist"
echo "started $label"
