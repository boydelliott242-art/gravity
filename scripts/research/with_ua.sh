#!/bin/sh
# Exports SEC_USER_AGENT from the existing LaunchAgent plist at run time (never stored in this repo), then runs "$@".
SEC_USER_AGENT="$(/usr/libexec/PlistBuddy -c 'Print :EnvironmentVariables:SEC_USER_AGENT' "$HOME/Library/LaunchAgents/com.convexity.serve.plist" 2>/dev/null)"
export SEC_USER_AGENT
exec "$@"
