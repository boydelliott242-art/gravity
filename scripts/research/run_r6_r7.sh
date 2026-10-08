#!/bin/sh
cd "$HOME/gravity" || exit 1
./.venv/bin/python scripts/research/size_tier_r6.py > data/logs/size_tier_r6.log 2>&1
./.venv/bin/python scripts/research/size_tier_r7.py > data/logs/size_tier_r7.log 2>&1
echo "R6R7 DONE"
