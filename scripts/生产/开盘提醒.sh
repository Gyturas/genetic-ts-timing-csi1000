#!/bin/bash
# 开盘前提醒:读取昨晚算好的目标仓位,弹 macOS 通知(不重算,秒级)
REPO="$HOME/Desktop/中证1000择时"
F="$REPO/今日仓位.txt"
[ -f "$F" ] || exit 0
SIG=$(awk -F'\t' '/^信号日/{print $2}' "$F")
POS=$(awk -F'\t' '/^目标仓位/{print $2}' "$F")
DIR=$(awk -F'\t' '/^方向/{print $2}' "$F")
PANIC=$(awk -F'\t' '/^恐慌态/{print $2}' "$F")
EXTRA=""; [ "$PANIC" = "是" ] && EXTRA=" · 恐慌态"
osascript -e "display notification \"512100 目标仓位 ${POS}%(${DIR})${EXTRA}\" with title \"中证1000 择时 · 今日开盘执行\" subtitle \"信号日 ${SIG}\" sound name \"Glass\""
