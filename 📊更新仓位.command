#!/bin/bash
# 双击运行:更新行情与个股 → 六库合奏+状态层 → 生成 面板.html
cd "$(dirname "$0")" && bash scripts/生产/每日更新.sh
echo; echo "按任意键关闭…"; read -n1
