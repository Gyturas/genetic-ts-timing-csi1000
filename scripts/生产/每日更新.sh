#!/bin/bash
# 中证1000 v5.2 每日更新核心脚本(.command 与 launchd 共用)
# 用法: 每日更新.sh [--git]   --git 时在成功后自动提交数据与结果
set -o pipefail
export PATH="/usr/bin:/bin:/usr/sbin:/sbin:/usr/local/bin:$PATH"
export MPLBACKEND=Agg
PY="/Library/Frameworks/Python.framework/Versions/3.11/bin/python3"
REPO="$HOME/Desktop/中证1000择时"
cd "$REPO" || { echo "❌ 找不到仓库 $REPO"; exit 1; }
[ -x "$PY" ] || { echo "❌ Python 不在 $PY"; exit 1; }

echo "════════ 中证1000 v5.2 · $(date '+%Y-%m-%d %H:%M') ════════"
PYTHONPATH=src "$PY" -m csi1000.engine.daily_dashboard --k 1.2
CODE=$?
[ $CODE -ne 0 ] && { echo "❌ 更新失败(退出码 $CODE)"; exit $CODE; }

if [ "$1" = "--git" ]; then
  echo; echo "── 自动提交数据与结果 ──"
  # 只提交数据/结果,不碰代码(避免把未完成的改动带上去)
  git add data results 2>/dev/null
  if git diff --cached --quiet; then
    echo "  无数据变更,跳过提交"
  else
    SIG=$(awk -F'\t' '/^信号日/{print $2}' 今日仓位.txt)
    POS=$(awk -F'\t' '/^目标仓位/{print $2}' 今日仓位.txt)
    git -c user.name="Gyturas" -c user.email="gytura.zyl@gmail.com" \
        commit -q -m "每日更新 $SIG:次日开盘目标仓位 ${POS}%"
    if git push -q origin main 2>/dev/null; then echo "  ✅ 已提交并推送"
    else echo "  ⚠️ 已本地提交,推送失败(网络?),下次会一并推送"; fi
  fi
fi
echo; echo "✅ 完成 $(date '+%H:%M')"
