#!/bin/bash
# RunPod 一键:开盘口径挖矿 h=1..7 并行 + 横向评估 + 打包。
# 前置: git clone 本仓库(数据已随仓库带全) && pip install pandas numpy scipy
# 用法: bash scripts/pod_mine_open.sh [并行数,默认7]
set -e
cd "$(dirname "$0")/.."
PAR=${1:-7}
mkdir -p results/mine_open/logs

for h in 1 2 3 4 5 6 7; do
  if [ -f "results/mine_open/h$h/名册.csv" ]; then
    echo "h=$h 已完成,跳过"
    continue
  fi
  echo "启动 h=$h(日志 results/mine_open/logs/h$h.log)"
  PYTHONPATH=src nohup python3 -m csi1000.engine.mine_open_horizons --h $h --end 2026-07-24 \
      > "results/mine_open/logs/h$h.log" 2>&1 &
  while [ "$(jobs -rp | wc -l)" -ge "$PAR" ]; do sleep 30; done
done
wait
echo "═══ 全部挖矿完成,开始横向评估 ═══"
PYTHONPATH=src python3 -m csi1000.engine.eval_open_horizons | tee results/mine_open/eval.log
tar czf mine_open_results.tgz results/mine_open
echo "═══ 完成:mine_open_results.tgz(拉回本地即可)═══"
