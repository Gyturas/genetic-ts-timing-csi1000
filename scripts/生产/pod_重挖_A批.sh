#!/bin/bash
# RunPod:A 批审计修复后的全量重挖(2026-08-05)。
#
# 背景:index_engine 修了 10 条治理层缺陷(A1~A10 + D1),名册构成会变,必须整体重挖。
#      详见 docs/因子库缺陷审计.md 的「A 批代码修复完成记录」。
#
# 前置(pod 上):
#   git clone <repo> && cd 中证1000择时
#   pip install pandas numpy scipy pyarrow
#   tmux new -s mine            ← 必须!nohup/setsid 在 pod 上会被连带回收
#   bash scripts/生产/pod_重挖_A批.sh 4
#
# 并行度:默认 4。容器 cgroup 内存上限实测 64G(而 free 显示 755G),
#        8 并行会 OOM。按 cgroup 实际上限调,不要按 free 调。
#   cat /sys/fs/cgroup/memory.max   # 看真实上限
#
# 断点续跑:已有 名册.csv 的库自动跳过,可随时 Ctrl-C 后重跑本脚本。
set -u
cd "$(dirname "$0")/.."/..
PAR=${1:-4}
END=${END:-2026-08-04}
PY=${PY:-python3}
mkdir -p logs/重挖

# 按【决策价值】排序:先出生产六库,再出 S4 的决胜两库,最后是横向表与 v4.1 基线。
# 中途没时间了也能保住最要紧的部分。
# 注意两个入口的目录构造【不一样】,别照抄 --tag:
#   mine_open_horizons → results/mine_open{tag}/h{h}        seed 不在叶子名里,换种子【要】 --tag
#   mine_xsec          → results/mine_xsec{tag}/h{h}_s{seed} seed 已在叶子名里,换种子【不要】 --tag
#        DIR                          MOD                   额外参数
JOBS=(
  "results/mine_open/h1|mine_open_horizons|--h 1"
  "results/mine_open/h5|mine_open_horizons|--h 5"
  "results/mine_xsec/h1_s42|mine_xsec|--h 1 --seed 42"
  "results/mine_xsec/h5_s42|mine_xsec|--h 5 --seed 42"
  "results/mine_xsec/h1_s137|mine_xsec|--h 1 --seed 137"
  "results/mine_xsec/h5_s137|mine_xsec|--h 5 --seed 137"
  "results/mine_open_s137/h1|mine_open_horizons|--h 1 --seed 137 --tag _s137"
  "results/mine_open_s137/h5|mine_open_horizons|--h 5 --seed 137 --tag _s137"
  "results/mine_open/h2|mine_open_horizons|--h 2"
  "results/mine_open/h3|mine_open_horizons|--h 3"
  "results/mine_open/h4|mine_open_horizons|--h 4"
  "results/mine_open/h6|mine_open_horizons|--h 6"
  "results/mine_open/h7|mine_open_horizons|--h 7"
  "results/ops26|mine_ops26|"
)

echo "═══ A 批重挖:${#JOBS[@]} 库,并行 $PAR,END=$END ═══"
echo "    重挖前先把旧产物挪走(不删,便于新旧对账)"
for t in "${JOBS[@]}"; do
  d="${t%%|*}"
  if [ -f "$d/名册.csv" ] && [ ! -f "$d/.A批已重挖" ]; then
    mkdir -p "旧库_A批前/$(dirname "$d")"
    cp -r "$d" "旧库_A批前/$d" 2>/dev/null || true
    rm -f "$d"/state.pkl "$d"/名册.csv "$d"/事件.csv "$d"/因子逐日信号.csv.gz
  fi
done

for t in "${JOBS[@]}"; do
  IFS='|' read -r DIR MOD ARGS <<< "$t"
  NAME=$(echo "$DIR" | tr '/' '_')
  if [ -f "$DIR/名册.csv" ]; then echo "  跳过(已完成) $DIR"; continue; fi
  echo "  启动 $DIR  → logs/重挖/$NAME.log"
  # shellcheck disable=SC2086
  PYTHONPATH=src nohup $PY -m "csi1000.engine.$MOD" $ARGS --end "$END" \
      > "logs/重挖/$NAME.log" 2>&1 &
  while [ "$(jobs -rp | wc -l)" -ge "$PAR" ]; do sleep 30; done
done
wait
echo "═══ 挖矿完成,开始验收 ═══"

PYTHONPATH=src $PY - <<'PYEOF'
# 验收判据在 docs/因子库缺陷审计.md 写死过,此处逐条自动核对。
import os, pickle, glob, collections, pandas as pd
库 = sorted(glob.glob("results/mine_open/h*") + glob.glob("results/mine_open_s137/h*")
            + glob.glob("results/mine_xsec/*") + ["results/ops26"])
print(f"\n{'库':28s} {'GA':>5s} {'类库':>4s} {'2015Q2后类库入库':>15s} {'元老/类库复职':>12s} {'重复拦截':>7s}")
坏 = []
for d in 库:
    p = os.path.join(d, "state.pkl")
    if not os.path.exists(p):
        print(f"{d:28s} —— 无产物"); 坏.append(f"{d}:无产物"); continue
    st = pickle.load(open(p, "rb"))
    ga  = [m for m in st["members"].values() if m.get("kind") == "ga"]
    lei = [m for m in ga if m.get("专属") is None]
    后 = sum(1 for m in lei if m["admit_q"] > "2015Q1")          # 判据①
    eld = [m for m in st["members"].values() if m.get("kind") in ("elder", "argarch")]
    复 = sum(1 for m in lei + eld if m.get("reenlist_used"))      # 判据②
    ev = pd.DataFrame(st["events"], columns=["季","事件","编号","表达式","kind","专属","明细"]) \
           if st["events"] and len(st["events"][0]) == 7 else None
    连不上 = "—"
    if ev is not None:                                            # 判据③
        入 = set(ev[ev.事件 == "入库"].编号)
        退 = set(ev[ev.事件.str.contains("退役")].编号)
        连不上 = f"{len(退 - 入)}"
    print(f"{d:28s} {len(ga):5d} {len(lei):4d} {后:15d} {复:12d} {连不上:>7s}")
    if 后 == 0: 坏.append(f"{d}: 判据① 2015Q2 后仍零类库入库 —— A1 没修对")
    if 复 == 0: 坏.append(f"{d}: 判据② 元老/类库零复职 —— A2 没修对")
    if 连不上 not in ("—", "0"): 坏.append(f"{d}: 判据③ {连不上} 个退役事件连不上入库")
print()
if 坏:
    print("✗ 验收未通过:"); [print("   ", x) for x in 坏]
else:
    print("✓ 全部验收判据通过")
PYEOF

echo "═══ 横向评估 ═══"
PYTHONPATH=src $PY -m csi1000.engine.eval_open_horizons --root mine_open      | tee logs/重挖/eval_open.log
PYTHONPATH=src $PY -m csi1000.engine.eval_open_horizons --root mine_open_s137 | tee logs/重挖/eval_open_s137.log
PYTHONPATH=src $PY -m csi1000.engine.eval_open_horizons --root mine_xsec --xsec | tee logs/重挖/eval_xsec.log

tar czf 重挖_A批.tgz results/mine_open results/mine_open_s137 results/mine_xsec results/ops26 logs/重挖
echo "═══ 完成:重挖_A批.tgz —— 拉回本地后跑 live_v51 与 S4 裁决 ═══"
