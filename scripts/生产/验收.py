# -*- coding: utf-8 -*-
"""生产自检(v5.2 口径,2026-08-05 重写——旧版是仓库重构前的遗物,import 都已失效)。

检查项:
  ① 生产库三件套齐备(live_v51.六库 里的每个库)
  ② 截面叶子新鲜度(落后行情即 live_v51 会抛错,提前在这里看到)
  ③ v51_逐日.csv 末日 = 行情末日(昨晚的更新没有静默失败)
  ④ launchd 两个 plist 在位
用法: PYTHONPATH=src python3 scripts/生产/验收.py
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
import pandas as pd
from csi1000 import paths
from csi1000.engine.live_v51 import 六库

坏 = []
def 查(名, ok, 详=""):
    print(f"  {'✓' if ok else '✗'} {名}" + (f"  {详}" if 详 else ""))
    if not ok: 坏.append(名)

print("① 生产库三件套")
for 目录, h, _ in 六库:
    全 = all(os.path.exists(os.path.join(paths.根, 目录, f))
             for f in ("state.pkl", "因子逐日信号.csv.gz", "名册.csv"))
    查(f"{目录}(h={h})", 全)

print("② 数据新鲜度")
行情末 = pd.read_csv(os.path.join(paths.行情缓存, "idx_sh000852.csv"),
                    usecols=["date"], parse_dates=["date"])["date"].max()
叶末 = pd.read_csv(os.path.join(paths.存档, "截面叶子.csv"),
                  usecols=["date"], parse_dates=["date"])["date"].max()
查("截面叶子 ≥ 行情末日", 叶末 >= 行情末, f"叶{叶末.date()} vs 行情{行情末.date()}")

print("③ 信号产出")
fp = os.path.join(paths.结果, "v51_逐日.csv")
if os.path.exists(fp):
    v = pd.read_csv(fp, index_col=0, parse_dates=True)   # 首列为无名日期索引
    查("v51_逐日 末日 = 行情末日", v.index.max() == 行情末,
       f"{v.index.max().date()} vs {行情末.date()}")
    查(f"库数 = {len(六库)}", int(v["库数"].iloc[-1]) == len(六库))
else:
    查("v51_逐日.csv 存在", False)

print("④ launchd")
for n in ("daily", "open"):
    查(f"com.gyturas.csi1000.{n}.plist",
       os.path.exists(os.path.expanduser(f"~/Library/LaunchAgents/com.gyturas.csi1000.{n}.plist")))

print()
if 坏:
    print(f"✗ {len(坏)} 项未过:", "、".join(坏)); sys.exit(1)
print("✓ 全部通过")
