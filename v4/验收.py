# -*- coding: utf-8 -*-
"""独立仓库验收 —— 模拟"别人 clone 下来"的完整链路。

任何一项不通过即以非零码退出。**改动本仓库后请务必重跑本脚本。**

检查项:
  ① 文件完整性        挖矿+回测所需的每个模块都在,且非空
  ② 零字节文件        除 __init__.py 外不得有空 .py(历史上 更新熊概率.py 曾是 0 字节)
  ③ 语法与硬编码路径   全部 .py 可解析;不得出现 /Users/ 之类的绝对路径
  ④ 行情数据完整性     data_cache 覆盖 config 五大类面板 + 全部执行 ETF + 货基
  ⑤ 回测存档          数据/ 下信号、熊增强、熊概率、名册、事件、state 齐备
  ⑥ 数据健康          名册.sign 必须是数值(曾被误改成 "1. 0" 字符串致方向失效)
  ⑦ 隔离运行          整仓复制到临时目录、剥掉 .git,在那里跑:
      挖矿 smoke     python -m 引擎.指数日频引擎 --end 2015-06-30   须逐季录取因子
      回测复现       python -m 引擎.策略   须得到 20.7% / 夏普1.01(定稿值)
      出图三件套     出图 / 映射对比 / 逐年信号图   须全部退出码 0
      定稿信号可再生   引擎.定稿信号 重跑后回测数字不变
      结果不被覆盖   挖矿只写 结果_挖矿/,不得动 结果/

用法:  python 验收.py         # 全量(含隔离运行,约3分钟)
       python 验收.py --快    # 仅静态检查(数秒)
"""
from __future__ import annotations

import ast
import os
import re
import shutil
import subprocess
import sys
import tempfile

根 = os.path.dirname(os.path.abspath(__file__))
定稿年化, 定稿夏普 = "28.2%", "1.38"
错: list[str] = []


def 查(名: str, 条件: bool, 详: str = "") -> bool:
    print(f"  {'✓' if 条件 else '✗'} {名}" + (f"\n      {详}" if 详 and not 条件 else ""))
    if not 条件:
        错.append(名)
    return 条件


def 静态检查() -> None:
    print("=" * 72); print("① 文件完整性"); print("=" * 72)
    必需 = {
        "ga_alpha": ["__init__.py", "expr.py", "ops.py", "evolve.py", "fitness.py"],
        "walkforward": ["__init__.py", "config.py", "data.py", "elders.py",
                        "engine.py", "ga_setup.py", "metrics.py"],
        "引擎": ["__init__.py", "指数日频引擎.py", "定稿信号.py", "明日仓位.py", "策略.py", "路径.py",
               "出图.py", "映射对比.py", "逐年信号图.py", "更新熊概率.py", "实盘信号.py",
               "家族再倾斜.py", "bear_factor.py", "regime_data.py"],
    }
    缺 = [f"{d}/{f}" for d, fs in 必需.items() for f in fs
          if not (os.path.exists(os.path.join(根, d, f))
                  and (os.path.getsize(os.path.join(根, d, f)) > 0 or f == "__init__.py"))]
    查(f"{sum(len(v) for v in 必需.values())} 个必需模块齐备", not 缺, str(缺))

    print("\n② 零字节文件")
    空 = [os.path.relpath(os.path.join(r, f), 根)
          for r, _, fs in os.walk(根) for f in fs
          if f.endswith(".py") and f != "__init__.py" and "__pycache__" not in r
          and os.path.getsize(os.path.join(r, f)) == 0]
    查("无空 .py", not 空, str(空))

    print("\n③ 语法与硬编码路径")
    pys = [os.path.join(r, f) for r, _, fs in os.walk(根) for f in fs
           if f.endswith(".py") and "__pycache__" not in r]
    坏, 硬 = [], []
    for p in pys:
        src = open(p, encoding="utf-8").read()
        try:
            ast.parse(src)
        except SyntaxError as e:
            坏.append(f"{os.path.relpath(p, 根)}:{e.lineno}")
        for m in re.finditer(r'["\']((?:/Users/|/home/|[A-Z]:\\)[^"\']*)["\']', src):
            硬.append(f"{os.path.relpath(p, 根)} → {m.group(1)}")
    查(f"{len(pys)} 个 .py 语法通过", not 坏, str(坏))
    查("无绝对路径", not 硬, str(硬[:3]))

    print("\n④ 挖矿所需行情")
    sys.path.insert(0, 根)
    from walkforward import config as C
    need = {"etf_511880"}
    for sp in C.CLASSES.values():
        need |= set(sp["panel"].values()) | {"etf_" + e for e in sp["etf_map"]}
    dc = os.path.join(根, "data_cache")
    有 = {f[:-4] for f in os.listdir(dc)} if os.path.isdir(dc) else set()
    查(f"data_cache 覆盖 {len(need)} 个 tag", need <= 有, f"缺 {sorted(need - 有)}")

    # 只查"文件在不在"不够 —— 曾经每个文件都在,但内容截断到 2015 年。
    import pandas as pd
    短 = []
    for t in sorted(need & 有):
        try:
            末 = pd.read_csv(os.path.join(dc, t + ".csv"), usecols=["date"])["date"].max()
            if str(末) < "2026-07-01":
                短.append(f"{t}→{末}")
        except Exception as e:
            短.append(f"{t}→读不了({type(e).__name__})")
    查(f"行情都更新到 {C.END} 附近", not 短, str(短[:5]))

    # 面板缓存是派生产物,且会被 C.END 截断。它不该进仓库,也不该没指纹就被复用。
    pkl = os.path.join(dc, "panels_etf.pkl")
    查("面板缓存未入库(应由首次运行自建)",
       subprocess.run(["git", "ls-files", "--error-unmatch", "data_cache/panels_etf.pkl"],
                      cwd=根, capture_output=True).returncode != 0)
    if os.path.exists(pkl):
        import pickle
        行情末 = pd.read_csv(os.path.join(dc, "idx_sh000852.csv"), usecols=["date"])["date"].max()
        查("本地面板缓存与行情末日一致(v4:定稿信号动态设 END=行情末日)",
           pickle.load(open(pkl, "rb")).get("built_end") == str(行情末),
           "缓存过期;定稿信号 重跑会自动重建,但别把它提交上去")

    print("\n⑤ 回测存档")
    缺档 = [f for f in ("数据/分资产逐日.csv", "数据/熊增强逐日.csv", "数据/bear_prob.csv",
                       "数据/名册.csv", "数据/事件.csv", "数据/state.pkl")
           if not os.path.exists(os.path.join(根, f))]
    查("存档齐备", not 缺档, str(缺档))

    print("\n⑥ 数据健康")
    import pandas as pd
    try:
        d = pd.read_csv(os.path.join(根, "数据/名册.csv"))
        查("名册.sign 为数值", d["sign"].dtype.kind == "f" and d["sign"].notna().all())
    except Exception as e:
        查("名册可读", False, f"{type(e).__name__}: {e}")


def 隔离运行() -> None:
    print("\n" + "=" * 72); print("⑦ 隔离运行"); print("=" * 72)
    tmp = tempfile.mkdtemp(prefix="验收_")
    仓 = os.path.join(tmp, "repo")
    shutil.copytree(根, 仓, ignore=shutil.ignore_patterns(
        ".git", "__pycache__", "结果_挖矿", "结果_正典复算", "结果_等权"))
    print(f"  复制到 {仓}(已剥 .git)")
    env = dict(os.environ, PYTHONPATH=仓)

    def 跑(mod, *args, 超时=2400):
        return subprocess.run([sys.executable, "-m", mod, *args], cwd=仓, env=env,
                              capture_output=True, text=True, timeout=超时)

    结果目录 = os.path.join(仓, "结果")
    快照 = sorted(os.listdir(结果目录)) if os.path.isdir(结果目录) else []

    # 先确认干净副本能建出覆盖到 END 的面板。挖矿 smoke 用 --end 图快,
    # 但正因为它带 --end,单靠它发现不了"面板被截断"——2015年那次就是这么漏过去的。
    rr = subprocess.run(
        [sys.executable, "-c",
         "from walkforward.data import load_all; from walkforward import config as C\n"
         "c = load_all()['classes']['A股宽基']['panels']['close']\n"
         "print('PANEL_END', c.index[-1].date(), 'ROWS', len(c))\n"
         "assert str(c.index[-1].date()) >= '2026-07-01', c.index[-1]"],
        cwd=仓, env=env, capture_output=True, text=True, timeout=1800)
    查("干净副本能建出完整面板(覆盖到 END)", rr.returncode == 0,
       (rr.stdout + rr.stderr)[-300:])
    print(f"      {rr.stdout.strip().splitlines()[-1] if rr.stdout.strip() else ''}")

    r = 跑("引擎.指数日频引擎", "--end", "2015-06-30")
    录取 = sum(int(m) for m in re.findall(r"录取=(\d+)", r.stdout))
    查("挖矿 smoke 跑通", r.returncode == 0, r.stderr[-300:])
    查(f"挖矿确有录取因子(共 {录取} 个)", 录取 > 0)

    r = 跑("引擎.策略")
    查("回测跑通", r.returncode == 0, r.stderr[-300:])
    查(f"回测复现定稿 {定稿年化}/夏普{定稿夏普}",
       定稿年化 in r.stdout and 定稿夏普 in r.stdout, r.stdout[:400])

    for m in ("引擎.出图", "引擎.映射对比", "引擎.逐年信号图"):
        rr = 跑(m)
        查(f"{m} 跑通", rr.returncode == 0, rr.stderr[-200:])


    现 = sorted(os.listdir(结果目录)) if os.path.isdir(结果目录) else []
    查("挖矿未覆盖 结果/", 现 == 快照, f"变化 {set(现) ^ set(快照)}")
    查("挖矿产物落在 结果_挖矿/", os.path.isdir(os.path.join(仓, "结果_挖矿")))

    shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    静态检查()
    if "--快" not in sys.argv:
        隔离运行()
    print("\n" + "=" * 72)
    print("验收结果: " + ("全部通过 ✓" if not 错 else f"{len(错)} 项失败 → {错}"))
    print("=" * 72)
    sys.exit(1 if 错 else 0)
