# -*- coding: utf-8 -*-
"""四档权重方案一次跑完并出对照表。挖矿完成后跑这一个脚本即可。

  正典     筛除(IC≤0剔除) + IC加权     ← 现在线上用的
  截正等权 筛除            + 等权       ← 只改加权那一层
  等权     不筛除          + 等权       ← 再把IC≤0的因子放回来
  纯等权   不筛除          + 等权 + 黄牌不减半

先做重放自检:正典重算必须与挖矿当场逐日相同(差<1e-12),不然全部结论作废。

用法: python -m 引擎.权重对照
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile

import pandas as pd

根 = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
方案 = ["正典", "截正等权", "等权", "纯等权"]


def 跑(*args, cwd=根):
    return subprocess.run([sys.executable, *args], cwd=cwd,
                          env=dict(os.environ, PYTHONPATH=根),
                          capture_output=True, text=True)


def main():
    存档 = os.path.join(根, "结果_挖矿", "因子逐日信号.csv.gz")
    assert os.path.exists(存档), "缺存档,请先跑 python -m 引擎.指数日频引擎"
    d = pd.read_csv(存档, parse_dates=["date"], usecols=["date"])
    print(f"存档 {len(d):,} 行  {d.date.min():%Y-%m-%d} ~ {d.date.max():%Y-%m-%d}")
    assert str(d.date.max().date()) >= "2026-07-01", "存档没跑到今天,是截断的"

    for s in 方案:
        r = 跑("-m", "引擎.重算权重", "--方案", s, "--输出", f"结果_{s}")
        assert r.returncode == 0, f"{s} 失败:\n{r.stderr[-800:]}"
        print(f"  {s:5s} {[l for l in r.stdout.splitlines() if '日均' in l][0]}")

    # ---- 自检:正典重放必须逐日复现引擎 ----
    print("\n重放自检(正典 vs 挖矿当场):")
    最大差 = 0.0
    for f in ("资产暴露.csv", "组合逐日.csv", "分资产逐日.csv"):
        x = pd.read_csv(os.path.join(根, "结果_挖矿", f), index_col=0, parse_dates=True)
        y = pd.read_csv(os.path.join(根, "结果_正典", f), index_col=0, parse_dates=True)
        i = x.index.intersection(y.index)
        最大差 = max(最大差, float((x.loc[i] - y.loc[i, x.columns]).abs().max().max()))
    print(f"  最大差 {最大差:.1e}  {'✓ 通过' if 最大差 < 1e-12 else '✗ 不通过 —— 以下结论全部作废'}")
    if 最大差 >= 1e-12:
        sys.exit(1)

    # ---- 各方案回测:把信号塞进仓库副本,用正典回测脚本算 ----
    print("\n回测:")
    行 = []
    for s in 方案:
        tmp = tempfile.mkdtemp(prefix=f"回测_{s}_")
        仓 = os.path.join(tmp, "repo")
        shutil.copytree(根, 仓, ignore=shutil.ignore_patterns(
            ".git", "__pycache__", "结果_*"))
        shutil.copy(os.path.join(根, f"结果_{s}", "分资产逐日.csv"),
                    os.path.join(仓, "数据", "分资产逐日.csv"))
        r = subprocess.run([sys.executable, "-m", "引擎.策略"], cwd=仓,
                           env=dict(os.environ, PYTHONPATH=仓),
                           capture_output=True, text=True)
        assert r.returncode == 0, f"{s} 回测失败:\n{r.stderr[-800:]}"
        open(os.path.join(根, f"结果_{s}", "回测.txt"), "w", encoding="utf-8").write(r.stdout)
        成绩 = pd.read_csv(os.path.join(仓, "结果", "成绩表.csv"), index_col=0)
        shutil.copy(os.path.join(仓, "结果", "成绩表.csv"),
                    os.path.join(根, f"结果_{s}", "成绩表.csv"))
        for 口径 in ("熊增强·多空(定稿)", "熊增强·纯多(禁空)", "仅量价·多空"):
            行.append({"方案": s, "口径": 口径,
                       **成绩.loc[口径, ["年化", "夏普", "最大回撤", "卡玛", "均|仓|"]].to_dict()})
        print(f"  {s} 完成")
        shutil.rmtree(tmp, ignore_errors=True)

    表 = pd.DataFrame(行)
    print("\n" + "=" * 78)
    for 口径 in 表["口径"].unique():
        print(f"【{口径}】")
        print(表[表["口径"] == 口径].drop(columns=["口径"]).to_string(index=False))
        print()
    print("=" * 78)
    表.to_csv(os.path.join(根, "结果", "权重方案对照.csv"), index=False)


if __name__ == "__main__":
    main()
