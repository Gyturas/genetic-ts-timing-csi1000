# -*- coding: utf-8 -*-
"""换一种加权方案重算仓位 —— 不重挖因子。

因子库和权重方案是两回事:挖矿决定"有哪些因子、每个因子逐日给什么信号",
加权只决定"怎么把它们加起来"。所以换权重根本不需要重跑遗传算法。

指数日频引擎 跑一遍会把每一季、每个指数、每个在役因子的逐日信号连同正典权重
存进 因子逐日信号.csv.gz。本脚本读它,换个权重合成,再调用引擎自己的结算函数
(_结算,原封不动那一份)出 分资产逐日.csv —— 结算逻辑不重写,避免手抄出错。

方案:
  正典    w = max(过去12月该指数IC均值, 0) × (黄牌?0.5:1)   ← 存档里的权重
  等权    w = 1                            × (黄牌?0.5:1)
  纯等权  w = 1                                             ← 黄牌也不减半

用法:
  python -m 引擎.重算权重 --方案 等权 --存档 结果_挖矿 --输出 结果_等权
  python -m 引擎.重算权重 --方案 正典 --存档 结果_挖矿 --输出 结果_正典复算   # 自检:应与存档一致
"""
from __future__ import annotations

import argparse
import os

import pandas as pd

方案表 = {
    # 正典权重 max(IC,0) 其实同时做了两件事:①把 IC≤0 的因子踢出组合 ②在剩下的里按IC分配。
    # 拆成三档才分得清是哪一层在起作用(实测:12月尾均值的因子间差异有75%是估计噪音,
    # 所以②大概率该收缩掉;而①是真的筛除,约1/4的因子受影响)。
    "正典":     lambda d: d["权重"],                            # 筛除 + IC加权
    "截正等权": lambda d: (d["权重"] > 0) * (1.0 - 0.5 * d["黄牌"]),  # 筛除 + 等权
    "等权":     lambda d: 1.0 - 0.5 * d["黄牌"],                # 不筛除 + 等权(黄牌仍减半)
    "纯等权":   lambda d: 1.0,                                  # 不筛除 + 等权 + 黄牌不减半
}


def 建指数仓(存档: pd.DataFrame, 方案: str, 指数表: list[str]) -> pd.DataFrame:
    d = 存档.copy()
    d["w"] = 方案表[方案](d)
    d = d[d["w"] > 0]
    # 每(日,指数)内部把权重归一化,与引擎里的 w/总 一致
    总 = d.groupby(["date", "指数"])["w"].transform("sum")
    d["贡献"] = d["信号"] * d["w"] / 总
    仓 = d.groupby(["date", "指数"])["贡献"].sum().unstack("指数")
    # 烧机期(在役因子太少)整体减半,与引擎一致
    减半 = 存档.groupby(["date", "指数"])["烧机减半"].max().unstack("指数").reindex_like(仓)
    仓 = 仓 * (1.0 - 0.5 * 减半.fillna(0))
    return 仓.reindex(columns=指数表).fillna(0.0).clip(-1, 1).sort_index()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--方案", required=True, choices=list(方案表))
    ap.add_argument("--存档", default="结果_挖矿", help="含 因子逐日信号.csv.gz 与 state.pkl 的目录")
    ap.add_argument("--输出", required=True)
    a = ap.parse_args()

    根 = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    存档目录 = a.存档 if os.path.isabs(a.存档) else os.path.join(根, a.存档)
    输出 = a.输出 if os.path.isabs(a.输出) else os.path.join(根, a.输出)

    # 引擎按模块级 输出目录 读写,先指向存档目录把已挖好的库读进来,再切到输出目录
    import 引擎.指数日频引擎 as E
    E.输出目录 = 存档目录
    E.安装GA()
    eng = E.宽基引擎(E.load_all())
    assert eng.状态["done_q"], "存档里没有跑完的季度,请先跑 引擎.指数日频引擎"

    fp = os.path.join(存档目录, "因子逐日信号.csv.gz")
    assert os.path.exists(fp), f"缺 {fp}(该存档是旧版引擎产出的,请重跑一次挖矿)"
    存档 = pd.read_csv(fp, parse_dates=["date"])
    print(f"存档 {len(存档):,} 行,{存档['因子'].nunique()} 个因子,"
          f"{存档['date'].min():%Y-%m-%d}~{存档['date'].max():%Y-%m-%d}")

    仓 = 建指数仓(存档, a.方案, eng.指数表)
    print(f"方案={a.方案}  日均绝对仓位 {仓.abs().mean().mean():.3f}")

    # 清掉挖矿时累积的逐日账,用同一套结算代码按新仓位重放一遍
    E.输出目录 = 输出
    os.makedirs(输出, exist_ok=True)
    eng.状态.update(daily=[], 资产仓位=[], 分资产=[], 前暴露={}, 前仓资产={})
    eng.组合切片 = []                       # 重放不再产存档,避免污染原始存档
    eng._结算(list(仓.index), 仓)
    eng._导出()
    print(f"已写出 {输出}")


if __name__ == "__main__":
    main()
