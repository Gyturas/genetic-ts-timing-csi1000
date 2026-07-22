# -*- coding: utf-8 -*-
"""用 26 算子口径重挖 —— 复现仓库发布版(年化20.7%/夏普1.01)的因子库。

背景:仓库自带的 数据/分资产逐日.csv(产出20.7%)是用 26 个算子挖的,
GA 因子数恰为 1232。后来算子表加了 ts_ac1 / ts_cj / ts_vr5 三个"结构传感"
算子(ga_alpha/ops.py 注释:"快慢手方案首发"),变成 29 个。算子表一变,
GA 的搜索空间就变了,同种子抽到的表达式树完全不同,因子库自然是另一批
——这跟浮点无关,是代码版本不一致。

本脚本在挖矿前把这三个算子从 OPS 里摘掉,其余(种子、GA参数、入库线、
数据、映射)全部不动,用来验证"算子表差异"就是 20.7% 复现不出来的原因。

用法: python -m 引擎.挖矿_26算子 [--end YYYY-MM-DD]
产出: 结果_26算子/  (含 因子逐日信号.csv.gz,可直接喂给 重算权重/cos_ic权重)
"""
from __future__ import annotations

import argparse
import os
import sys

正典外算子 = ("ts_ac1", "ts_cj", "ts_vr5")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--end", default=None)
    a = ap.parse_args()

    根 = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    import csi1000.engine.index_engine as E
    from csi1000.walkforward import config as C
    if a.end:
        C.END = a.end

    E.输出目录 = os.path.join(根, "结果_26算子")
    E.安装GA()

    import csi1000.ga_alpha.ops as gops
    n0 = len(gops.OPS)
    for k in 正典外算子:
        gops.OPS.pop(k, None)
    n1 = len(gops.OPS)
    print(f"[26算子] 算子表 {n0} → {n1}(摘掉 {', '.join(正典外算子)})")
    assert n1 == 26, f"摘完应为26个算子,实为 {n1};算子表可能又变过,请核对"

    E.宽基引擎(E.load_all()).run()


if __name__ == "__main__":
    main()
