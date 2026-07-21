# -*- coding: utf-8 -*-
"""v2 挖矿入口:cos IC 打分 + 26 算子。

与 v1(引擎/挖矿_26算子.py)唯一的区别是打分口径——ga_alpha/fitness.py 已把
进化打分从 rank IC 换成 cos IC,引擎的入库门槛与记账也改用 混合月度cosIC
(残差闸仍 rank)。算子表照样摘掉后加的 3 个,保持 26 算子,确保与 v1 是
只差"打分口径"这一个变量的受控对比。

用法: python -m 引擎.挖矿_cosIC [--end YYYY-MM-DD]
产出: 结果_cosIC挖矿/  (含 因子逐日信号.csv.gz,可直接喂给 重算权重/cos_ic权重)
"""
from __future__ import annotations

import argparse
import os

正典外算子 = ("ts_ac1", "ts_cj", "ts_vr5")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--end", default=None)
    a = ap.parse_args()

    根 = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    import 引擎.指数日频引擎 as E
    from walkforward import config as C
    if a.end:
        C.END = a.end

    E.输出目录 = os.path.join(根, "结果_cosIC挖矿")
    E.安装GA()

    import ga_alpha.ops as gops
    n0 = len(gops.OPS)
    for k in 正典外算子:
        gops.OPS.pop(k, None)
    n1 = len(gops.OPS)
    print(f"[cosIC挖矿] 算子表 {n0} → {n1}(摘掉 {', '.join(正典外算子)}),打分=cos IC")
    assert n1 == 26, f"摘完应为26个算子,实为 {n1};算子表可能又变过,请核对"

    E.宽基引擎(E.load_all()).run()


if __name__ == "__main__":
    main()
