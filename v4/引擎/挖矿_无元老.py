# -*- coding: utf-8 -*-
"""受控实验:不入册元老(9趋势+4argarch),其余与 挖矿_26算子 完全相同,从零重挖。

动机:元老不只占权重(在役7人平均占zz1000组合18%,cos全负,剔除重放+2.1pp),
还充当早期残差闸的正交基准——月动量20/通道位置60/双均线20等把带动量成分的
候选挡在库外。去掉元老,库的进化路径会整体改变,方向不可预测,故重挖实测。

用法: python -m 引擎.挖矿_无元老 [--end YYYY-MM-DD]
产出: 结果_无元老挖矿/
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
    from walkforward.elders import ArgarchRecorder
    if a.end:
        C.END = a.end

    E.输出目录 = os.path.join(根, "结果_无元老挖矿")
    E.安装GA()

    import ga_alpha.ops as gops
    for k in 正典外算子:
        gops.OPS.pop(k, None)
    assert len(gops.OPS) == 26

    # 不入册元老;AR-GARCH记录器只服务argarch元老,一并停掉(纯提速,无他用)
    E.宽基引擎._元老入册 = lambda self, q: None
    ArgarchRecorder.extend = lambda self, cutoff, days: None
    print("[无元老] 26算子,元老不入册,argarch记录器停用")

    E.宽基引擎(E.load_all()).run()


if __name__ == "__main__":
    main()
