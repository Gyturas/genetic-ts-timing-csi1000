# -*- coding: utf-8 -*-
"""开盘口径挖矿 × 短期 horizon(1~7天)。

口径闭环(与 v4.1 实盘执行一致):信号 T 收盘产生 → T+1 开盘建仓 → 持 h 日 → T+1+h 开盘结算。
故预测目标 Y_T = open[T+1+h]/open[T+1] − 1 = open.pct_change(h).shift(-(h+1))。
(对比定稿挖矿:Y = close.pct_change().shift(-1),尾盘口径 h=1。A股隔夜跳空频繁,
 close 口径的 Y 里有实盘吃不到的成分——本实验让挖矿目标与结算价完全一致。)

挖矿(GA适应度)与治理(入库/黄牌/退役的月度IC)统一换到该 Y;其余制度全部照抄 v4 定稿:
26 算子、双库结构(类库+专属)、季度 walk-forward、入库三关、黄牌退役。

注意:h>1 时按天滚动的 Y 相邻重叠(h-1)/h,月度IC绝对值系统性偏大,入库线 0.015 保持
不变意味着大 h 的入库数量不可与 h=1 直接比;不同 h 的优劣以 eval_open_horizons 的
统一回测(逐日调仓、open→open 结算)为准。

产出: results/mine_open/h{h}/(state.pkl、因子逐日信号.csv.gz、名册.csv 等,结构同 ops26)
用法: PYTHONPATH=src python -m csi1000.engine.mine_open_horizons --h 1 [--end YYYY-MM-DD] [--smoke]
"""
from __future__ import annotations

import argparse
import os

正典外算子 = ("ts_ac1", "ts_cj", "ts_vr5")   # 与 mine_ops26 一致:摘掉后回到 26 算子定稿口径


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--h", type=int, required=True, choices=range(1, 8),
                    help="持有期(天):Y=open[T+1+h]/open[T+1]-1")
    ap.add_argument("--end", default=None)
    ap.add_argument("--smoke", action="store_true", help="GA缩到40×5,快速验证管线")
    ap.add_argument("--seed", type=int, default=42, help="GA种子(换种子复验稳健性)")
    ap.add_argument("--tag", default="", help="输出目录后缀,如 _s137 → results/mine_open_s137/")
    a = ap.parse_args()

    from csi1000 import paths
    import csi1000.engine.index_engine as E
    from csi1000.walkforward import config as C
    import csi1000.ga_alpha.fitness as gfit
    import csi1000.ga_alpha.ops as gops

    if a.end:
        C.END = a.end
    if a.smoke:
        E.GA参数.update(population=40, generations=5)

    E.输出目录 = os.path.join(paths.根, "results", f"mine_open{a.tag}", f"h{a.h}")
    E.GA参数["seed"] = a.seed
    E.安装GA()
    n0 = len(gops.OPS)
    for k in 正典外算子:
        gops.OPS.pop(k, None)
    assert len(gops.OPS) == 26, f"算子表应为26,实为{len(gops.OPS)}(原{n0})"

    数据 = E.load_all()
    全开 = 数据["classes"]["A股宽基"]["panels"]["open"]
    h = a.h

    def 开盘前瞻(close, horizon, _o=全开, _h=h):
        # close 仅用来对齐窗口/列(类库6列、专属库1列);目标价一律用开盘
        o = _o.reindex(index=close.index, columns=close.columns)
        return o.pct_change(_h, fill_method=None).shift(-(_h + 1))

    gfit.forward_returns = 开盘前瞻                       # 挖矿适应度的 Y

    eng = E.宽基引擎(数据)
    eng.次日收益 = 全开.pct_change(h, fill_method=None).shift(-(h + 1))   # 治理记账的 Y
    eng.Y丢尾 = h + 1     # A3:Y[t] 用到 t+1+h 日开盘价,决策时点 cutoff 只知 ≤cutoff
                          # —— 必须与 次日收益 同步设置,否则 _干净Y 会按 close 口径只丢 1 行
    print(f"[开盘挖矿] h={h}  Y=open[T+1+h]/open[T+1]-1  END={C.END}  输出={E.输出目录}")
    eng.run()


if __name__ == "__main__":
    main()
