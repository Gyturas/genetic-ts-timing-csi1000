# -*- coding: utf-8 -*-
"""v3 实盘:延算最新收盘日信号,给出下一交易日开盘的中证1000推荐仓位。

口径 = v3 定稿:26算子库 + cos IC 加权(季度锁定权重) + 元老明文剔除 + 熊增强ΔS。
自检制度:延算的组合信号必须在近期已有日子上与 数据/分资产逐日.csv(定稿信号)
位位一致(<1e-8),否则拒绝输出——手工重建历史上翻过车,先复现再报数。

用法: PYTHONPATH=. python -m 引擎.明日仓位 [--end 2026-07-21]
"""
from __future__ import annotations

import argparse
import os
import pickle

import numpy as np
import pandas as pd

根 = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAILING月 = 12


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--end", default="2026-07-21")
    a = ap.parse_args()

    import 引擎.指数日频引擎 as E
    from walkforward import config as C
    import ga_alpha.expr as gexpr
    from 引擎.cos_ic权重 import 单因子仓位, cos_ic
    import 引擎.策略 as st

    C.END = a.end
    E.输出目录 = os.path.join(根, "结果_26算子")
    E.安装GA()
    from walkforward.data import load_all
    数据 = load_all()
    面板 = 数据["classes"]["A股宽基"]["panels"]
    日历 = 面板["close"].index
    assert str(日历[-1].date()) == a.end, f"面板末日 {日历[-1].date()} ≠ {a.end},行情没更新到位"
    次日收益 = 面板["returns"].shift(-1)["zz1000"]

    st_ = pickle.load(open(os.path.join(根, "结果_26算子", "state.pkl"), "rb"))
    在役 = {i: m for i, m in st_["members"].items()
           if m["status"] == "在役" and m["kind"] == "ga" and m["专属"] in (None, "zz1000")}
    print(f"在役可用因子(剔元老后): {len(在役)} 个")

    # 黄牌须用存档 as-of cutoff 口径(cos_ic权重.py 同款):Q3 权重由 Q2 季度步的黄牌
    # 状态决定,state.pkl 的 yellow 是 Q3 季度步后的终态,跨季变动的因子会差 0.5 倍权重
    存tmp = pd.read_csv(os.path.join(根, "结果_26算子", "因子逐日信号.csv.gz"),
                        parse_dates=["date"])
    ztmp = 存tmp[存tmp.指数 == "zz1000"]
    黄牌asof = {}
    for fid, g in ztmp.groupby("因子"):
        h = g.set_index("date")["黄牌"].sort_index().loc[:"2026-06-30"]
        黄牌asof[fid] = bool(h.iloc[-1]) if len(h) else False

    # ---- 权重与合成:与 定稿信号(cos_ic权重.py)完全同源——一律用存档信号 ----
    # 重新求值的信号只用来延伸存档之后的新交易日(该段已验证与存档位位一致)。
    q3 = pd.Timestamp("2026-07-01")
    cutoff = pd.Timestamp("2026-06-30")
    起 = cutoff - pd.DateOffset(months=TRAILING月)
    档末 = ztmp["date"].max()
    权重, 信号 = {}, {}
    for i, m in 在役.items():
        g = ztmp[ztmp.因子 == i]
        if len(g) == 0:
            continue
        s档 = g.set_index("date")["信号"].sort_index()
        s档 = s档[~s档.index.duplicated(keep="last")]
        # 延算段:重新求值,只取存档之后的日子
        面 = 面板 if m["专属"] is None else {k: p[["zz1000"]] for k, p in 面板.items()}
        v = gexpr.evaluate(m["node"], 面).replace([np.inf, -np.inf], np.nan)
        S = (v.rolling(E.RANK窗).rank(pct=True) * 2 - 1) * m["sign"]
        s新 = (S["zz1000"] if "zz1000" in S.columns else S.iloc[:, 0]).loc[档末 + pd.Timedelta(days=1):]
        p = 单因子仓位(s档)                       # 权重用纯存档段:与定稿同源
        c = cos_ic(p.loc[起:cutoff], 次日收益.loc[起:cutoff])
        if not np.isfinite(c) or c <= 0:
            continue
        权重[i] = c * (0.5 if 黄牌asof.get(i, False) else 1.0)
        信号[i] = pd.concat([s档, s新])
    总 = sum(权重.values())
    assert 总 > 0 and len(权重) >= E.烧机最少, "可用因子不足"
    print(f"cos>0 拿到权重的因子: {len(权重)} 个")

    idx = 日历[(日历 >= q3) ]
    comb = pd.Series(0.0, index=idx)
    for i, w in 权重.items():
        comb = comb.add(信号[i].reindex(idx).fillna(0.0) * (w / 总), fill_value=0.0)
    comb = comb.clip(-1, 1)

    # ---- 自检:Q3 已有日子必须复现定稿信号 ----
    定稿 = pd.read_csv(os.path.join(根, "数据", "分资产逐日.csv"),
                     index_col=0, parse_dates=True)[f"{st.ETF}_仓位"]
    共 = 定稿.loc[q3:].index.intersection(comb.index)
    差 = (comb.loc[共] - 定稿.loc[共]).abs().max()
    print(f"自检: Q3 重叠 {len(共)} 天,延算 vs 定稿 最大差 {差:.2e}", end="  ")
    assert 差 < 1e-8, "✗ 复现失败,拒绝输出仓位!"
    print("✓ 复现通过")

    # ---- 组合信号 + 熊增强 + 仓位映射 ----
    新日 = comb.index[comb.index > 定稿.index.max()]
    S全 = pd.concat([定稿, comb.loc[新日]])
    ΔS = pd.read_csv(os.path.join(根, "数据", "熊增强逐日.csv"),
                     parse_dates=["date"]).set_index("date")[f"ΔS_{st.ETF}"]
    尾ΔS = float(ΔS.iloc[-1])
    S增强 = S全 + ΔS.reindex(S全.index).fillna(尾ΔS)   # 新日沿用最后ΔS(当前=0,慢变量)
    r = st.读价(f"etf_{st.ETF}").pct_change().reindex(S全.index)
    p多空 = st.建仓位(S增强, r, 禁空=False)
    p纯多 = st.建仓位(S增强, r, 禁空=True)

    print("\n" + "=" * 62)
    print(f"最新收盘日 {S全.index[-1].date()}(信号) → 下一交易日开盘执行")
    print("=" * 62)
    print(f"  组合信号 comb+ΔS = {float(S增强.iloc[-1]):+.4f}   (ΔS={尾ΔS:+.3f})")
    print(f"  多空口径推荐仓位: {float(p多空.iloc[-1]):+.1%}")
    print(f"  纯多口径推荐仓位: {float(p纯多.iloc[-1]):+.1%}")
    print("\n近8个交易日仓位轨迹(多空):")
    for d, v in p多空.tail(8).items():
        print(f"  {d.date()}  {v:+.1%}")


if __name__ == "__main__":
    main()
