# -*- coding: utf-8 -*-
"""v5.1 生产信号:六库合奏 + 状态层风险预算回收。

六库 = 旧叶子(9)h1/h5 × 种子42 + 新叶子(16)h1/h5 × 种子42/137
       —— 叶子集 × horizon × 种子 三维分散,单库种子噪音 ±2~4pp 被平均掉。
状态层 = limit_net(涨停−跌停占比)20日均的滚动2年分位 < 10% 判为恐慌态(约占11%交易日),
         恐慌时多头×0.5;省出的回撤预算以整体×k 回收。

口径:信号 T 日收盘生成 → T+1 开盘建仓 → T+2 开盘结算(open→open,结算滞后2格)。
每日全量重算(不做增量延算——v3 延算曾因黄牌口径/存档缺行翻车)。

产出 results/final/v51_逐日.csv:date, p六库, 恐慌, 仓位, 收益
用法: PYTHONPATH=src python -m csi1000.engine.live_v51 [--k 1.2]
"""
from __future__ import annotations

import argparse
import os
import pickle

import numpy as np
import pandas as pd
from scipy.stats import norm

from csi1000 import paths

因子窗 = 映射窗 = 40
参照回看 = 60
TRAILING月 = 12
恐慌分位, 恐慌窗, 恐慌均 = 0.10, 504, 20      # 预注册,勿调参
多头折 = 0.5

六库 = [("results/mine_open/h1",       1, False),
       ("results/mine_open/h5",       5, False),
       ("results/mine_xsec/h1_s42",   1, True),
       ("results/mine_xsec/h1_s137",  1, True),
       ("results/mine_xsec/h5_s42",   5, True),
       ("results/mine_xsec/h5_s137",  5, True)]


def _单仓(S):
    π = (S.rolling(映射窗).rank(pct=True) - 0.5 / 映射窗).clip(1e-6, 1 - 1e-6)
    return pd.Series(np.tanh(norm.ppf(π)), index=S.index)


def _cos(p, r):
    d = pd.concat([p.rename("p"), r.rename("r")], axis=1).dropna()
    den = np.sqrt((d.p ** 2).sum() * (d.r ** 2).sum())
    return float((d.p * d.r).sum() / den) if den > 0 else np.nan


def 一库π(目录: str, h: int, 面板, 日历, E, gexpr) -> pd.Series | None:
    """重算单库的映射分位 π(与 eval_open_horizons 同口径,季内治理沿用存档)。"""
    需 = [os.path.join(paths.根, 目录, f) for f in ("state.pkl", "因子逐日信号.csv.gz", "名册.csv")]
    if not all(os.path.exists(f) for f in 需):
        return None
    st_ = pickle.load(open(需[0], "rb"))
    存 = pd.read_csv(需[1], parse_dates=["date"])
    名 = pd.read_csv(需[2])
    元老 = set(名[名.kind.isin(["elder", "argarch"])]["编号"])
    治理 = 存[(存.指数 == "zz1000") & (~存.因子.isin(元老))]
    if len(治理) == 0:
        return None
    r次 = 面板["open"]["zz1000"].pct_change(h, fill_method=None).shift(-(h + 1))

    信号, p表 = {}, {}
    for f in sorted(治理.因子.unique()):
        m = st_["members"][f]
        面 = 面板 if m["专属"] is None else {k: p[["zz1000"]] for k, p in 面板.items()}
        v = gexpr.evaluate(m["node"], 面).replace([np.inf, -np.inf], np.nan)
        v = v["zz1000"] if "zz1000" in v.columns else v.iloc[:, 0]
        信号[f] = (v.rolling(因子窗).rank(pct=True) * 2 - 1) * m["sign"]
        p表[f] = _单仓(信号[f])

    π全 = pd.Series(dtype=float)
    for q in E.quarter_list("2015-01-01", str(日历[-1].date())):
        本 = 治理[(治理.date >= q.start_time) & (治理.date <= q.end_time)]
        if len(本) == 0:
            continue
        # 季日取交易日历:存档末日之后的新交易日照常进入本季,名单/黄牌沿用本季存档值
        # (季内治理事件不变,故合法)——实盘延伸依赖此行为
        季日 = 日历[(日历 >= q.start_time) & (日历 <= q.end_time)]
        季日 = 季日[季日 >= 本.date.min()]
        if len(季日) == 0:
            continue
        cutoff = q.start_time - pd.Timedelta(days=1)
        起 = cutoff - pd.DateOffset(months=TRAILING月)
        权重 = {}
        for f, g in 本.groupby("因子"):
            黄 = g.sort_values("date")["黄牌"].iloc[0]
            c = _cos(p表[f].loc[起:cutoff].iloc[:-1], r次.loc[起:cutoff].iloc[:-1])
            if np.isfinite(c) and c > 0:
                权重[f] = c * (0.5 if 黄 else 1.0)
        总 = sum(权重.values())
        if 总 <= 0:
            continue
        减半 = 0.5 if len(权重) < E.烧机最少 else 1.0
        i0 = max(0, 日历.searchsorted(季日[0]) - 参照回看)
        扩 = 日历[i0:日历.searchsorted(季日[-1]) + 1]
        合 = pd.Series(0.0, index=扩)
        for f, w in 权重.items():
            合 = 合.add(信号[f].reindex(扩).fillna(0.0) * (w / 总), fill_value=0.0)
        合 = (合 * 减半).clip(-1, 1)
        πq = (合.rolling(映射窗).rank(pct=True) - 0.5 / 映射窗).clip(1e-6, 1 - 1e-6)
        πq = πq.where(合.abs().gt(1e-12))
        π全 = pd.concat([π全, πq.loc[季日]])
    return π全[~π全.index.duplicated()]


def 算(k: float = 1.2, 详细: bool = True) -> pd.DataFrame:
    import csi1000.engine.index_engine as E
    import csi1000.ga_alpha.expr as gexpr
    import csi1000.engine.strategy as st
    from csi1000.walkforward import config as C
    from csi1000.engine.xsec_leaves import 注入

    C.END = str(pd.read_csv(os.path.join(paths.行情缓存, "idx_sh000852.csv"),
                            usecols=["date"])["date"].max())
    E.输出目录 = paths.结果_26算子
    E.安装GA()
    from csi1000.walkforward.data import load_all
    基面板 = load_all()["classes"]["A股宽基"]["panels"]
    日历 = 基面板["close"].index
    xsec面板 = {k2: v.copy() for k2, v in 基面板.items()}
    注入(xsec面板)

    仓位表 = []
    for 目录, h, 用xsec in 六库:
        π = 一库π(目录, h, xsec面板 if 用xsec else 基面板, 日历, E, gexpr)
        if π is None or len(π) == 0:
            if 详细: print(f"  {目录}: 跳过(产物不全)")
            continue
        仓位表.append(pd.Series(np.tanh(norm.ppf(π)), index=π.index).where(π.notna(), 0.0))
        if 详细: print(f"  {os.path.basename(目录):12s} h={h} → {len(π)}天,末值 {仓位表[-1].iloc[-1]:+.3f}")
    if not 仓位表:
        raise RuntimeError("六库全部不可用")
    共 = 仓位表[0].index
    for x in 仓位表[1:]:
        共 = 共.union(x.index)
    p6 = pd.concat([x.reindex(共) for x in 仓位表], axis=1).mean(axis=1).dropna()

    # ---- 状态层:恐慌态多头折半 + 整体×k 回收风险预算 ----
    叶 = pd.read_csv(os.path.join(paths.存档, "截面叶子.csv"),
                     parse_dates=["date"]).set_index("date")
    恐慌 = (叶["limit_net"].rolling(恐慌均).mean()
            .rolling(恐慌窗).rank(pct=True).reindex(p6.index) < 恐慌分位)
    调 = p6.where(~恐慌, p6.where(p6 < 0, p6 * 多头折))
    仓位 = (调 * k).clip(-1, 1)

    r = st.读价("etf_512100", "open").pct_change()
    rf = st.读价("etf_511880").pct_change().clip(lower=0)
    仓位 = 仓位.where(r.reindex(仓位.index).notna(), 0.0).fillna(0.0)
    收益 = st.结算(仓位, r, rf, "open")
    出 = pd.DataFrame({"p六库": p6, "恐慌": 恐慌.astype(int), "仓位": 仓位,
                      "收益": 收益, "标的开→开": r.reindex(仓位.index),
                      "货基": rf.reindex(仓位.index)}).dropna(subset=["仓位"])
    os.makedirs(paths.结果, exist_ok=True)
    出.to_csv(os.path.join(paths.结果, "v51_逐日.csv"))
    return 出


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=float, default=1.2, help="风险预算回收系数(1.2保守/1.3档案值)")
    a = ap.parse_args()
    print(f"v5.1 六库合奏 + 状态层(k={a.k})")
    出 = 算(a.k)
    末 = 出.index[-1]
    ret = 出["收益"].dropna().loc["2018":]
    nav = (1 + ret).cumprod(); 年 = len(ret) / 244
    ann = nav.iloc[-1] ** (1 / 年) - 1
    ex = ret - 出["货基"].reindex(ret.index).fillna(0)
    dd = (nav / nav.cummax() - 1).min()
    print(f"\n2018起: 年化{ann*100:.1f}%  夏普{ex.mean()/ex.std()*np.sqrt(244):.2f}  "
          f"回撤{dd*100:.1f}%  卡玛{ann/abs(dd):.2f}")
    print(f"信号末日 {末.date()}  恐慌态={'是' if 出['恐慌'].iloc[-1] else '否'}  "
          f"→ 下一交易日开盘目标仓位 {出['仓位'].iloc[-1]*100:+.1f}%")
    return 出


if __name__ == "__main__":
    main()
