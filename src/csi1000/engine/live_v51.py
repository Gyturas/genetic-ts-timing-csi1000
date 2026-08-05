# -*- coding: utf-8 -*-
"""v5.2 生产信号:四库合奏(旧叶双种子) + 状态层风险预算回收。

四库 = 旧叶子(9)h1/h5 × 种子42/137 —— horizon × 种子 双维分散(种子对称)。
(v5.1 的六库含 4 个新叶库;A批重挖后按 S4 预注册判据回退,见上方常量注释。)
状态层 = limit_net(涨停−跌停占比)20日均的滚动2年分位 < 10% 判为恐慌态(约占11%交易日),
         恐慌时多头×0.5;省出的回撤预算以整体×k 回收。

口径:信号 T 日收盘生成 → T+1 开盘建仓 → T+2 开盘结算(open→open,结算滞后2格)。
每日全量重算(不做增量延算——v3 延算曾因黄牌口径/存档缺行翻车)。

口径纪律(2026-08-05 审计后固化,详见 docs/因子库缺陷审计.md):
  · 剔尾条数 = h+1(不是 1)。r次[d]=open[d+1+h]/open[d+1]−1 用到 d+1+h 日开盘价。
  · cos IC 至少 最少配对 对样本才给权重。
  · 生产库缺一即抛错,不静默降级;截面叶子落后于信号即抛错(NaN<0.10 恒 False)。
  · 整季无正权写 NaN 而非丢日期(结算 shift(2) 是位置型,缺季会错接跨季仓位)。

产出 results/final/v51_逐日.csv:date, p六库, 恐慌, 仓位, 收益, 库数
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
最少配对 = 120                                # cos IC 最小样本量(≈半年),预注册阈值

# 生产库构成(2026-08-05 A批重挖后按 S4 预注册判据裁决):
#   旧四库(9叶,h1/h5 × 种子42/137)28.0%/1.31/2022+0.91 全面优于
#   八库(+新叶四库)26.9%/1.26/0.90 与新四库单独 25.4%/1.18/0.86
#   → 依「八库需年化或夏普改善且另一边不劣、2022+夏普不劣」判据,新叶四库出生产。
#   判据先于看表写死于 docs/因子库缺陷审计.md S4 章节;新叶库仍保留存档,v6 可另行预注册再议。
六库 = [("results/mine_open/h1",        1, False),
       ("results/mine_open/h5",        5, False),
       ("results/mine_open_s137/h1",   1, False),
       ("results/mine_open_s137/h5",   5, False)]


def _单仓(S):
    π = (S.rolling(映射窗).rank(pct=True) - 0.5 / 映射窗).clip(1e-6, 1 - 1e-6)
    return pd.Series(np.tanh(norm.ppf(π)), index=S.index)


def _cos(p, r):
    """P&L 对齐的 cos 相似度。样本不足 最少配对 时返回 nan(该因子当季不给权重)。"""
    d = pd.concat([p.rename("p"), r.rename("r")], axis=1).dropna()
    if len(d) < 最少配对:
        return np.nan
    den = np.sqrt((d.p ** 2).sum() * (d.r ** 2).sum())
    return float((d.p * d.r).sum() / den) if den > 0 else np.nan


def 一库π(目录: str, h: int, 面板, 日历, E, gexpr) -> pd.Series | None:
    """重算单库的映射分位 π(与 eval_open_horizons 同口径,季内治理沿用存档)。"""
    需 = [os.path.join(paths.根, 目录, f) for f in ("state.pkl", "因子逐日信号.csv.gz", "名册.csv")]
    if not all(os.path.exists(f) for f in 需):
        return None, {"错": "产物不全"}
    st_ = pickle.load(open(需[0], "rb"))
    存 = pd.read_csv(需[1], parse_dates=["date"])
    名 = pd.read_csv(需[2])
    元老 = set(名[名.kind.isin(["elder", "argarch"])]["编号"])
    治理 = 存[(存.指数 == "zz1000") & (~存.因子.isin(元老))]
    if len(治理) == 0:
        return None, {"错": "无在役非元老因子"}
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
    诊断 = {"缺季": [], "薄季": [], "零权季": [], "末季正权": None, "末季在册": None}
    for q in E.quarter_list("2015-01-01", str(日历[-1].date())):
        本 = 治理[(治理.date >= q.start_time) & (治理.date <= q.end_time)]
        if len(本) == 0:
            # 存档滞后于行情时本季无名册行 → 该库整季无 π,六库合奏会静默退化为五库
            if len(日历[(日历 >= q.start_time) & (日历 <= q.end_time)]):
                诊断["缺季"].append(str(q))
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
            # 剔尾 h+1:r次[d]=open[d+1+h]/open[d+1]−1 用到 d+1+h 日开盘价,
            # 季初定权重时只知道 ≤cutoff 的价格,窗口末 h+1 行决策时点不可得。
            # (close 口径的 final_signal 用 returns.shift(-1),只需剔 1 行,勿照抄。)
            c = _cos(p表[f].loc[起:cutoff].iloc[:-(h + 1)],
                     r次.loc[起:cutoff].iloc[:-(h + 1)])
            if np.isfinite(c) and c > 0:
                权重[f] = c * (0.5 if 黄 else 1.0)
        总 = sum(权重.values())
        诊断["末季正权"], 诊断["末季在册"] = len(权重), 本.因子.nunique()
        if 总 <= 0:
            # 整季无正权 → 写 NaN 占位(下游 where(notna) 变空仓),不能丢日期:
            # strategy.结算 的 shift(2) 是位置型,缺季会把跨季仓位错接且不报错
            诊断["零权季"].append(str(q))
            π全 = pd.concat([π全, pd.Series(np.nan, index=季日)])
            continue
        if len(权重) < E.烧机最少:
            诊断["薄季"].append(f"{q}({len(权重)})")
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
    return π全[~π全.index.duplicated()], 诊断


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

    仓位表, 缺库 = [], []
    for 目录, h, 用xsec in 六库:
        π, 诊 = 一库π(目录, h, xsec面板 if 用xsec else 基面板, 日历, E, gexpr)
        if π is None or len(π) == 0:
            缺库.append(f"{os.path.basename(目录)}({诊.get('错', '空π')})")
            continue
        仓位表.append(pd.Series(np.tanh(norm.ppf(π)), index=π.index).where(π.notna(), 0.0))
        if 详细:
            警 = ""
            if 诊["缺季"]: 警 += f"  ⚠存档缺{len(诊['缺季'])}季(最近{诊['缺季'][-1]})"
            if 诊["零权季"]: 警 += f"  ⚠零权{len(诊['零权季'])}季"
            if 诊["薄季"]: 警 += f"  ⚠薄季{len(诊['薄季'])}次"
            print(f"  {os.path.basename(目录):12s} h={h} → {len(π)}天,末值 {仓位表[-1].iloc[-1]:+.3f}"
                  f"  末季正权{诊['末季正权']}/{诊['末季在册']}{警}")
    # 分散化前提是六库都在。缺库不是「少一点分散」,是把种子/horizon 噪音又放回来了
    if len(仓位表) != len(六库):
        raise RuntimeError(f"仅 {len(仓位表)}/{len(六库)} 库可用,缺:{'、'.join(缺库)}。"
                           f"生产库集是预注册配置,不接受静默降级——修好产物或显式改 六库 常量(v5.2=旧四库)。")
    共 = 仓位表[0].index
    for x in 仓位表[1:]:
        共 = 共.union(x.index)
    p6 = pd.concat([x.reindex(共) for x in 仓位表], axis=1).mean(axis=1).dropna()

    # ---- 状态层:恐慌态多头折半 + 整体×k 回收风险预算 ----
    叶 = pd.read_csv(os.path.join(paths.存档, "截面叶子.csv"),
                     parse_dates=["date"]).set_index("date")
    # NaN < 0.10 恒为 False → 叶子断更时状态层静默失效,且失效方向是「不保护」。必须显式校验。
    if 叶.index.max() < p6.index.max():
        raise RuntimeError(f"截面叶子末日 {叶.index.max().date()} 落后于信号末日 "
                           f"{p6.index.max().date()},状态层会静默失效(方向=不保护)。"
                           f"先跑 update_stocks。")
    分位 = (叶["limit_net"].rolling(恐慌均).mean()
            .rolling(恐慌窗).rank(pct=True).reindex(p6.index))
    缺 = 分位.isna().sum()
    if 缺 and 详细:
        print(f"  ⚠ 状态层 {缺} 天无分位(建仓期),按非恐慌处理:{list(分位[分位.isna()].index[:3].date)}")
    恐慌 = 分位 < 恐慌分位
    调 = p6.where(~恐慌, p6.where(p6 < 0, p6 * 多头折))
    仓位 = (调 * k).clip(-1, 1)

    r = st.读价("etf_512100", "open").pct_change()
    rf = st.读价("etf_511880").pct_change().clip(lower=0)
    仓位 = 仓位.where(r.reindex(仓位.index).notna(), 0.0).fillna(0.0)
    收益 = st.结算(仓位, r, rf, "open")
    出 = pd.DataFrame({"p六库": p6, "恐慌": 恐慌.astype(int), "仓位": 仓位,
                      "收益": 收益, "标的开→开": r.reindex(仓位.index),
                      "货基": rf.reindex(仓位.index)}).dropna(subset=["仓位"])
    出["库数"] = len(仓位表)          # 面板显示用:合奏实际由几个库构成
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
