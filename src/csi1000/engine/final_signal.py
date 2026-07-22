# -*- coding: utf-8 -*-
"""v4 定稿信号:两层 40 日窗 + 同质参照系 + cos IC 加权 + 元老剔除。

与 v3 的差异(全部经受控实验验证,见 说明.md):
  ① 第一层因子信号窗 120→40(zz1000 +0.10夏普,40~80平台,zz500 交叉复现 +0.12)
  ② 第二层映射分位改"同质参照系":每季用本季权重回算 [季前60日,季末] 的
     comb 序列当排位参照,窗口内不再混新旧权重体制(+0.03,按机制正确性采纳)
  沿用 v3:cos IC 季度权重、元老(13人)明文剔除、治理层(每季名单/黄牌/烧机)
  取自挖矿存档 结果_26算子/(rank120 记账口径,挖矿层不动)。

因子信号不能再读存档(存档是 rank120 口径),改为表达式重算 rank40。

产出:
  数据/分资产逐日.csv   512100_仓位 列 = 定格拼接的 comb(逐季当季权重,兼容出图)
  数据/映射分位.csv     π_增强 / π_量价 两列(同质参照系口径,策略.py 直接使用)

用法: PYTHONPATH=. python -m 引擎.定稿信号
"""
from __future__ import annotations

import os
import pickle

import numpy as np
import pandas as pd
from scipy.stats import norm

from csi1000 import paths
因子窗 = 40          # ① 第一层
映射窗 = 40          # 第二层(同 v3)
参照回看 = 60        # 同质参照系向季前多算的交易日数(>映射窗 即可)
TRAILING月 = 12


def main():
    import csi1000.engine.index_engine as E
    import csi1000.ga_alpha.expr as gexpr
    from csi1000.walkforward import config as C
    # END 跟随行情实际末日:行情更新后信号自动延伸到最新收盘日(实盘依赖此行为)
    C.END = str(pd.read_csv(os.path.join(paths.行情缓存, "idx_sh000852.csv"),
                            usecols=["date"])["date"].max())
    E.输出目录 = paths.结果_26算子
    E.安装GA()
    from csi1000.walkforward.data import load_all
    面板 = load_all()["classes"]["A股宽基"]["panels"]
    日历 = 面板["close"].index
    次日收益 = 面板["returns"].shift(-1)["zz1000"]

    st_ = pickle.load(open(os.path.join(paths.结果_26算子, "state.pkl"), "rb"))
    名册 = pd.read_csv(os.path.join(paths.存档, "名册.csv"))
    元老 = set(名册[名册.kind.isin(["elder", "argarch"])]["编号"])
    存 = pd.read_csv(os.path.join(paths.结果_26算子, "因子逐日信号.csv.gz"), parse_dates=["date"])
    治理 = 存[(存.指数 == "zz1000") & (~存.因子.isin(元老))]   # 每季名单/黄牌/烧机,rank120口径记账

    # ---- 因子信号:表达式重算,rank40,乘录取时的方向号 ----
    def 单仓(S):
        π = (S.rolling(映射窗).rank(pct=True) - 0.5 / 映射窗).clip(1e-6, 1 - 1e-6)
        return pd.Series(np.tanh(norm.ppf(π)), index=S.index)

    信号, p表 = {}, {}
    for f in sorted(治理.因子.unique()):
        m = st_["members"][f]
        面 = 面板 if m["专属"] is None else {k: p[["zz1000"]] for k, p in 面板.items()}
        v = gexpr.evaluate(m["node"], 面).replace([np.inf, -np.inf], np.nan)
        v = v["zz1000"] if "zz1000" in v.columns else v.iloc[:, 0]
        信号[f] = (v.rolling(因子窗).rank(pct=True) * 2 - 1) * m["sign"]
        p表[f] = 单仓(信号[f])
    print(f"重算 {len(信号)} 个因子信号(rank{因子窗})")

    def cos_ic(p, r):
        d = pd.concat([p.rename("p"), r.rename("r")], axis=1).dropna()
        den = np.sqrt((d.p ** 2).sum() * (d.r ** 2).sum())
        return float((d.p * d.r).sum() / den) if den > 0 else np.nan

    ΔS = pd.read_csv(os.path.join(paths.存档, "熊增强逐日.csv"),
                     parse_dates=["date"]).set_index("date")["ΔS_512100"]

    comb定格 = pd.Series(dtype=float)
    π增, π价 = pd.Series(dtype=float), pd.Series(dtype=float)
    for q in E.quarter_list("2015-01-01", str(日历[-1].date())):
        本季治理 = 治理[(治理.date >= q.start_time) & (治理.date <= q.end_time)]
        if len(本季治理) == 0:
            continue
        # 季日用交易日历(而非存档日期):存档末日之后的新交易日照常进入本季,
        # 名单/黄牌沿用本季存档值(季内治理事件不变,故合法)——实盘延伸依赖此行为
        季日 = 日历[(日历 >= q.start_time) & (日历 <= q.end_time)]
        季日 = 季日[季日 >= 本季治理.date.min()]
        cutoff = q.start_time - pd.Timedelta(days=1)
        起 = cutoff - pd.DateOffset(months=TRAILING月)
        本季 = 本季治理
        权重 = {}
        for f, g in 本季.groupby("因子"):
            黄 = g.sort_values("date")["黄牌"].iloc[0]
            # iloc[:-1]:窗口末行(上季最后交易日)的"次日收益"=本季首日收益,
            # 而权重本季首日开盘即生效——该观测在决策时点不可得,必须剔除(审计修复)
            c = cos_ic(p表[f].loc[起:cutoff].iloc[:-1], 次日收益.loc[起:cutoff].iloc[:-1])
            if np.isfinite(c) and c > 0:
                权重[f] = c * (0.5 if 黄 else 1.0)
        总 = sum(权重.values())
        if 总 <= 0:
            continue
        减半 = 0.5 if len(权重) < E.烧机最少 else 1.0
        # 同质参照系:本季权重回算 [季前参照回看日, 季末]
        i0 = max(0, 日历.searchsorted(季日[0]) - 参照回看)
        扩 = 日历[i0:日历.searchsorted(季日[-1]) + 1]
        合 = pd.Series(0.0, index=扩)
        for f, w in 权重.items():
            合 = 合.add(信号[f].reindex(扩).fillna(0.0) * (w / 总), fill_value=0.0)
        合 = (合 * 减半).clip(-1, 1)
        comb定格 = pd.concat([comb定格, 合.loc[季日]])
        S增 = 合 + ΔS.reindex(扩).fillna(0.0)
        for 源, 收 in [(S增, "增"), (合, "价")]:
            πq = (源.rolling(映射窗).rank(pct=True) - 0.5 / 映射窗).clip(1e-6, 1 - 1e-6)
            πq = πq.where(源.abs().gt(1e-12))          # 信号为0处留NaN→策略层强制空仓
            if 收 == "增":
                π增 = pd.concat([π增, πq.loc[季日]])
            else:
                π价 = pd.concat([π价, πq.loc[季日]])
        print(f"  {q} 权重因子{len(权重)}", end="\r")

    comb定格 = comb定格[~comb定格.index.duplicated()]
    π增 = π增[~π增.index.duplicated()]; π价 = π价[~π价.index.duplicated()]

    fp1 = os.path.join(paths.存档, "分资产逐日.csv")
    df = pd.read_csv(fp1, index_col=0, parse_dates=True)
    # comb 要延伸到最新交易日(不能只 reindex 旧文件索引,否则新日 comb 缺失)
    新增 = comb定格.index.difference(df.index)
    if len(新增):
        df = df.reindex(df.index.union(comb定格.index))
    df["512100_仓位"] = comb定格.reindex(df.index)
    df.to_csv(fp1)
    fp2 = os.path.join(paths.存档, "映射分位.csv")
    pd.DataFrame({"π_增强": π增, "π_量价": π价}).rename_axis("date").to_csv(fp2)
    print(f"\n已写 {fp1}(512100_仓位=rank{因子窗} comb)与 {fp2}({len(π增)} 行)")
    print("请跑 python -m 引擎.策略 验证定稿成绩。")


if __name__ == "__main__":
    main()
