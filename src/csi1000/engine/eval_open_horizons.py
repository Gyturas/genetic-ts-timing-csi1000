# -*- coding: utf-8 -*-
"""横向评估 results/mine_open/h*/ 各库(开盘口径挖矿,h=1..7)。

统一口径:v4 治理(季度 cos IC 加权 + 双40窗 + 同质参照系 + 剔元老),仓位逐日,
结算 = 次日开盘 open→open(strategy.结算 执行="open",滞后2格)。
加权用的前瞻收益与该库挖矿目标一致(h 日开盘前瞻),保证选出因子被按其所长加权;
最终结算一律逐日,7 个 h 可直接同表相比,并与 ops26 定稿基线(26.9%/1.25)对照。

产出: results/mine_open/横向对比.csv
用法: PYTHONPATH=src python -m csi1000.engine.eval_open_horizons [--only 3]
"""
from __future__ import annotations

import argparse
import glob
import os
import pickle

import numpy as np
import pandas as pd
from scipy.stats import norm

import csi1000.engine.index_engine as E
import csi1000.ga_alpha.expr as gexpr
import csi1000.engine.strategy as st
from csi1000 import paths

指数, ETF = "zz1000", "512100"
因子窗 = 映射窗 = 40
参照回看 = 60


def 单仓(S):
    π = (S.rolling(映射窗).rank(pct=True) - 0.5 / 映射窗).clip(1e-6, 1 - 1e-6)
    return pd.Series(np.tanh(norm.ppf(π)), index=S.index)


def cos_ic(p, r):
    d = pd.concat([p.rename("p"), r.rename("r")], axis=1).dropna()
    den = np.sqrt((d.p ** 2).sum() * (d.r ** 2).sum())
    return float((d.p * d.r).sum() / den) if den > 0 else np.nan


def 评一库(目录: str, h: int, 面板, 日历) -> dict | None:
    需 = [os.path.join(目录, f) for f in ("state.pkl", "因子逐日信号.csv.gz", "名册.csv")]
    if not all(os.path.exists(f) for f in 需):
        return None
    st_ = pickle.load(open(需[0], "rb"))
    存 = pd.read_csv(需[1], parse_dates=["date"])
    名 = pd.read_csv(需[2])
    元老 = set(名[名.kind.isin(["elder", "argarch"])]["编号"])
    治理 = 存[(存.指数 == 指数) & (~存.因子.isin(元老))]
    if len(治理) == 0:
        return None
    # 加权目标:与挖矿一致的 h 日开盘前瞻
    r次 = 面板["open"][指数].pct_change(h, fill_method=None).shift(-(h + 1))

    信号, p表 = {}, {}
    for f in sorted(治理.因子.unique()):
        m = st_["members"][f]
        面 = 面板 if m["专属"] is None else {k: p[[指数]] for k, p in 面板.items()}
        v = gexpr.evaluate(m["node"], 面).replace([np.inf, -np.inf], np.nan)
        v = v[指数] if 指数 in v.columns else v.iloc[:, 0]
        信号[f] = (v.rolling(因子窗).rank(pct=True) * 2 - 1) * m["sign"]
        p表[f] = 单仓(信号[f])

    π全 = pd.Series(dtype=float)
    for q in E.quarter_list("2015-01-01", str(日历[-1].date())):
        本 = 治理[(治理.date >= q.start_time) & (治理.date <= q.end_time)]
        if len(本) == 0:
            continue
        季日 = 日历[(日历 >= q.start_time) & (日历 <= q.end_time)]
        季日 = 季日[季日 >= 本.date.min()]
        cutoff = q.start_time - pd.Timedelta(days=1)
        起 = cutoff - pd.DateOffset(months=12)
        权重 = {}
        for f, g in 本.groupby("因子"):
            黄 = g.sort_values("date")["黄牌"].iloc[0]
            c = cos_ic(p表[f].loc[起:cutoff].iloc[:-1], r次.loc[起:cutoff].iloc[:-1])
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
    π全 = π全[~π全.index.duplicated()]

    p = pd.Series(np.tanh(norm.ppf(π全)), index=π全.index).where(π全.notna(), 0.0)
    r = st.读价(f"etf_{ETF}", "open").pct_change().reindex(p.index)
    rf = st.读价("etf_511880").pct_change().clip(lower=0).reindex(p.index).fillna(0)
    p = p.where(r.notna(), 0.0).fillna(0.0)
    ret = st.结算(p, r, rf, "open").dropna().loc["2018":]
    if len(ret) < 244:
        return None
    nav = (1 + ret).cumprod(); 年 = len(ret) / 244
    ann = nav.iloc[-1] ** (1 / 年) - 1
    dd = (nav / nav.cummax() - 1).min()
    ex = ret - rf.reindex(ret.index).fillna(0)
    return dict(h=h, 在役因子=治理.因子.nunique(), 年化=round(ann * 100, 1),
                夏普=round(float(ex.mean() / ex.std() * np.sqrt(244)), 2),
                回撤=round(dd * 100, 1), 卡玛=round(ann / abs(dd), 2),
                均仓=round(float(p.abs().mean() * 100)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", type=int, default=None)
    a = ap.parse_args()

    E.输出目录 = paths.结果_26算子      # 仅为 安装GA 初始化,不写入
    E.安装GA()
    from csi1000.walkforward.data import load_all
    面板 = load_all()["classes"]["A股宽基"]["panels"]
    日历 = 面板["close"].index

    根 = os.path.join(paths.根, "results", "mine_open")
    行 = []
    for 目录 in sorted(glob.glob(os.path.join(根, "h*"))):
        h = int(os.path.basename(目录)[1:])
        if a.only and h != a.only:
            continue
        o = 评一库(目录, h, 面板, 日历)
        print(f"h={h}: " + (str(o) if o else "产物不全/无在役因子,跳过"))
        if o:
            行.append(o)
    if 行:
        df = pd.DataFrame(行).sort_values("h")
        fp = os.path.join(根, "横向对比.csv")
        df.to_csv(fp, index=False)
        print("\n═══ 开盘口径挖矿 · h=1..7 横向(中证1000,逐日调仓,open→open,2018起)═══")
        print(df.to_string(index=False))
        print(f"\n基线(ops26 定稿,close口径挖矿+open结算): 年化26.9% 夏普1.25 卡玛1.44")
        print("已存:", fp)


if __name__ == "__main__":
    main()
