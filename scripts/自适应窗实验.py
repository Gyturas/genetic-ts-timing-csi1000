# -*- coding: utf-8 -*-
"""因子层自适应rank窗实验:W_i=clip(30*(1+ρ)/(1-ρ),20,250),ρ=trailing2年AC1,逐季刷新。
组合层映射窗/单仓层保持40。对照=统一40(须复现档案)。库:mine_open h1_s42/h5_s42。"""
import os, pickle
import numpy as np, pandas as pd
from scipy.stats import norm
import csi1000.engine.index_engine as E
import csi1000.ga_alpha.expr as gexpr
import csi1000.engine.strategy as st
from csi1000 import paths

指数, ETF, 映射窗, 参照回看 = "zz1000", "512100", 40, 60
E.输出目录 = paths.结果_26算子; E.安装GA()
from csi1000.walkforward.data import load_all
面板 = load_all()["classes"]["A股宽基"]["panels"]; 日历 = 面板["close"].index


def 单仓(S):
    π = (S.rolling(40).rank(pct=True) - 0.5/40).clip(1e-6, 1-1e-6)
    return pd.Series(np.tanh(norm.ppf(π)), index=S.index)


def cos_ic(p, r):
    d = pd.concat([p.rename("p"), r.rename("r")], axis=1).dropna()
    den = np.sqrt((d.p**2).sum() * (d.r**2).sum())
    return float((d.p*d.r).sum()/den) if den > 0 else np.nan


def 跑库(目录, h, 自适应):
    st_ = pickle.load(open(os.path.join(目录, "state.pkl"), "rb"))
    存 = pd.read_csv(os.path.join(目录, "因子逐日信号.csv.gz"), parse_dates=["date"])
    名 = pd.read_csv(os.path.join(目录, "名册.csv"))
    元老 = set(名[名.kind.isin(["elder", "argarch"])]["编号"])
    治理 = 存[(存.指数 == 指数) & (~存.因子.isin(元老))]
    r次 = 面板["open"][指数].pct_change(h, fill_method=None).shift(-(h+1))
    值, sign = {}, {}
    for f in sorted(治理.因子.unique()):
        m = st_["members"][f]
        面 = 面板 if m["专属"] is None else {k: p[[指数]] for k, p in 面板.items()}
        v = gexpr.evaluate(m["node"], 面).replace([np.inf, -np.inf], np.nan)
        值[f] = v[指数] if 指数 in v.columns else v.iloc[:, 0]
        sign[f] = m["sign"]
    π全 = pd.Series(dtype=float); 窗记 = []; ic记 = []
    for q in E.quarter_list("2015-01-01", str(日历[-1].date())):
        本 = 治理[(治理.date >= q.start_time) & (治理.date <= q.end_time)]
        if len(本) == 0: continue
        季日 = 日历[(日历 >= q.start_time) & (日历 <= q.end_time)]
        季日 = 季日[季日 >= 本.date.min()]
        cutoff = q.start_time - pd.Timedelta(days=1)
        权起 = cutoff - pd.DateOffset(months=12)
        i0 = max(0, 日历.searchsorted(季日[0]) - 参照回看)
        扩 = 日历[i0:日历.searchsorted(季日[-1]) + 1]
        权重, S缓 = {}, {}
        for f, g in 本.groupby("因子"):
            v = 值[f]
            if 自适应:
                vv = v.loc[cutoff - pd.DateOffset(years=2):cutoff].dropna()
                ρ = vv.autocorr(1) if len(vv) >= 250 else np.nan
                W = int(np.clip(30*(1+ρ)/(1-ρ), 20, 250)) if np.isfinite(ρ) and ρ < 1 else 40
            else:
                W = 40
            起算 = 权起 - pd.Timedelta(days=int((W + 40) * 1.7) + 20)   # 覆盖W+单仓40共80+行的交易日
            S = (v.loc[起算:季日[-1]].rolling(W).rank(pct=True)*2 - 1) * sign[f]
            S缓[f] = S
            黄 = g.sort_values("date")["黄牌"].iloc[0]
            c = cos_ic(单仓(S).loc[权起:cutoff].iloc[:-1], r次.loc[权起:cutoff].iloc[:-1])
            窗记.append((str(q), f, W)); 
            if np.isfinite(c): ic记.append((f, W, c))
            if np.isfinite(c) and c > 0:
                权重[f] = c * (0.5 if 黄 else 1.0)
        总 = sum(权重.values())
        if 总 <= 0: continue
        减半 = 0.5 if len(权重) < E.烧机最少 else 1.0
        合 = pd.Series(0.0, index=扩)
        for f, w in 权重.items():
            合 = 合.add(S缓[f].reindex(扩).fillna(0.0) * (w/总), fill_value=0.0)
        合 = (合 * 减半).clip(-1, 1)
        πq = (合.rolling(映射窗).rank(pct=True) - 0.5/映射窗).clip(1e-6, 1-1e-6)
        πq = πq.where(合.abs().gt(1e-12))
        π全 = pd.concat([π全, πq.loc[季日]])
    π全 = π全[~π全.index.duplicated()]
    p = pd.Series(np.tanh(norm.ppf(π全)), index=π全.index).where(π全.notna(), 0.0)
    r = st.读价(f"etf_{ETF}", "open").pct_change().reindex(p.index)
    rf = st.读价("etf_511880").pct_change().clip(lower=0).reindex(p.index).fillna(0)
    p = p.where(r.notna(), 0.0).fillna(0.0)
    ret = st.结算(p, r, rf, "open").dropna().loc["2018":]
    return ret, pd.DataFrame(窗记, columns=["q", "f", "W"]), pd.DataFrame(ic记, columns=["f", "W", "c"])


def 统(名, r, rf):
    nav = (1+r).cumprod(); 年 = len(r)/244; ann = nav.iloc[-1]**(1/年)-1
    ex = r - rf.reindex(r.index).fillna(0); dd = (nav/nav.cummax()-1).min()
    半 = r.index[len(r)//2]; ex后 = ex[ex.index >= 半]
    print(f"{名:24s} 年化{ann*100:5.1f}%  夏普{ex.mean()/ex.std()*np.sqrt(244):5.2f}  回撤{dd*100:6.1f}%  后半夏普{ex后.mean()/ex后.std()*np.sqrt(244):5.2f}", flush=True)
    return r


rf = st.读价("etf_511880").pct_change().clip(lower=0)
出 = {}
for lib, h in [("results/mine_open/h1", 1), ("results/mine_open/h5", 5)]:
    名 = os.path.basename(lib)
    for 自 in (False, True):
        tag = f"{名}_{'自适应' if 自 else '固定40'}"
        ret, 窗, ic = 跑库(lib, h, 自)
        出[tag] = ret
        统(tag, ret, rf)
        if 自:
            print(f"  窗分布: 中位{窗.W.median():.0f}  P25/P75 {窗.W.quantile(.25):.0f}/{窗.W.quantile(.75):.0f}  "
                  f"=250占比{(窗.W>=250).mean()*100:.0f}%  ≤40占比{(窗.W<=40).mean()*100:.0f}%", flush=True)
        else:
            旧 = pd.read_csv(f"docs/开盘挖矿实验/逐日收益/{名}_s42.csv", index_col=0, parse_dates=True).iloc[:, 0]
            差 = (ret - 旧.reindex(ret.index)).abs().max()
            print(f"  对照复现检查 vs 档案: 最大偏差 {差:.2e} {'✓' if 差 < 1e-10 else '✗不一致!'}", flush=True)
for 名t, keys in [("双库合奏·固定40", ["h1_固定40", "h5_固定40"]), ("双库合奏·自适应", ["h1_自适应", "h5_自适应"])]:
    统(名t, pd.concat([出[k] for k in keys], axis=1).dropna().mean(axis=1), rf)
pd.DataFrame({k: v for k, v in 出.items()}).to_csv("results/自适应窗实验_逐日.csv")
print("DONE", flush=True)
