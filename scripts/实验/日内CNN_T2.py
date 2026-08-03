# -*- coding: utf-8 -*-
"""T2:个股 pooled 训练日内 CNN,迁移到指数。

与 T1 唯一的差别是训练集:150 只个股 × 2019-2026(约 25 万样本,参数比 ~75:1)
而非 6 个高度相关的指数(1.1 万样本,3.5:1)。模型/损失/协议全部沿用 T1 规格。
核心检验:在个股上训练的编码器,能否迁移到中证1000 指数上产生 P&L。

用法: PYTHONPATH=src python3 scripts/日内CNN_T2.py
"""
from __future__ import annotations
import os, glob, warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
import torch

import importlib.util
_spec = importlib.util.spec_from_file_location("t1", os.path.join(os.path.dirname(__file__), "日内CNN.py"))
t1 = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(t1)
日内CNN, cosic损失, 造样本, T, C, dev = t1.日内CNN, t1.cosic损失, t1.造样本, t1.T, t1.C, t1.dev

股目录 = "data/cache/intraday/stocks"


def 造个股样本(fp: str):
    """个股版:目标用个股自身次日开盘口径收益(日线由5m聚合,避免额外数据依赖)。"""
    d = pd.read_parquet(fp)
    d["时间"] = pd.to_datetime(d["时间"]); d["日"] = d["时间"].dt.date
    d = d[d["时间"].dt.time > pd.Timestamp("09:25").time()].copy()
    d["bar"] = d.groupby("日").cumcount(); d = d[d.bar < T]
    if d["日"].nunique() < 300: return None
    d["r"] = d.groupby("日")["close"].pct_change().fillna(0) * 100
    d["vsh"] = d["vol"] / d.groupby("日")["vol"].transform("sum").replace(0, np.nan)
    铺 = lambda c: d.pivot_table(index="日", columns="bar", values=c).reindex(columns=range(T))
    R, V = 铺("r"), 铺("vsh")
    R.index = pd.to_datetime(R.index); V.index = pd.to_datetime(V.index)
    R = R.fillna(0.0).clip(-20, 20); V = V.fillna(1.0 / T)
    σ = R.std(axis=1).replace(0, np.nan)
    rn = R.div(σ, axis=0).clip(-8, 8); an = rn.abs()
    cum = rn.cumsum(axis=1) / np.sqrt(T)
    vbar = V.rolling(20, min_periods=5).mean().shift(1)
    va = np.log((V / vbar).clip(0.05, 20)).fillna(0.0)
    pos = np.arange(T) / T
    s_ch = np.tile(np.sin(2*np.pi*pos), (len(R), 1)); c_ch = np.tile(np.cos(2*np.pi*pos), (len(R), 1))
    X = np.stack([rn.values, an.values, cum.values, (V.values - 1/T)*T, va.values, s_ch, c_ch],
                 axis=1).astype("float32")
    # 日线(由5m聚合):开盘=首bar开,收盘=末bar收
    日 = d.groupby("日").agg(开=("open","first"), 收=("close","last"))
    日.index = pd.to_datetime(日.index)
    日 = 日.reindex(R.index)
    σ20 = σ.rolling(20, min_periods=5).mean()
    g1 = np.log((σ / σ20).clip(0.2, 5)).fillna(0.0)
    g2 = (日["开"] / 日["收"].shift(1) - 1).fillna(0.0) * 100
    G = np.stack([g1.values, g2.values], axis=1).astype("float32")
    y = (日["开"].shift(-2) / 日["开"].shift(-1) - 1) * 100
    y = y.clip(-15, 15)                                  # 个股极端值截断(涨跌停/异常)
    ok = (np.isfinite(X).all(axis=(1,2)) & np.isfinite(G).all(axis=1)
          & y.notna().values & σ.notna().values)
    if ok.sum() < 200: return None
    return X[ok], G[ok], y.values[ok].astype("float32"), R.index[ok]


def main():
    档 = sorted(glob.glob(f"{股目录}/*.parquet"))
    print(f"个股文件 {len(档)} 个,构造样本…", flush=True)
    股 = []
    for i, fp in enumerate(档):
        s = 造个股样本(fp)
        if s: 股.append(s)
        if (i+1) % 30 == 0: print(f"  {i+1}/{len(档)} 已处理,有效{len(股)}", flush=True)
    Xs = np.concatenate([s[0] for s in 股]); Gs = np.concatenate([s[1] for s in 股])
    ys = np.concatenate([s[2] for s in 股]); ds = np.concatenate([s[3].values for s in 股])
    npar = sum(p.numel() for p in 日内CNN().parameters())
    print(f"\n个股 pooled: {len(ys):,} 样本 / {len(股)} 只  参数{npar}  比例 {len(ys)/npar:.0f}:1", flush=True)
    print(f"(对照 T1 六指数: 11,401 样本,比例 3.5:1)", flush=True)

    指数数据 = {t: 造样本(t) for t in t1.指数表 if os.path.exists(f"{t1.D}/{t}_5m.parquet")}
    预 = {t: [] for t in 指数数据}
    诊断 = []
    for yy in range(2021, 2027):
        m股 = ds < np.datetime64(f"{yy}-01-01")
        if m股.sum() < 20000: continue
        Xtr, Gtr, ytr, dtr = Xs[m股], Gs[m股], ys[m股], ds[m股]
        种子预 = {t: [] for t in 指数数据}; vs = []
        for seed in range(5):
            m, v = t1.训练一折(Xtr, Gtr, ytr, dtr, seed, epochs=40, bs=2048)
            vs.append(v)
            for t, (X, G, y, ix) in 指数数据.items():
                msk = (ix >= f"{yy}-01-01") & (ix < f"{yy+1}-01-01")
                if msk.sum() == 0: continue
                with torch.no_grad():
                    p = m(torch.tensor(X[msk], device=dev), torch.tensor(G[msk], device=dev)).cpu().numpy()
                种子预[t].append(pd.Series(p, index=ix[msk]))
        for t in 指数数据:
            if 种子预[t]: 预[t].append(pd.concat(种子预[t], axis=1).mean(axis=1))
        # 迁移诊断:同一批模型在指数上的当年 cos IC
        zz = pd.concat(种子预["zz1000"], axis=1).mean(axis=1)
        yzz = pd.Series(指数数据["zz1000"][2], index=指数数据["zz1000"][3]).reindex(zz.index)
        c = float((zz*yzz).sum()/np.sqrt((zz**2).sum()*(yzz**2).sum()))
        诊断.append((yy, np.mean(vs), c))
        print(f"  {yy}: 训练{m股.sum():,}样本  个股验证cosIC {np.mean(vs):+.4f}±{np.std(vs):.4f}  "
              f"→ 迁移到zz1000当年cosIC {c:+.4f}", flush=True)
    out = {t: pd.concat(v).sort_index() for t, v in 预.items() if v}
    pd.DataFrame(out).to_csv("results/日内CNN_T2信号.csv")
    print(f"\n信号已存 results/日内CNN_T2信号.csv  zz1000 {len(out['zz1000'])}天", flush=True)


if __name__ == "__main__":
    main()
