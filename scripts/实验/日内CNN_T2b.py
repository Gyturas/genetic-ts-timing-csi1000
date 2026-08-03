# -*- coding: utf-8 -*-
"""T2b:个股训练 → 应用于个股 → 横截面聚合成市场信号(避免指数域偏移)。
关键区分:个股层cosIC=+0.04稳定为真,但直接套指数为负 → 检验该信号是特质还是含共同成分。"""
from __future__ import annotations
import os, glob, warnings, importlib.util
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
import torch
_s = importlib.util.spec_from_file_location("t2", os.path.join(os.path.dirname(__file__), "日内CNN_T2.py"))
t2m = importlib.util.module_from_spec(_s); _s.loader.exec_module(t2m)
t1, 造个股样本 = t2m.t1, t2m.造个股样本
dev = t1.dev

档 = sorted(glob.glob("data/cache/intraday/stocks/*.parquet"))
print(f"构造 {len(档)} 只个股样本…", flush=True)
股 = []
for fp in 档:
    s = 造个股样本(fp)
    if s: 股.append((os.path.basename(fp).replace(".parquet",""), s))
print(f"有效 {len(股)} 只", flush=True)
Xs=np.concatenate([s[1][0] for s in 股]); Gs=np.concatenate([s[1][1] for s in 股])
ys=np.concatenate([s[1][2] for s in 股]); ds=np.concatenate([s[1][3].values for s in 股])

聚合 = []
for yy in range(2021, 2027):
    m = ds < np.datetime64(f"{yy}-01-01")
    if m.sum() < 20000: continue
    种子日均 = []
    for seed in range(5):
        mdl, v = t1.训练一折(Xs[m], Gs[m], ys[m], ds[m], seed, epochs=40, bs=2048)
        逐股 = []
        for 名, (X,G,y,ix) in 股:
            k = (ix >= f"{yy}-01-01") & (ix < f"{yy+1}-01-01")
            if k.sum()==0: continue
            with torch.no_grad():
                p = mdl(torch.tensor(X[k],device=dev), torch.tensor(G[k],device=dev)).cpu().numpy()
            逐股.append(pd.Series(p, index=ix[k], name=名))
        种子日均.append(pd.concat(逐股, axis=1).mean(axis=1))     # 横截面均值=市场信号
    聚合.append(pd.concat(种子日均, axis=1).mean(axis=1))
    print(f"  {yy}: 完成,当年 {len(聚合[-1])} 天", flush=True)
sig = pd.concat(聚合).sort_index()
sig.rename("信号").to_csv("results/日内CNN_T2b截面聚合.csv")
print(f"DONE 截面聚合信号 {len(sig)} 天", flush=True)
