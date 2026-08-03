# -*- coding: utf-8 -*-
"""日内形态 1D-CNN(T1:六指数 pooled 训练)。

输入 (C=7, T=48):归一化收益/绝对收益/累计路径/量占比/同时刻量能异常/时间编码sin,cos
      + 2个全局标量(波动状态、隔夜收益)
结构:三层膨胀卷积(d=1,2,4,感受野29bar≈145分钟)+ Avg&Max双池化 + Dropout + tanh
损失:cos IC(与生产端因子加权口径同尺,吸取"两把尺反向"教训)
协议:walk-forward 逐年重训、每折5种子、验证集取训练期时间末20%、P&L判据

用法: PYTHONPATH=src python3 scripts/日内CNN.py
"""
from __future__ import annotations
import os, warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
import torch, torch.nn as nn

D = "data/cache/intraday"
指数表 = ["hs300", "zz500", "zz1000", "cyb", "kc50", "hongli"]
日线 = {"hs300":"idx_sh000300","zz500":"idx_sh000905","zz1000":"idx_sh000852",
       "cyb":"idx_sz399006","kc50":"idx_sh000688","hongli":"idx_sh000922"}
T, C = 48, 7
dev = "mps" if torch.backends.mps.is_available() else "cpu"


def 造样本(tag: str):
    """→ X(N,C,T), G(N,2), y(N,), 日期index。y = 次日开盘口径收益 open[t+2]/open[t+1]-1"""
    d = pd.read_parquet(f"{D}/{tag}_5m.parquet")
    d["时间"] = pd.to_datetime(d["时间"]); d["日"] = d["时间"].dt.date
    d = d[d["时间"].dt.time > pd.Timestamp("09:25").time()].copy()
    d["bar"] = d.groupby("日").cumcount()
    d = d[d.bar < T]
    d["r"] = d.groupby("日")["close"].pct_change().fillna(0) * 100
    d["vsh"] = d["vol"] / d.groupby("日")["vol"].transform("sum").replace(0, np.nan)
    铺 = lambda c: d.pivot_table(index="日", columns="bar", values=c).reindex(columns=range(T))
    R, V = 铺("r"), 铺("vsh")
    R.index = pd.to_datetime(R.index); V.index = pd.to_datetime(V.index)
    R = R.fillna(0.0).clip(-20, 20); V = V.fillna(1.0 / T)
    σ = R.std(axis=1).replace(0, np.nan)                       # 当日已实现波动
    rn = R.div(σ, axis=0).clip(-8, 8)                          # ①形状(去掉波动水平)
    an = rn.abs()                                              # ②绝对值(把平方喂进去)
    cum = rn.cumsum(axis=1) / np.sqrt(T)                       # ③累计路径
    vbar = V.rolling(20, min_periods=5).mean().shift(1)        # 同时刻量能基准(仅用过去)
    va = np.log((V / vbar).clip(0.05, 20)).fillna(0.0)         # ⑤剥离U形季节性
    pos = np.arange(T) / T
    s_ch = np.tile(np.sin(2 * np.pi * pos), (len(R), 1))
    c_ch = np.tile(np.cos(2 * np.pi * pos), (len(R), 1))
    X = np.stack([rn.values, an.values, cum.values, (V.values - 1/T) * T,
                  va.values, s_ch, c_ch], axis=1).astype("float32")
    # 全局标量:波动状态 + 隔夜收益
    σ20 = σ.rolling(20, min_periods=5).mean()
    g1 = np.log((σ / σ20).clip(0.2, 5)).fillna(0.0)
    dl = pd.read_csv(f"data/cache/{日线[tag]}.csv", parse_dates=["date"]).set_index("date")
    g2 = (dl["open"] / dl["close"].shift(1) - 1).reindex(R.index).fillna(0.0) * 100
    G = np.stack([g1.values, g2.values], axis=1).astype("float32")
    # 目标:次日开盘口径(与生产结算一致)
    y = (dl["open"].shift(-2) / dl["open"].shift(-1) - 1).reindex(R.index) * 100
    ok = np.isfinite(X).all(axis=(1, 2)) & np.isfinite(G).all(axis=1) & y.notna().values & σ.notna().values
    return X[ok], G[ok], y.values[ok].astype("float32"), R.index[ok]


class 日内CNN(nn.Module):
    def __init__(self, C=C, h=16, p=0.3):
        super().__init__()
        def blk(i, o, d):
            return nn.Sequential(nn.Conv1d(i, o, 5, padding=2*d, dilation=d),
                                 nn.BatchNorm1d(o), nn.GELU())
        self.c1, self.c2, self.c3 = blk(C, h, 1), blk(h, h, 2), blk(h, h, 4)
        self.drop = nn.Dropout(p)
        self.fc = nn.Linear(2 * h + 2, 1)

    def forward(self, x, g):
        z = self.c3(self.c2(self.c1(x)))
        z = torch.cat([z.mean(-1), z.amax(-1), g], dim=1)
        return torch.tanh(self.fc(self.drop(z))).squeeze(-1)


def cosic损失(p, y):
    return -(p * y).sum() / (p.pow(2).sum().sqrt() * y.pow(2).sum().sqrt() + 1e-9)


def 训练一折(Xtr, Gtr, ytr, dtr, seed, epochs=60, bs=1024):
    """dtr: 每个样本的日期(用于按时间划验证集,避免跨指数同期泄漏)"""
    torch.manual_seed(seed); np.random.seed(seed)
    分界 = pd.Series(dtr).quantile(0.8)                          # 训练期时间末20%为验证
    tr = np.where(dtr <= 分界)[0]; va = np.where(dtr > 分界)[0]
    m = 日内CNN().to(dev)
    opt = torch.optim.AdamW(m.parameters(), lr=2e-3, weight_decay=1e-3)
    X_ = torch.tensor(Xtr, device=dev); G_ = torch.tensor(Gtr, device=dev); y_ = torch.tensor(ytr, device=dev)
    最好, 最好态, 耐心 = -9, None, 0
    for ep in range(epochs):
        m.train(); perm = np.random.permutation(tr)
        for i in range(0, len(perm), bs):
            b = perm[i:i+bs]
            if len(b) < 64: continue
            opt.zero_grad()
            loss = cosic损失(m(X_[b], G_[b]), y_[b])
            loss.backward(); opt.step()
        m.eval()
        with torch.no_grad():
            v = -cosic损失(m(X_[va], G_[va]), y_[va]).item()
        if v > 最好:
            最好, 最好态, 耐心 = v, {k: t.clone() for k, t in m.state_dict().items()}, 0
        else:
            耐心 += 1
            if 耐心 >= 12: break
    m.load_state_dict(最好态); m.eval()
    return m, 最好


def main():
    数据 = {t: 造样本(t) for t in 指数表 if os.path.exists(f"{D}/{t}_5m.parquet")}
    print(f"载入 {len(数据)} 个指数: " + "  ".join(f"{t}={len(v[2])}天" for t, v in 数据.items()), flush=True)
    总 = sum(len(v[2]) for v in 数据.values())
    print(f"pooled 样本量 {总}  参数量 {sum(p.numel() for p in 日内CNN().parameters())}  比例 {总/sum(p.numel() for p in 日内CNN().parameters()):.1f}:1", flush=True)
    预 = {t: [] for t in 数据}
    for yy in range(2021, 2027):
        Xtr = np.concatenate([v[0][v[3] < f"{yy}-01-01"] for v in 数据.values()])
        Gtr = np.concatenate([v[1][v[3] < f"{yy}-01-01"] for v in 数据.values()])
        ytr = np.concatenate([v[2][v[3] < f"{yy}-01-01"] for v in 数据.values()])
        dtr = np.concatenate([v[3][v[3] < f"{yy}-01-01"].values for v in 数据.values()])
        if len(ytr) < 2000: continue
        种子预 = {t: [] for t in 数据}
        vs = []
        for seed in range(5):
            m, v = 训练一折(Xtr, Gtr, ytr, dtr, seed); vs.append(v)
            for t, (X, G, y, ix) in 数据.items():
                msk = (ix >= f"{yy}-01-01") & (ix < f"{yy+1}-01-01")
                if msk.sum() == 0: continue
                with torch.no_grad():
                    p = m(torch.tensor(X[msk], device=dev), torch.tensor(G[msk], device=dev)).cpu().numpy()
                种子预[t].append(pd.Series(p, index=ix[msk]))
        for t in 数据:
            if 种子预[t]: 预[t].append(pd.concat(种子预[t], axis=1).mean(axis=1))
        print(f"  {yy}: 训练{len(ytr)}样本  验证cosIC(5种子) {np.mean(vs):+.4f}±{np.std(vs):.4f}", flush=True)
    out = {t: pd.concat(v).sort_index() for t, v in 预.items() if v}
    pd.DataFrame(out).to_csv("results/日内CNN_信号v2.csv")
    print(f"\n信号已存 results/日内CNN_信号.csv  {len(out['zz1000'])}天", flush=True)


if __name__ == "__main__":
    main()
