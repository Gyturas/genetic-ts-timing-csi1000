# -*- coding: utf-8 -*-
"""v2 熊概率因子 · walk-forward elastic-net 逻辑回归(论文§3.4/A.4复刻)。

每季 cutoff 用 expanding 窗重估:块内|corr|前2筛选、中位插补、训练窗标准化、
类别平衡加权似然、C×α 网格按时序验证加权对数损失选、elastic-net 逻辑回归;
季内每日用该模型出 OOS 熊概率 p_t。全程严守 as-of(模型只见 <cutoff 数据)。

简化声明(相对论文):标签可靠性权重(规则×统计标签一致性)未实现——需第二套统计标签,
本版只用类别平衡权重(论文式37的 b 项,主导项);其余(块筛选/加权似然/TSCV调参)完整。
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import TimeSeriesSplit

warnings.filterwarnings("ignore")

from 引擎.路径 import 行情文件
from 引擎.regime_data import 建面板

覆盖门槛, 块内保留 = 0.80, 2
C网格 = [0.03, 0.10, 0.30, 1.00, 3.00]
α网格 = [0.0, 0.25, 0.50, 0.75, 1.00]
最少熊, 最少非熊 = 8, 30
验证块月 = 24


def _筛选(Xtr: pd.DataFrame, ytr: pd.Series, 块: dict) -> list:
    """块内保留|corr|前2且覆盖≥80%的列。"""
    选 = []
    for b, cols in 块.items():
        候选 = [c for c in cols if Xtr[c].notna().mean() >= 覆盖门槛]
        if not 候选:
            continue
        相关 = {c: abs(Xtr[c].corr(ytr)) for c in 候选}
        相关 = {c: (v if np.isfinite(v) else 0) for c, v in 相关.items()}
        选 += sorted(候选, key=lambda c: -相关[c])[:块内保留]
    return 选


def _拟合(Xtr, ytr, w):
    """C×α 网格按时序验证加权对数损失选,返回最优 LogisticRegression。"""
    from sklearn.metrics import log_loss
    n1, n0 = int(ytr.sum()), int((1 - ytr).sum())
    if n1 < 最少熊 or n0 < 最少非熊 or len(np.unique(ytr)) < 2:
        return None
    best, best_ll = None, np.inf
    tscv = TimeSeriesSplit(n_splits=3)
    idx = np.arange(len(ytr))
    for C in C网格:
        for α in α网格:
            lls = []
            for tr_i, va_i in tscv.split(idx):
                if len(np.unique(ytr.iloc[tr_i])) < 2:
                    continue
                m = LogisticRegression(penalty="elasticnet", l1_ratio=α, C=C,
                                       solver="saga", max_iter=2000, tol=1e-3)
                try:
                    m.fit(Xtr.iloc[tr_i], ytr.iloc[tr_i], sample_weight=w[tr_i])
                    pv = m.predict_proba(Xtr.iloc[va_i])[:, 1]
                    lls.append(log_loss(ytr.iloc[va_i], pv, labels=[0, 1],
                                        sample_weight=w[va_i]))
                except Exception:
                    pass
            if lls and np.mean(lls) < best_ll:
                best_ll, best = np.mean(lls), (C, α)
    C, α = best if best else (0.30, 0.50)
    m = LogisticRegression(penalty="elasticnet", l1_ratio=α, C=C,
                           solver="saga", max_iter=3000, tol=1e-4)
    m.fit(Xtr, ytr, sample_weight=w)
    return m


def 熊概率序列(基准指数="idx_sh000300", 结束="2026-07-16",
             起测="2015-01-01", log=print) -> pd.Series:
    """walk-forward 生成全日频 OOS 熊概率。季频重估,季内逐日预测。"""
    d = 建面板(基准指数, 结束)
    X, y, 块, 日历 = d["X"], d["y"], d["块"], d["日历"]
    季 = list(pd.period_range(起测, 结束, freq="Q"))
    out = pd.Series(index=日历, dtype=float)
    for q in 季:
        cutoff = 日历[日历 < q.start_time]
        if len(cutoff) < 500:
            continue
        cutoff = cutoff[-1]
        季日 = 日历[(日历 >= q.start_time) & (日历 <= q.end_time)]
        if len(季日) == 0:
            continue
        # 训练集:标签可得(需未来21日已实现)⇒ 截到 cutoff-21日
        训尾 = 日历[日历 <= cutoff]
        训尾 = 训尾[-22] if len(训尾) > 22 else cutoff
        mask = (X.index <= 训尾) & y.notna()
        Xtr全, ytr = X[mask], y[mask].astype(int)
        列 = _筛选(Xtr全, ytr, 块)
        if len(列) < 2:
            continue
        med = Xtr全[列].median()
        mu, sd = Xtr全[列].mean(), Xtr全[列].std().replace(0, 1)
        Xtr = ((Xtr全[列].fillna(med) - mu) / sd)
        n1, n0 = int(ytr.sum()), int((1 - ytr).sum())
        if n1 < 最少熊 or n0 < 最少非熊:
            continue
        w = np.where(ytr == 1, len(ytr) / (2 * n1), len(ytr) / (2 * n0))
        m = _拟合(Xtr, ytr, w)
        if m is None:
            continue
        Xq = ((X.loc[季日, 列].fillna(med) - mu) / sd)
        out.loc[季日] = m.predict_proba(Xq.fillna(0))[:, 1]
        if log:
            log(f"[熊因子] {q} 训练{len(ytr)}(熊{n1}) 特征{len(列)} 季均p={out.loc[季日].mean():.2f}")
    return out


if __name__ == "__main__":
    import os
    from scipy.stats import spearmanr
    p = 熊概率序列()
    结果 = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "结果")
    os.makedirs(结果, exist_ok=True)
    p.to_frame("p_bear").to_csv(os.path.join(结果, "bear_prob.csv"))
    # 诊断:分类指标 + 择时IC
    d = 建面板()
    y实 = d["状态"].reindex(p.index)
    j = pd.concat([p, y实], axis=1).dropna()
    pred = (j.iloc[:, 0] > 0.5).astype(int); 真 = j.iloc[:, 1].astype(int)
    acc = (pred == 真).mean()
    prec = (真[pred == 1].mean()) if (pred == 1).any() else np.nan
    rec = (pred[真 == 1].mean()) if (真 == 1).any() else np.nan
    print(f"\n分类(OOS): 准确率{acc:.0%} 精确率{prec:.0%} 召回率{rec:.0%} 熊基率{真.mean():.0%}")
    for tag, 名 in [("idx_sh000300","300"),("idx_sh000905","500"),("idx_sh000852","1000")]:
        r = pd.read_csv(行情文件(tag),
                        parse_dates=["date"]).set_index("date")["close"].pct_change()
        f5 = (r.shift(-5).rolling(5).sum()).reindex(p.index)  # 近似未来5日
        jj = pd.concat([-p, f5], axis=1).dropna().loc["2016":]
        ic = spearmanr(jj.iloc[:,0], jj.iloc[:,1])[0]
        print(f"  −p_bear 对 {名} 未来5日收益 IC(2016~): {ic:+.4f}  (正=高熊概率预示下跌)")
