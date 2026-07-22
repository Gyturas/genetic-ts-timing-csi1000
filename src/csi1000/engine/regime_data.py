# -*- coding: utf-8 -*-
"""v2 熊概率因子 · 特征面板 + 熊态标签(论文§3.4/A.4口径)。

全部特征滞后到形成日 t(shift 1),标签为未来21日是否处于熊态(仅训练用)。
熊态定义:较250日滚动峰值回撤>8% 视为进入,恢复到 (滚动峰值×0.97) 以上视为退出(迟滞)。
特征分块保留原始列(标准化/筛选在 bear_factor 内按训练窗做,严守 as-of)。
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

from csi1000 import paths  # 行情目录

缓存 = 行情目录()          # 优先仓库自带 数据/行情/,缺失回退父项目 data_cache

摆动阈 = 0.12          # Pagan-Sossounov:自峰跌12%转熊,自谷涨12%转牛
标签前瞻 = 21


def _读(tag, 列=None):
    df = pd.read_csv(os.path.join(缓存, tag + ".csv"), parse_dates=["date"]).set_index("date").sort_index()
    return df[~df.index.duplicated()] if 列 is None else df[[列]][~df.index.duplicated()]


def 熊态标签(close: pd.Series) -> pd.Series:
    """Pagan-Sossounov 转折点定态:牛市中价跌破running峰×(1−12%)→转熊(自峰日起标熊);
    熊市中价涨破running谷×(1+12%)→转牛(自谷日起标牛)。返回0/1状态(1=熊)。"""
    c = close.to_numpy(); n = len(c)
    状态 = np.zeros(n, dtype=int)
    在熊 = False; 峰 = c[0]; 峰i = 0; 谷 = c[0]; 谷i = 0
    for i in range(n):
        if not 在熊:
            if c[i] > 峰:
                峰 = c[i]; 峰i = i
            elif c[i] < 峰 * (1 - 摆动阈):        # 确认转熊,回填峰→今为熊
                状态[峰i:i + 1] = 1
                在熊 = True; 谷 = c[i]; 谷i = i
        else:
            if c[i] < 谷:
                谷 = c[i]; 谷i = i
            elif c[i] > 谷 * (1 + 摆动阈):        # 确认转牛,谷→今为牛
                状态[谷i:i + 1] = 0
                在熊 = False; 峰 = c[i]; 峰i = i
    if 在熊:
        状态[峰i:] = 1
    return pd.Series(状态, index=close.index)


def 建面板(基准指数="idx_sh000300", 结束="2026-07-16") -> dict:
    """返回 {日历, X(特征DataFrame,原始未标准化), y(未来21日熊标签), 块映射, close}。"""
    px = _读(基准指数, "close")["close"].loc[:结束]
    日历 = px.index
    r = px.pct_change()
    F = {}

    # 收益块
    F["收益_r21"] = px.pct_change(21); F["收益_r63"] = px.pct_change(63); F["收益_r126"] = px.pct_change(126)
    # 波动块
    F["波动_rv21"] = r.rolling(21).std(); F["波动_rv63"] = r.rolling(63).std()
    F["波动_下行21"] = r.clip(upper=0).rolling(21).std()
    # 回撤块
    F["回撤_peak250"] = px / px.rolling(250).max() - 1
    F["回撤_peak63"] = px / px.rolling(63).max() - 1

    def 对齐(s):
        return s.reindex(日历).ffill()

    # 隐波块(iVIX,2015~)
    try:
        iv = _读("regime_ivix", "ivix")["ivix"]
        F["隐波_ivix"] = 对齐(iv); F["隐波_ivix变化21"] = 对齐(iv).pct_change(21)
    except Exception:
        pass
    # 信用块(企债-国债 价格比代理利差;涨=利差收窄)
    try:
        gz = _读("idx_sh000012", "close")["close"]; qz = _读("idx_sh000013", "close")["close"]
        spread = 对齐(qz) / 对齐(gz)
        F["信用_利差代理"] = spread; F["信用_利差变化21"] = spread.pct_change(21)
    except Exception:
        pass
    # 利率/曲线块
    try:
        yy = _读("bond_yields")
        F["利率_期限利差"] = 对齐(yy["y10"] - yy["y2"])
    except Exception:
        pass
    try:
        sh = _读("regime_shibor", "shibor3m")["shibor3m"]
        F["利率_shibor3m"] = 对齐(sh); F["利率_shibor变化21"] = 对齐(sh).diff(21)
    except Exception:
        pass
    # 资金块(北向,2014~)
    try:
        nb = _读("regime_northbound", "northbound")["northbound"]
        F["资金_北向21"] = 对齐(nb).rolling(21).sum(); F["资金_北向63"] = 对齐(nb).rolling(63).sum()
    except Exception:
        pass

    X = pd.DataFrame(F).shift(1)                       # 全部滞后到形成日
    状态 = 熊态标签(px)
    y = 状态.shift(-标签前瞻)                          # 未来21日是否熊态(训练目标)
    块 = {}
    for c in X.columns:
        块.setdefault(c.split("_")[0], []).append(c)
    return {"日历": 日历, "X": X, "y": y, "块": 块, "close": px, "状态": 状态}


if __name__ == "__main__":
    d = 建面板()
    print("日历:", d["日历"][0].date(), "~", d["日历"][-1].date(), len(d["日历"]))
    print("特征块:")
    for b, cols in d["块"].items():
        cov = d["X"][cols].notna().mean().min()
        起 = d["X"][cols].dropna(how="all").index[0].date()
        print(f"  {b:4s} {len(cols)}列 起{起} 最低覆盖{cov:.0%}: {[c.split('_',1)[1] for c in cols]}")
    print(f"\n熊态占比(全史): {d['状态'].mean():.1%}   熊态段数: {int((d['状态'].diff()==1).sum())}")
    近 = d["状态"].loc["2020":]
    print("各年熊态占比:", {str(y): f"{d['状态'].loc[str(y)].mean():.0%}" for y in range(2018,2027)})
