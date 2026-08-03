# -*- coding: utf-8 -*-
"""方案①负区间重映射 · 公募等权篮子口径实验。

映射:s = tanh(Φ⁻¹(π)) ∈ (−1,1),中性仓 B、下限 F:
  s≥0: p = B + (1−B)·s        (长边)
  s<0: p = B + (B−F)·s        (空头强度分级减仓,p(−1)=F)
旧纯多 = clip(s,0,1)(B=0 特例,负半边全平)。
结算:等权篮子 open→open,双边万20,维护费 0.28%/年×暴露,可选不交易带。
用法: PYTHONPATH=src python3 scripts/公募映射实验.py
"""
import os
import numpy as np
import pandas as pd
from scipy.stats import norm

import csi1000.engine.strategy as st
from csi1000 import paths

万20 = 0.0020
维护日 = 0.0028 / 244


def 映射(s: pd.Series, B: float, F: float = 0.0) -> pd.Series:
    长 = B + (1 - B) * s
    短 = B + (B - F) * s
    return pd.Series(np.where(s >= 0, 长, 短), index=s.index).clip(F, 1.0)


def 部分调整(目标: pd.Series, λ: float) -> pd.Series:
    """Gârleanu-Pedersen 式部分调整:每日只向目标走 λ 步,换手随 λ 线性降。"""
    if λ >= 1:
        return 目标
    v = 目标.to_numpy(); q = np.empty_like(v); prev = 0.0
    for i, t in enumerate(v):
        if np.isfinite(t):
            prev = prev + λ * (t - prev)
        q[i] = prev
    return pd.Series(q, index=目标.index)


def 执行带(目标: pd.Series, 带: float) -> pd.Series:
    if 带 <= 0:
        return 目标
    v = 目标.to_numpy(); q = np.empty_like(v); prev = 0.0
    for i, t in enumerate(v):
        if np.isfinite(t) and abs(t - prev) > 带:
            prev = t
        q[i] = prev
    return pd.Series(q, index=目标.index)


def 结算EW(p: pd.Series, r: pd.Series, rf: pd.Series) -> pd.Series:
    rfi = rf.reindex(p.index).fillna(0)
    持 = p.shift(2).fillna(0)
    return (rfi + 持 * (r.reindex(p.index).fillna(0) - rfi)
            - p.diff().abs().shift(1).fillna(0) * 万20 - 持 * 维护日)


def 统(r: pd.Series, rf, p=None, 基=None) -> dict:
    r = r.dropna().loc["2018":]
    nav = (1 + r).cumprod(); 年 = len(r) / 244
    ann = nav.iloc[-1] ** (1 / 年) - 1
    ex = r - rf.reindex(r.index).fillna(0)
    dd = (nav / nav.cummax() - 1).min()
    o = dict(年化=round(ann * 100, 1), 夏普=round(float(ex.mean() / ex.std() * np.sqrt(244)), 2),
             回撤=round(dd * 100, 1), 卡玛=round(ann / abs(dd), 2))
    if p is not None:
        o["均仓"] = round(float(p.reindex(r.index).mean() * 100))
        o["年换手"] = round(float(p.diff().abs().reindex(r.index).mean() * 244 * 100))
    if 基 is not None:
        b = 基.reindex(r.index).fillna(0)
        bann = (1 + b).prod() ** (1 / 年) - 1
        o["超额"] = round((ann - bann) * 100, 1)
    return o


def main():
    r_ew = st.读价("etf_ew1000", "open").pct_change()
    rf = st.读价("etf_511880").pct_change().clip(lower=0)
    πs = {}
    for lib, h in [("h1", "results/mine_open/h1"), ("h5", "results/mine_open/h5")]:
        fp = os.path.join(paths.根, h, "π_zz1000.csv")
        πs[lib] = pd.read_csv(fp, index_col=0, parse_dates=True).iloc[:, 0]
    共 = πs["h1"].index.union(πs["h5"].index)
    s表 = {k: pd.Series(np.tanh(norm.ppf(v)), index=v.index).reindex(共)
          for k, v in πs.items()}
    s合 = pd.concat(s表, axis=1).mean(axis=1)          # h1+h5 双库合奏信号

    行 = []
    方案 = [("旧纯多 clip(B=0)", 0.0, 0.0, 0.0, 1.0),
          ("①remap B=0.3", 0.3, 0.0, 0.0, 1.0),
          ("①remap B=0.5", 0.5, 0.0, 0.0, 1.0),
          ("①B=0.7 F=0.6(偏股混合)", 0.7, 0.6, 0.02, 1.0),
          ("clip+部分调整λ=0.33(推荐)", 0.0, 0.0, 0.0, 0.33),
          ("clip+部分调整λ=0.2", 0.0, 0.0, 0.0, 0.20),
          ("偏股混合+部分调整λ=0.33", 0.7, 0.6, 0.0, 0.33)]
    for 名, B, F, 带, λ in 方案:
        s = s合.dropna()
        p = s.clip(0, 1) if B == 0 else 映射(s, B, F)
        p = 部分调整(p, λ)
        p = 执行带(p, 带)
        p = p.where(r_ew.reindex(p.index).notna(), 0.0).fillna(0.0)
        ret = 结算EW(p, r_ew, rf)
        o = {"方案": 名}; o.update(统(ret, rf, p, r_ew)); 行.append(o)
    # 基准
    b = r_ew.reindex(s合.dropna().index).fillna(0)
    o = {"方案": "等权篮子买入持有"}; o.update(统(b, rf)); o["均仓"] = 100; 行.append(o)

    t = pd.DataFrame(行)
    out = os.path.join(paths.根, "docs", "产品线", "公募等权篮子")
    os.makedirs(out, exist_ok=True)
    t.to_csv(os.path.join(out, "映射方案对比.csv"), index=False)
    print("═══ 方案① 负区间重映射 · 等权篮子(万20+维护费, h1+h5双库信号, 2018起)═══")
    print(t.to_string(index=False))

    # 最优方案分年
    s = s合.dropna(); p = 部分调整(s.clip(0, 1), 0.33)
    p = p.where(r_ew.reindex(p.index).notna(), 0.0).fillna(0.0)
    ret = 结算EW(p, r_ew, rf).dropna().loc["2018":]
    print("\n推荐方案(clip+λ0.33)分年(策略 vs 篮子持有):")
    for y in range(2018, 2027):
        a = ret.loc[str(y)]; bb = r_ew.reindex(ret.index).fillna(0).loc[str(y)]
        if len(a) > 20:
            print(f"  {y}: {(1+a).prod()-1:+7.1%}  vs 持有 {(1+bb).prod()-1:+7.1%}")


if __name__ == "__main__":
    main()
