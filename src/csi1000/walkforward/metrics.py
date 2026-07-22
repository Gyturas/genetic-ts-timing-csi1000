# -*- coding: utf-8 -*-
"""因子评价:记账原子=日对(S_t, r_{t+1}-rf_{t+1})。全部指标定义于此,入库/体检/选用共用。

口径:
- 信号 S = 方向号 × z250(因子值),方向号入库时冻结;
- 月度IC = 当月日对 Spearman(月内有效对≥8),类均IC = 跨品种平均;
- NW t: 月度类均IC序列的 Newey-West t(lag=3);
- ICIR: trailing 36 个月度类均IC 的 mean/std;
- 快分量IC: (S − ema20(S)) 的日对IC;
- 事件收益: 窗口内 S 前20% 日的次日超额,双侧去尾5%后取均值;
- 残差IC: 对现役中与候选相关最高的≤6个信号 OLS 取残差再算IC。
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore", category=stats.ConstantInputWarning)

from csi1000.walkforward import config as C


# ---------------- 基础 ----------------

def fwd_excess(returns: pd.DataFrame, rf: pd.Series) -> pd.DataFrame:
    """fwd_t = r_{t+1} − rf_{t+1}(末日为 NaN,天然 as-of 安全)。"""
    return returns.sub(rf, axis=0).shift(-1)


def zsig(values: pd.DataFrame, sign: float = 1.0) -> pd.DataFrame:
    r = values.rolling(C.Z_WIN, min_periods=C.Z_WIN // 2)
    z = (values - r.mean()) / r.std().where(r.std() > 1e-12)
    return sign * z.clip(-5, 5)


def _spearman_cols(S: pd.DataFrame, F: pd.DataFrame, min_n: int = 30) -> pd.Series:
    out = {}
    for c in S.columns:
        j = pd.concat([S[c], F[c]], axis=1).dropna()
        out[c] = j.iloc[:, 0].corr(j.iloc[:, 1], method="spearman") if len(j) >= min_n else np.nan
    return pd.Series(out)


def daily_ic(S: pd.DataFrame, F: pd.DataFrame) -> pd.Series:
    """窗口内逐品种日对 Spearman IC。"""
    return _spearman_cols(S, F)


def monthly_ic_series(S: pd.DataFrame, F: pd.DataFrame) -> pd.Series:
    """月度类均IC序列:逐品种逐月 spearman → 跨品种平均。"""
    rows = {}
    for (y, m), idx in S.groupby([S.index.year, S.index.month]).groups.items():
        s, f = S.loc[idx], F.loc[idx]
        ics = []
        for c in s.columns:
            j = pd.concat([s[c], f[c]], axis=1).dropna()
            if len(j) >= 8:
                ics.append(j.iloc[:, 0].corr(j.iloc[:, 1], method="spearman"))
        if ics:
            rows[pd.Timestamp(int(y), int(m), 1)] = float(np.nanmean(ics))
    return pd.Series(rows).sort_index()


def nw_t(x: pd.Series, lags: int = 3) -> float:
    x = x.dropna().to_numpy(dtype=float)
    n = len(x)
    if n < 8:
        return np.nan
    mu = x.mean()
    e = x - mu
    g0 = float(e @ e) / n
    lrv = g0
    for k in range(1, min(lags, n - 1) + 1):
        gk = float(e[k:] @ e[:-k]) / n
        lrv += 2 * (1 - k / (lags + 1)) * gk
    if lrv <= 0:
        return np.nan
    return mu / np.sqrt(lrv / n)


# ---------------- 记分卡 ----------------

def scorecard(S: pd.DataFrame, F: pd.DataFrame) -> dict:
    """在给定窗口(S/F已切片)上算全套指标。"""
    per_ic = daily_ic(S, F)
    mic = monthly_ic_series(S, F)
    fast = S - S.ewm(span=20, min_periods=20).mean()
    fast_ic = float(daily_ic(fast, F).mean())
    pnl = float(np.nanmean((S * F).to_numpy()))
    hits, bases, events, dsc = [], [], [], []
    for c in S.columns:
        j = pd.concat([S[c], F[c]], axis=1).dropna()
        if len(j) < 60:
            continue
        s, f = j.iloc[:, 0], j.iloc[:, 1]
        thr = s.quantile(1 - C.EVENT_TOP)
        top = f[s >= thr]
        if len(top) >= 10:
            bases.append(float((f > 0).mean()))
            hits.append(float((top > 0).mean()))
            lo, hi = top.quantile(C.EVENT_TRIM), top.quantile(1 - C.EVENT_TRIM)
            events.append(float(top.clip(lo, hi).mean()))
        ds = s.diff()
        j2 = pd.concat([ds, f], axis=1).dropna()
        if len(j2) >= 60:
            dsc.append(float(j2.iloc[:, 0].corr(j2.iloc[:, 1])))
    return {
        "ic": float(per_ic.mean()), "ic_per": per_ic,
        "xs_pos": float((per_ic.dropna() > 0).mean()) if per_ic.notna().any() else np.nan,
        "mic": mic, "t": nw_t(mic), "fast_ic": fast_ic, "pnl": pnl,
        "hit_gain": float(np.mean(hits) - np.mean(bases)) if hits else np.nan,
        "event": float(np.mean(events)) if events else np.nan,
        "dscorr": float(np.mean(dsc)) if dsc else np.nan,
    }


def icir36(values: pd.DataFrame, sign: float, F_full: pd.DataFrame,
           cutoff: pd.Timestamp) -> float:
    """trailing 36 个月度类均IC 的 mean/std(不足36月记 NaN)。"""
    S = zsig(values, sign).loc[:cutoff].tail(C.ICIR_MONTHS * 22 + C.Z_WIN)
    mic = monthly_ic_series(S.tail(C.ICIR_MONTHS * 22), F_full.loc[S.index].tail(C.ICIR_MONTHS * 22))
    mic = mic.tail(C.ICIR_MONTHS)
    if len(mic) < C.ICIR_MONTHS or mic.std() < 1e-12:
        return np.nan
    return float(mic.mean() / mic.std())


# ---------------- 行为家族 ----------------

def archetypes(panels: dict) -> dict[str, pd.DataFrame]:
    close, ret, vol = panels["close"], panels["returns"], panels["volume"]
    return {
        "动量": close.pct_change(20, fill_method=None),
        "反转": -close.pct_change(5, fill_method=None),
        "波动": ret.rolling(20).std(),
        "量能": zsig(vol),
    }


def family_tag(S: pd.DataFrame, arch: dict[str, pd.DataFrame]) -> str:
    best_name, best = "其他", C.FAMILY_TAG_MIN
    for name, a in arch.items():
        c = _spearman_cols(S, a.reindex(S.index)).abs().mean()
        if np.isfinite(c) and c > best:
            best_name, best = name, float(c)
    return best_name


# ---------------- 残差与相关 ----------------

def max_corr_vs(S: pd.DataFrame, actives: list[pd.DataFrame]) -> float:
    if not actives:
        return 0.0
    return max(abs(_spearman_cols(S, A.reindex(S.index)).mean()) for A in actives)


def residual_ic(S: pd.DataFrame, F: pd.DataFrame, actives: list[pd.DataFrame]) -> float:
    if not actives:
        j = daily_ic(S, F)
        return float(j.mean())
    actives = [A.reindex(S.index) for A in actives]
    corrs = [(abs(_spearman_cols(S, A).mean()), A) for A in actives]
    base = [A for _, A in sorted(corrs, key=lambda x: -x[0])[:6]]
    resid = pd.DataFrame(index=S.index, columns=S.columns, dtype=float)
    for c in S.columns:
        X = pd.concat([A[c] for A in base], axis=1)
        j = pd.concat([S[c], X], axis=1).dropna()
        if len(j) < 60:
            continue
        y, x = j.iloc[:, 0].to_numpy(), j.iloc[:, 1:].to_numpy()
        x = np.column_stack([np.ones(len(x)), x])
        try:
            beta, *_ = np.linalg.lstsq(x, y, rcond=None)
            resid.loc[j.index, c] = y - x @ beta
        except np.linalg.LinAlgError:
            continue
    return float(daily_ic(resid, F).mean())
