# -*- coding: utf-8 -*-
"""元老因子(13,无豁免):9个趋势(纯因果算子,直接pandas实现)+ 4个 AR(1)-GARCH(1,1)。

AR-GARCH as-of:每季 cutoff 重估参数(不收敛沿用上期),季内冻结逐日滤波,逐季持久化。
预热期(2013-2014)末定方向号,之后与 GA 因子同卷考核、同规退役。
"""
from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from walkforward import config as C


def _tsi(a: pd.DataFrame, w: int) -> pd.DataFrame:
    d = a.diff()
    r = d.rolling(w)
    return r.mean() / r.std().where(r.std() > 1e-12)


def _er(a: pd.DataFrame, w: int) -> pd.DataFrame:
    path = a.diff().abs().rolling(w).sum()
    return (a - a.shift(w)) / path.where(path > 1e-12)


def trend_elder_values(panels: dict) -> dict[str, pd.DataFrame]:
    c, h, low = panels["close"], panels["high"], panels["low"]
    rng = (c.rolling(60).max() - low.rolling(60).min())
    return {
        "元老:双均线20/120": c.rolling(20).mean() / c.rolling(120).mean(),
        "元老:年动量244":    c / c.shift(244),
        "元老:月动量20":     c / c.shift(20),
        "元老:通道位置60":   (c - low.rolling(60).min()) / rng.where(rng > 1e-12),
        "元老:MACD12/26":   (c.ewm(span=12, min_periods=12).mean()
                             - c.ewm(span=26, min_periods=26).mean()) / c,
        "元老:距52周高点":   c / h.rolling(244).max(),
        "元老:动量244扣20":  c.shift(20) / c.shift(244),
        "元老:TSI_120":     _tsi(c, 120),
        "元老:ER_120":      _er(c, 120),
    }


ARGARCH_NAMES = ("元老:AR系数", "元老:GARCH预测波动", "元老:标准化残差5日", "元老:风险调整预期")


class ArgarchRecorder:
    """逐季扩展并持久化 4 个 AR-GARCH 因子面板(date × 品种)。"""

    def __init__(self, ret_panel: pd.DataFrame, store: Path):
        self.ret = ret_panel * C.ARGARCH["ret_scale"]
        self.window = C.ARGARCH["refit_window"]
        self.min_obs = C.ARGARCH["min_obs"]
        self.store = Path(store)
        self.params: dict[str, pd.Series] = {}
        self.hist: dict[str, pd.DataFrame] = {
            n: pd.DataFrame(columns=ret_panel.columns, dtype=float) for n in ARGARCH_NAMES}
        self.store.mkdir(parents=True, exist_ok=True)

    def save(self) -> None:
        for i, n in enumerate(ARGARCH_NAMES):
            self.hist[n].to_csv(self.store / f"factor_{i}.csv")

    def load(self) -> bool:
        ok = True
        for i, n in enumerate(ARGARCH_NAMES):
            fp = self.store / f"factor_{i}.csv"
            if fp.exists():
                self.hist[n] = pd.read_csv(fp, index_col=0, parse_dates=True)
            else:
                ok = False
        return ok

    def extend(self, cutoff: pd.Timestamp, quarter_days: pd.DatetimeIndex) -> None:
        from arch import arch_model

        end = quarter_days.max()
        new = {n: pd.DataFrame(index=quarter_days, columns=self.ret.columns, dtype=float)
               for n in ARGARCH_NAMES}
        for asset in self.ret.columns:
            fit_series = self.ret[asset].loc[:cutoff].dropna().tail(self.window)
            params = None
            if len(fit_series) >= self.min_obs:
                try:
                    res = arch_model(fit_series, mean="AR", lags=1, vol="GARCH",
                                     p=1, q=1).fit(disp="off", show_warning=False)
                    if np.isfinite(res.params).all():
                        params = res.params
                except Exception:
                    params = None
            if params is None:
                params = self.params.get(asset)
            if params is None:
                continue
            self.params[asset] = params

            full = self.ret[asset].loc[:end].dropna().tail(self.window + len(quarter_days) + 10)
            try:
                fixed = arch_model(full, mean="AR", lags=1, vol="GARCH", p=1, q=1).fix(
                    np.asarray(params))
            except Exception:
                continue
            c, phi = params.iloc[0], params.iloc[1]
            omega, alpha, beta = params.iloc[2], params.iloc[3], params.iloc[4]
            sigma = pd.Series(fixed.conditional_volatility, index=full.index)
            resid = pd.Series(np.asarray(fixed.resid), index=full.index)
            sigma_next = np.sqrt((omega + alpha * resid**2 + beta * sigma**2).clip(lower=1e-12))
            mu_next = c + phi * full
            z5 = (resid / sigma).rolling(5).mean()
            vals = {
                "元老:AR系数": pd.Series(phi, index=full.index),
                "元老:GARCH预测波动": sigma_next,
                "元老:标准化残差5日": z5,
                "元老:风险调整预期": mu_next / sigma_next,
            }
            for n in ARGARCH_NAMES:
                new[n][asset] = vals[n].reindex(quarter_days)

        for n in ARGARCH_NAMES:
            keep = self.hist[n][~self.hist[n].index.isin(quarter_days)]
            self.hist[n] = pd.concat([keep, new[n]]).sort_index()

    def values(self, name: str) -> pd.DataFrame:
        return self.hist[name]
