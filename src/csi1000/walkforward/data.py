# -*- coding: utf-8 -*-
"""面板构建:全部对齐 A股交易日历(以000300日历为准)。

细则:
- 缺日按 as-of 前向对齐(reindex ffill),上市前保持 NaN;
- 跨境类面板序列整体 +1 个A股交易日(其收盘时点晚于/等于A股收盘,不 +1 即前视);
  例外:来源是 A股场内基金(etf_*)的品种(原油LOF)不错位;
- amount 缺失(期货/海外指数)用 close×volume 代理;
- entropy120: 收益率120日滚动等宽10箱直方图熵; beta120: 对本类基准120日滚动β;
- returns 用 ffill 后 close 的 pct_change(缺日=0收益)。
"""
from __future__ import annotations

import os
import pickle
import warnings

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

from csi1000.walkforward import config as C


def _read(tag: str) -> pd.DataFrame:
    fp = os.path.join(C.CACHE, tag + ".csv")
    df = pd.read_csv(fp, parse_dates=["date"]).set_index("date").sort_index()
    return df[~df.index.duplicated()]


def _entropy120(ret: pd.Series, win: int = 120, bins: int = 10) -> pd.Series:
    x = ret.to_numpy(dtype=float)
    out = np.full(len(x), np.nan)
    if len(x) < win:
        return pd.Series(out, index=ret.index)
    sw = sliding_window_view(x, win)                       # (T-win+1, win)
    ok = ~np.isnan(sw).any(axis=1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        lo = np.nanmin(sw, axis=1, keepdims=True)
        hi = np.nanmax(sw, axis=1, keepdims=True)
        span = np.where(hi - lo < 1e-12, 1.0, hi - lo)
        idx = np.clip(np.nan_to_num((sw - lo) / span * bins).astype(int), 0, bins - 1)
    ent = np.zeros(len(sw))
    for b in range(bins):
        p = (idx == b).sum(axis=1) / win
        with np.errstate(divide="ignore", invalid="ignore"):
            ent -= np.where(p > 0, p * np.log(p), 0.0)
    ent[~ok] = np.nan
    out[win - 1:] = ent
    return pd.Series(out, index=ret.index)


def build_all(save: bool = True) -> dict:
    cal = _read("idx_sh000300").index                       # 主日历
    cal = cal[(cal >= "2004-01-01") & (cal <= C.END)]

    def align(tag: str, col: str, shift: bool) -> pd.Series:
        raw = _read(tag)
        s = raw[col] if col in raw.columns else pd.Series(np.nan, index=raw.index)
        s = pd.to_numeric(s, errors="coerce")
        s = s.reindex(cal, method="ffill")
        return s.shift(1) if shift else s

    classes = {}
    for cname, spec in CLASSES_ITEMS():
        insts = list(spec["panel"])
        panels: dict[str, pd.DataFrame] = {}
        for leaf in ("open", "high", "low", "close", "volume", "amount"):
            cols = {}
            for inst in insts:
                tag = spec["panel"][inst]
                sh = spec["offshore"] and not tag.startswith("etf_")
                cols[inst] = align(tag, leaf, sh)
            panels[leaf] = pd.DataFrame(cols)
        # amount 代理
        miss = panels["amount"].isna().all()
        for inst in miss[miss].index:
            panels["amount"][inst] = panels["close"][inst] * panels["volume"][inst]
        panels["returns"] = panels["close"].pct_change(fill_method=None)
        # 基准与 beta120
        bench = (panels["returns"].mean(axis=1) if spec["bench"] == "EW"
                 else panels["returns"][spec["bench"]])
        cov = panels["returns"].rolling(120).cov(bench)
        var = bench.rolling(120).var()
        panels["beta120"] = cov.div(var, axis=0)
        panels["entropy120"] = pd.DataFrame(
            {i: _entropy120(panels["returns"][i]) for i in insts})
        classes[cname] = {"panels": panels, "bench_ret": bench, "spec": spec}

    # 执行层: ETF 后复权收益 + 上市掩码
    all_etfs = [e for s in classes.values() for e in s["spec"]["etf_map"]]
    etf_close = pd.DataFrame({e: align("etf_" + e, "close", False) for e in all_etfs})
    etf_ret = etf_close.pct_change(fill_method=None)
    listed = etf_close.notna()

    rf = align("etf_511880", "close", False).pct_change(fill_method=None) \
        .reindex(cal).fillna(0.0).clip(lower=0)             # 货基日收益,缺段计0

    out = {"cal": cal, "classes": classes, "etf_ret": etf_ret, "listed": listed, "rf": rf,
           "built_end": C.END}          # 缓存指纹:见 load_all,END 变了必须重建
    if save:
        with open(os.path.join(C.CACHE, "panels_etf.pkl"), "wb") as f:
            pickle.dump(out, f)
    return out


def CLASSES_ITEMS():
    return C.CLASSES.items()


def load_all() -> dict:
    """读面板缓存;缓存不是用当前 C.END 建的就重建。

    别去掉这个校验:面板日历会按 C.END 截断,而缓存文件没有任何标识。
    以前用 --end 2015-06-30 跑一次短测,就会把截断到 2015 年的面板落盘,
    之后所有全量挖矿都静默复用它 —— 只挖到 2015Q2 就没数据了,还不报错。
    """
    fp = os.path.join(C.CACHE, "panels_etf.pkl")
    if os.path.exists(fp):
        with open(fp, "rb") as f:
            d = pickle.load(f)
        if d.get("built_end") == C.END:
            return d
        print(f"[data] 面板缓存按 END={d.get('built_end')} 建,当前 END={C.END},重建")
    return build_all()


if __name__ == "__main__":
    d = build_all()
    print("日历:", d["cal"][0].date(), "~", d["cal"][-1].date(), len(d["cal"]), "天")
    for cn, cd in d["classes"].items():
        cl = cd["panels"]["close"]
        first = cl.apply(lambda s: s.first_valid_index())
        print(f"[{cn}] 品种起点: " + ", ".join(f"{i}={str(first[i].date())[:7]}" for i in cl))
        na_beta = cd["panels"]["beta120"].notna().sum().min()
        print(f"        beta120最少有效数={na_beta}, entropy有效={cd['panels']['entropy120'].notna().sum().min()}")
    print("ETF上市数(2020-01):", int(d["listed"].loc["2020-01-03"].sum()) if "2020-01-03" in d["listed"].index.astype(str) else d["listed"].loc["2020"].iloc[0].sum())
    print("rf 2024均值年化:", float(d["rf"].loc["2024"].mean() * 244))
