# -*- coding: utf-8 -*-
"""指数v2 实盘信号:算最新收盘日的三巨头推荐仓位(下一交易日执行)。

口径与回测引擎逐字一致:名册as-of重建→季度权重→家族再倾斜(A2_λ.5,熊强度h)
→组合信号→rank40分位→tanh(probit)。同时输出基线(纯量价)对照。
用法: python -m 引擎.实盘信号 [--end 2026-07-17]   (cwd=etf择时/指数v2)
"""
from __future__ import annotations

import argparse
import os
import pickle
import warnings

import numpy as np
import pandas as pd
from scipy.stats import norm

warnings.filterwarnings("ignore")

# ---- 外部依赖守卫:本脚本需要因子挖掘框架,不随本仓库分发 ----
try:
    import ga_alpha, walkforward          # noqa: F401
except ModuleNotFoundError as _e:
    raise SystemExit(
        f"缺少外部模块 {_e.name}。\n"
        "本脚本(实盘信号 / 家族再倾斜)需要遗传规划因子挖掘框架 ga_alpha + walkforward,\n"
        "它们不随本仓库分发。回测复现不需要它们——策略.py / 出图.py / 映射对比.py /\n"
        "逐年信号图.py 用 数据/ 下的信号存档即可独立运行。\n"
        "若要重算实盘信号,请把 ga_alpha 与 walkforward 所在目录加入 PYTHONPATH。")

import ga_alpha.expr as gexpr
from walkforward import config as C
from walkforward.elders import ARGARCH_NAMES, ArgarchRecorder, trend_elder_values




基 = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
指数结果 = os.path.join(os.path.dirname(基), "指数", "结果")
三巨头 = {"hs300": ("510300", "沪深300"), "zz500": ("510500", "中证500"),
         "zz1000": ("512100", "中证1000")}
λ, 压族 = 0.5, ("反转", "波动")           # A2_λ.5
标签相关阈, 起算季 = 0.30, "2025-01-01"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--end", default="2026-07-17")
    a = ap.parse_args()
    C.END = a.end
    from walkforward.data import build_all
    def 熊概率(end):
        """读 v2/结果/bear_prob.csv(须先在 v2 目录跑 更新熊概率.py 延伸到最新日)。"""
        fp = os.path.join(os.path.dirname(基), "v2", "结果", "bear_prob.csv")
        s = pd.read_csv(fp, parse_dates=["date"]).set_index("date")["p_bear"].dropna()
        assert str(s.index.max().date()) >= end, \
            f"熊概率只到{s.index.max().date()},请先在 v2 目录运行 引擎/更新熊概率.py --end {end}"
        return s

    数据 = build_all(save=False)
    面板 = 数据["classes"]["A股宽基"]["panels"]
    日历 = 面板["close"].index
    最新 = 日历[-1]
    print(f"最新收盘日: {最新.date()}  → 该信号用于下一交易日建仓\n")

    # ---- 熊概率(延伸到最新) ----
    p = 熊概率(a.end)
    p_now = float(p.reindex(日历).ffill().loc[最新])
    h_now = np.clip((abs(p_now - 0.5) - 0.10) / 0.40, 0, None) * np.sign(p_now - 0.5)
    h_now = float(np.clip(h_now, -1, 1))
    print(f"熊概率 p_bear = {p_now:.2f}  → 熊强度 h = {h_now:+.2f} "
          f"({'压抄底派/抬动量' if h_now>0 else '放开抄底派' if h_now<0 else '中性带,不调制'})\n")
    h = (((p.reindex(日历) - 0.5).abs() - 0.10).clip(lower=0) / 0.40
         * np.sign(p.reindex(日历) - 0.5)).clip(-1, 1).fillna(0)

    with open(os.path.join(指数结果, "state.pkl"), "rb") as f:
        st = pickle.load(f)
    记 = ArgarchRecorder(面板["returns"], os.path.join(指数结果, "argarch"))
    记.load()
    新日 = 日历[日历 > 记.values(ARGARCH_NAMES[0]).index.max()]
    if len(新日):
        记.extend(pd.Timestamp("2026-06-30"), 新日)      # 季内冻结参数,滤波延伸
    趋势 = trend_elder_values(面板)
    ret, close, vol = 面板["returns"], 面板["close"], 面板["volume"]
    原型 = {"动量": close.pct_change(20), "反转": -close.pct_change(5),
           "波动": ret.rolling(20).std(), "量能": vol.rolling(5).mean() / vol.rolling(60).mean()}

    成员 = {}
    for 编号, m in st["members"].items():
        if m.get("专属") not in (None, "hs300", "zz500", "zz1000"):
            continue
        if m["kind"] == "argarch":
            v = 记.values(m["name"]).reindex(日历)
        elif m["kind"] == "elder":
            v = 趋势[m["name"]]
        else:
            子 = 面板 if m["专属"] is None else {k: pp[[m["专属"]]] for k, pp in 面板.items()}
            v = gexpr.evaluate(m["node"], 子).replace([np.inf, -np.inf], np.nan)
        成员[编号] = {"m": m, "sig": (v.rolling(120).rank(pct=True) * 2 - 1) * m["sign"]}

    尾均 = lambda hist, n=12: float(np.mean([hist[k] for k in sorted(hist)[-n:]])) if hist else np.nan
    标签缓存 = {}

    def 家族(编号, 指数, cutoff):
        键 = (编号, 指数, cutoff.year)
        if 键 in 标签缓存:
            return 标签缓存[键]
        sig = 成员[编号]["sig"]
        col = sig[指数] if 指数 in sig.columns else sig.iloc[:, 0]
        窗 = col.loc[:cutoff].tail(750)
        最好, 族 = 标签相关阈, "其他"
        for 名, ap_ in 原型.items():
            j = pd.concat([窗, ap_[指数].reindex(窗.index)], axis=1).dropna()
            if len(j) < 250:
                continue
            c = abs(j.iloc[:, 0].corr(j.iloc[:, 1], method="spearman"))
            if np.isfinite(c) and c > 最好:
                最好, 族 = c, 名
        标签缓存[键] = 族
        return 族

    季度表 = list(pd.period_range(起算季, a.end, freq="Q"))
    print(f"{'指数':8s} {'基线仓位':>8s} {'熊增强仓位':>10s} {'差异':>7s}   在役因子")
    结果行 = []
    for 指数, (etf, 名) in 三巨头.items():
        基线S = pd.Series(0.0, index=日历); 增强S = pd.Series(0.0, index=日历)
        n役 = 0
        for q in 季度表:
            前 = 日历[日历 < q.start_time]; 季日 = 日历[(日历 >= q.start_time) & (日历 <= q.end_time)]
            if len(前) == 0 or len(季日) == 0:
                continue
            cutoff = 前[-1]
            在役 = []
            for 编号, d in 成员.items():
                m = d["m"]
                if m.get("专属") not in (None, 指数) or pd.Period(m["admit_q"]) > q:
                    continue
                rq = m.get("retire_q")
                if isinstance(rq, str) and pd.Period(rq) <= q and m["status"] == "退役":
                    continue
                w = 尾均(m.get("分账", {}).get(指数, {}))
                w = max(w, 0.0) if np.isfinite(w) else 0.0
                if w > 0:
                    在役.append((编号, w))
            总w = sum(w for _, w in 在役)
            if 总w <= 0:
                continue
            if q == 季度表[-1]:
                n役 = len(在役)
            hq = h.reindex(季日).fillna(0)
            基分, 增分 = {}, {}
            for 编号, w in 在役:
                族 = 家族(编号, 指数, cutoff)
                调 = (np.exp(-λ * hq) if 族 in 压族 else
                      np.exp(+λ * hq) if 族 == "动量" else pd.Series(1.0, index=季日))
                基分[编号] = pd.Series(w / 总w, index=季日)
                增分[编号] = (w / 总w) * 调
            for 表, 目标 in ((基分, 基线S), (增分, 增强S)):
                归 = sum(表.values())
                for 编号, wt in 表.items():
                    sig = 成员[编号]["sig"]
                    col = (sig[指数] if 指数 in sig.columns else sig.iloc[:, 0]).reindex(季日).fillna(0)
                    目标.loc[季日] += col * (wt / 归)

        def 仓(S):
            # v1.2:与 引擎/策略.py 的 建仓位() 严格同口径(40日窗 + tanh)
            π = (S.rolling(40).rank(pct=True) - 0.5 / 40).clip(1e-6, 1 - 1e-6)
            return float(np.tanh(norm.ppf(π.loc[最新])))
        b, e = 仓(基线S), 仓(增强S)
        print(f"{名:8s} {b:+8.2f} {e:+10.2f} {e-b:+7.2f}   {n役}个")
        结果行.append({"指数": 名, "ETF": etf, "基线仓位": round(b, 3),
                     "熊增强仓位": round(e, 3), "差异": round(e - b, 3),
                     "在役因子": n役, "p_bear": round(p_now, 3), "h": round(h_now, 3),
                     "信号日": str(最新.date())})
    out = os.path.join(基, "结果", "实盘信号.csv")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    pd.DataFrame(结果行).to_csv(out, index=False)
    print(f"\n落盘: {out}")


if __name__ == "__main__":
    main()
