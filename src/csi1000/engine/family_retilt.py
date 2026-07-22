# -*- coding: utf-8 -*-
"""路线A·家族再倾斜(论文exp(±λh)机制移植到组合信号内部)。

机制:每日 w_i(t) = w_i^季 × exp(s_f·λ·h_t),s_f: 反转=−1(A2含波动族),动量=+1,其余0;
日内重归一;组合→rank120→probit→结算。h来自v2熊概率(论文中性带)。
名册/权重按指数引擎口径as-of重建(admit/retire近似,复职连续;家族标签逐年as-of,
用trailing750日与原型的|spearman相关|,<0.3=其他)。
变体网格: {A1:只调反转/动量, A2:反转+波动同压} × λ∈{0.25,0.5} + 基线(λ=0)。
用法: python -m 引擎.家族再倾斜   (cwd=etf择时/v2)
"""
from __future__ import annotations

import os
import pickle
import time
import warnings

import numpy as np
import pandas as pd
from scipy.stats import norm

warnings.filterwarnings("ignore")

# ---- 外部依赖守卫:本脚本需要因子挖掘框架,不随本仓库分发 ----
try:
    import csi1000.ga_alpha, walkforward          # noqa: F401
except ModuleNotFoundError as _e:
    raise SystemExit(
        f"缺少外部模块 {_e.name}。\n"
        "本脚本(实盘信号 / 家族再倾斜)需要遗传规划因子挖掘框架 ga_alpha + walkforward,\n"
        "它们不随本仓库分发。回测复现不需要它们——策略.py / 出图.py / 映射对比.py /\n"
        "逐年信号图.py 用 数据/ 下的信号存档即可独立运行。\n"
        "若要重算实盘信号,请把 ga_alpha 与 walkforward 所在目录加入 PYTHONPATH。")

import csi1000.ga_alpha.expr as gexpr
from csi1000.walkforward.data import load_all
from csi1000.walkforward.elders import ARGARCH_NAMES, ArgarchRecorder, trend_elder_values




基 = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
指数结果 = os.path.join(os.path.dirname(基), "指数", "结果")
缓存 = os.path.join(os.path.dirname(基), "data_cache")
结果目录 = os.path.join(基, "结果")
三巨头 = {"hs300": "510300", "zz500": "510500", "zz1000": "512100"}
口径起, 成本 = "2018-01-01", 0.0005
标签相关阈 = 0.30


def 主流程():
    数据 = load_all()
    面板 = 数据["classes"]["A股宽基"]["panels"]
    日历 = 面板["close"].index
    rf = 数据["rf"].reindex(日历).fillna(0)

    # 熊强度h(论文式18)
    p = pd.read_csv(os.path.join(基, "结果", "bear_prob.csv"), parse_dates=["date"]) \
        .set_index("date")["p_bear"].reindex(日历)
    h = (((p - 0.5).abs() - 0.10).clip(lower=0) / 0.40 * np.sign(p - 0.5)).clip(-1, 1).fillna(0)

    with open(os.path.join(指数结果, "state.pkl"), "rb") as f:
        st = pickle.load(f)
    记 = ArgarchRecorder(面板["returns"], os.path.join(指数结果, "argarch"))
    assert 记.load()
    趋势 = trend_elder_values(面板)

    # 原型信号(逐指数列):动量/反转/波动/量能
    ret, close, vol = 面板["returns"], 面板["close"], 面板["volume"]
    原型 = {"动量": close.pct_change(20), "反转": -close.pct_change(5),
           "波动": ret.rolling(20).std(), "量能": vol.rolling(5).mean() / vol.rolling(60).mean()}

    # 成员信号重建(每指数一列),只留服务三巨头的
    print("重建成员信号...")
    t0 = time.time()
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
        s = (v.rolling(120).rank(pct=True) * 2 - 1) * m["sign"]
        成员[编号] = {"m": m, "sig": s}
    print(f"  {len(成员)}个成员, {time.time()-t0:.0f}s")

    尾均 = lambda hist, n=12: float(np.mean([hist[k] for k in sorted(hist)[-n:]])) if hist else np.nan
    季度表 = list(pd.period_range("2015-01-01", "2026-07-16", freq="Q"))

    # 逐年as-of家族标签: {(编号,指数,年份): 族}
    标签缓存 = {}

    def 家族(编号, 指数, cutoff):
        年 = cutoff.year
        键 = (编号, 指数, 年)
        if 键 in 标签缓存:
            return 标签缓存[键]
        sig = 成员[编号]["sig"]
        col = sig[指数] if 指数 in sig.columns else sig.iloc[:, 0]
        窗 = col.loc[:cutoff].tail(750)
        最好, 最好族 = 标签相关阈, "其他"
        for 族名, ap in 原型.items():
            a = ap[指数].reindex(窗.index)
            j = pd.concat([窗, a], axis=1).dropna()
            if len(j) < 250:
                continue
            c = abs(j.iloc[:, 0].corr(j.iloc[:, 1], method="spearman"))
            if np.isfinite(c) and c > 最好:
                最好, 最好族 = c, 族名
        标签缓存[键] = 最好族
        return 最好族

    变体表 = {"基线": (0.0, ()), "A1_λ.25": (0.25, ("反转",)), "A1_λ.5": (0.5, ("反转",)),
             "A2_λ.25": (0.25, ("反转", "波动")), "A2_λ.5": (0.5, ("反转", "波动"))}
    仓集 = {名: {} for 名 in 变体表}
    ΔS集 = {}                          # A2_λ.5 相对基线的组合信号增量(供定稿报告锚定正典)

    for 指数, etf in 三巨头.items():
        t0 = time.time()
        组合集 = {名: pd.Series(0.0, index=日历) for 名 in 变体表}
        for q in 季度表:
            前 = 日历[日历 < q.start_time]
            季日 = 日历[(日历 >= q.start_time) & (日历 <= q.end_time)]
            if len(前) == 0 or len(季日) == 0:
                continue
            cutoff = 前[-1]
            在役 = []
            for 编号, d in 成员.items():
                m = d["m"]
                if m.get("专属") not in (None, 指数):
                    continue
                if pd.Period(m["admit_q"]) > q:
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
            hq = h.reindex(季日).fillna(0)
            for 名, (λ, 压族) in 变体表.items():
                # 日频调制权重并归一
                分子 = {}
                for 编号, w in 在役:
                    族 = 家族(编号, 指数, cutoff) if λ > 0 else "其他"
                    if 族 in 压族:
                        调 = np.exp(-λ * hq)
                    elif 族 == "动量":
                        调 = np.exp(+λ * hq) if λ > 0 else 1.0
                    else:
                        调 = 1.0
                    分子[编号] = (w / 总w) * (调 if isinstance(调, pd.Series)
                                            else pd.Series(1.0, index=季日))
                归一 = sum(分子.values())
                for 编号, wt in 分子.items():
                    sig = 成员[编号]["sig"]
                    col = (sig[指数] if 指数 in sig.columns else sig.iloc[:, 0]) \
                        .reindex(季日).fillna(0)
                    组合集[名].loc[季日] += col * (wt / 归一)
        # A2_λ.5 相对基线的信号增量(锚定正典用)
        ΔS集[etf] = 组合集["A2_λ.5"] - 组合集["基线"]
        # 映射与结算
        r_etf = pd.read_csv(os.path.join(缓存, f"etf_{etf}.csv"), parse_dates=["date"]) \
            .set_index("date")["close"].pct_change().reindex(日历)
        for 名 in 变体表:
            S = 组合集[名]
            # 注:此处刻意保留 v1.0 口径(120日+clip),用于复现 数据/熊增强逐日.csv
            # 的生成过程;ΔS 取自映射之前,不受影响。实盘口径见 引擎/策略.py。
            π = (S.rolling(120).rank(pct=True) - 0.5 / 120).clip(1e-6, 1 - 1e-6)
            仓 = pd.Series(norm.ppf(π), index=日历).clip(-1, 1) \
                .where(S.abs().gt(1e-12), 0.0).where(r_etf.notna(), 0.0).fillna(0)
            仓集[名][etf] = (rf + 仓.shift(1).fillna(0) * (r_etf.fillna(0) - rf)
                           - 仓.diff().abs().fillna(0) * 成本)
        print(f"  [{指数}] 完成 {time.time()-t0:.0f}s")

    def 统计(收):
        收 = 收.loc[口径起:]
        rfd = 0.02 / 244
        nav = (1 + 收).prod(); 年 = len(收) / 244; ex = 收 - rfd
        dd = ((1 + 收).cumprod() / (1 + 收).cumprod().cummax() - 1).min()
        return (f"年化{nav**(1/年)-1:+6.1%} 夏普{ex.mean()/ex.std()*np.sqrt(244):+.2f} "
                f"回撤{dd:6.1%} 卡玛{(nav**(1/年)-1)/abs(dd):.2f}")

    行 = []
    print("\n== 三巨头等权组合(2018~) ==")
    for 名 in 变体表:
        组 = pd.concat(仓集[名].values(), axis=1).mean(axis=1)
        out = 统计(组)
        print(f"  {名:8s} {out}")
        行.append({"变体": 名, "层级": "组合", "结果": out})
    print("\n== 分资产(2018~) ==")
    for etf in ["510300", "510500", "512100"]:
        for 名 in 变体表:
            out = 统计(仓集[名][etf])
            if 名 in ("基线", "A2_λ.5"):
                print(f"  {etf}·{名:8s} {out}")
            行.append({"变体": 名, "层级": etf, "结果": out})
    pd.DataFrame(行).to_csv(os.path.join(结果目录, "家族再倾斜实验.csv"), index=False)

    # 持久化逐日序列(供定稿报告出图出表):基线 + A2_λ.5,分资产 + 组合
    逐日 = {}
    for 名 in ("基线", "A2_λ.5"):
        for etf in ["510300", "510500", "512100"]:
            逐日[f"{名}_{etf}"] = 仓集[名][etf]
        逐日[f"{名}_组合"] = pd.concat(仓集[名].values(), axis=1).mean(axis=1)
    for etf in ["510300", "510500", "512100"]:
        逐日[f"持有_{etf}"] = pd.read_csv(os.path.join(缓存, f"etf_{etf}.csv"),
                                        parse_dates=["date"]).set_index("date")["close"].pct_change().reindex(日历)
        逐日[f"ΔS_{etf}"] = ΔS集[etf]        # A2信号增量,供报告锚定正典
    逐日["p_bear"] = p
    pd.DataFrame(逐日).to_csv(os.path.join(结果目录, "熊增强逐日.csv"))
    print("落盘:", os.path.join(结果目录, "熊增强逐日.csv"))


if __name__ == "__main__":
    主流程()
