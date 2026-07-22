# -*- coding: utf-8 -*-
"""中证1000择时 v1 · 自包含策略模块(信号→仓位→结算的唯一真源)。

策略定义(定稿,跑完不改):
  基础信号 = 量价因子库(genetic-cta制度)全库IC加权组合信号  [数据/分资产逐日.csv 的 512100_仓位]
  熊增强   = + 家族再倾斜增量ΔS(熊态压反转/波动族、抬动量族)  [数据/熊增强逐日.csv 的 ΔS_512100]
  仓位     = tanh(Φ⁻¹(信号的40日滚动分位))                可做空,均|仓|约61%
             (v1.1:由clip改tanh——极端区保留分辨率,84分位不再一刀切满仓)
             (v1.2:映射窗120→40——空头腿夏普在40~50日见顶,中证500独立复现)
  结算     = 货基 + 仓位_{t-1}×(512100收益 − 货基) − 万5×|Δ仓位|
成绩(2018~2026):年化20.7% 夏普1.01 回撤−20.9% 卡玛0.99 vs 持有 4.1%/0.19/−46.3%
用法: from 引擎.策略 import 建仓位, 结算, 统计   或直接 python -m 引擎.策略
"""
from __future__ import annotations

import os
import warnings

import numpy as np
import pandas as pd
from scipy.stats import norm

warnings.filterwarnings("ignore")

本目录 = os.path.dirname(os.path.abspath(__file__))
根 = os.path.dirname(本目录)                                  # 中证1000/v1
数据 = os.path.join(根, "数据")
结果 = os.path.join(根, "结果")
缓存 = os.path.join(数据, "行情")                              # 自带行情(仓库自包含)
if not os.path.isdir(缓存):                                   # 回退到项目公共缓存
    缓存 = os.path.join(os.path.dirname(os.path.dirname(根)), "data_cache")

ETF, 指数列 = "512100", "zz1000"
映射窗, 成本, 口径起 = 40, 0.0005, "2018-01-01"   # v1.2:映射窗由120改40


def 读价(tag: str) -> pd.Series:
    return pd.read_csv(os.path.join(缓存, tag + ".csv"), parse_dates=["date"]) \
        .set_index("date")["close"]


def 建信号(含熊增强: bool = True) -> pd.Series:
    """基础量价组合信号(+熊增强ΔS)。两者都是全史因果序列。"""
    S = pd.read_csv(os.path.join(数据, "分资产逐日.csv"), index_col=0,
                    parse_dates=True)[f"{ETF}_仓位"]
    if 含熊增强:
        fp = os.path.join(数据, "熊增强逐日.csv")
        if os.path.exists(fp):
            ΔS = pd.read_csv(fp, parse_dates=["date"]).set_index("date")[f"ΔS_{ETF}"]
            S = S + ΔS.reindex(S.index).fillna(0)
    return S


def 建仓位(S: pd.Series, 标的收益: pd.Series | None = None,
         禁空: bool = False) -> pd.Series:
    """映射:信号→40日滚动分位→tanh(Φ⁻¹(π))。信号为0处强制空仓。
    v1.1:tanh 取代 clip——clip在84分位即饱和成平顶,tanh在极端区保留分辨率。
    v1.2:映射窗120→40。腿分解显示窗口效应几乎全在空头腿:空头腿夏普
    随窗口呈单峰,40~50日见顶(1000:0.76→0.98,500:0.74→0.83),多头腿平坦、
    纯多口径全窗口无差别。沪深300无edge亦无虚假峰,为空白对照。"""
    π = (S.rolling(映射窗).rank(pct=True) - 0.5 / 映射窗).clip(1e-6, 1 - 1e-6)
    p = pd.Series(np.tanh(norm.ppf(π)), index=S.index).where(S.abs().gt(1e-12), 0.0)
    if 禁空:                     # 纯多版:clip_[0,1](tanh(Φ⁻¹(π))),看空即平仓吃货基
        p = p.clip(lower=0.0)
    if 标的收益 is not None:
        p = p.where(标的收益.reindex(S.index).notna(), 0.0)
    return p.fillna(0.0)


def 结算(p: pd.Series, r: pd.Series, rf: pd.Series) -> pd.Series:
    """日结算:仓位T-1生效,闲钱吃货基,双边万5。"""
    rfi = rf.reindex(p.index).fillna(0)
    return (rfi + p.shift(1).fillna(0) * (r.reindex(p.index).fillna(0) - rfi)
            - p.diff().abs().fillna(0) * 成本)


def 统计(r: pd.Series, rf: pd.Series, 仓: pd.Series | None = None, 起=口径起) -> dict:
    r = r.dropna().loc[起:]
    nav = (1 + r).cumprod(); 年 = len(r) / 244
    ex = r - rf.reindex(r.index).fillna(0)
    dd = (nav / nav.cummax() - 1).min()
    ann = nav.iloc[-1] ** (1 / 年) - 1
    出 = {"年化": ann, "夏普": ex.mean() / ex.std() * np.sqrt(244) if ex.std() > 0 else np.nan,
          "最大回撤": dd, "卡玛": ann / abs(dd) if dd < 0 else np.nan}
    if 仓 is not None:
        出["均|仓|"] = float(仓.reindex(r.index).abs().mean())
    return 出


def 读映射分位(含熊增强: bool) -> pd.Series | None:
    """v4:定稿信号 已按同质参照系(每季用本季权重回算窗口)算好 π,直接用。
    没有该文件时返回 None,建仓位 回退为旧的滚动分位口径。"""
    fp = os.path.join(数据, "映射分位.csv")
    if not os.path.exists(fp):
        return None
    d = pd.read_csv(fp, parse_dates=["date"]).set_index("date")
    return d["π_增强" if 含熊增强 else "π_量价"]


def 跑一遍(含熊增强: bool = True, 禁空: bool = False):
    """返回 (逐日收益, 仓位, 标的收益, 货基)。"""
    S = 建信号(含熊增强)
    r = 读价(f"etf_{ETF}").pct_change().reindex(S.index)
    rf = 读价("etf_511880").pct_change().clip(lower=0).reindex(S.index).fillna(0)
    π = 读映射分位(含熊增强)
    if π is not None:
        p = pd.Series(np.tanh(norm.ppf(π.reindex(S.index))), index=S.index) \
            .where(π.reindex(S.index).notna(), 0.0)
        if 禁空:
            p = p.clip(lower=0.0)
        p = p.where(r.notna(), 0.0).fillna(0.0)
    else:
        p = 建仓位(S, r, 禁空=禁空)
    return 结算(p, r, rf), p, r, rf


if __name__ == "__main__":
    收增强, p增强, r, rf = 跑一遍(True)
    收多, p多, _, _ = 跑一遍(True, 禁空=True)
    收基础, p基础, _, _ = 跑一遍(False)
    行 = [("熊增强·多空(定稿)", 统计(收增强, rf, p增强)),
         ("熊增强·纯多(禁空)", 统计(收多, rf, p多)),
         ("仅量价·多空", 统计(收基础, rf, p基础)),
         ("持有512100", 统计(r.fillna(0), rf)), ("", {})]
    for y in range(2018, 2027):
        if len(收增强.loc[str(y)]) > 20:
            行.append((f"{y}·多空", 统计(收增强.loc[str(y)], rf, 起=f"{y}-01-01")))
            行.append((f"{y}·纯多", 统计(收多.loc[str(y)], rf, 起=f"{y}-01-01")))
            行.append((f"{y}·持有", 统计(r.loc[str(y)].fillna(0), rf, 起=f"{y}-01-01")))
            行.append(("", {}))
    表 = pd.DataFrame([v for _, v in 行], index=[k for k, _ in 行]) \
        .reindex(columns=["年化", "夏普", "最大回撤", "卡玛", "均|仓|"])
    fmt = 表.copy()
    for c in fmt.columns:
        fmt[c] = fmt[c].apply(lambda v: (f"{v:.2f}" if c in ("夏普", "卡玛") else f"{v*100:.1f}%")
                              if isinstance(v, float) and np.isfinite(v) else "")
    os.makedirs(结果, exist_ok=True)
    fmt.to_csv(os.path.join(结果, "成绩表.csv"))
    print(fmt.to_string())
    pd.DataFrame({"多空收益": 收增强, "多空仓位": p增强, "纯多收益": 收多, "纯多仓位": p多,
                  "持有收益": r, "仅量价收益": 收基础}).to_csv(os.path.join(结果, "逐日.csv"))
    print("\n落盘:", 结果)
