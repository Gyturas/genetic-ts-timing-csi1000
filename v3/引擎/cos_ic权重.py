# -*- coding: utf-8 -*-
"""cos IC 加权 —— 用户提出的备选打分口径,替代 trailing rank IC 定权重。

问题背景:正典权重用的 rank IC(混合月度IC)是"信号排序 vs 收益排序"的相关性,
完全丢掉了信号的绝对大小——一个因子月度信号天天报 +0.9、行情天天涨,排序没有
变化则 rank IC≈0,但这实际是最赚钱的情形。cos IC 直接量分子是"仓位×收益"本身
(就是这个因子单独跑会赚多少钱),对齐真正的目标函数。

  单因子仓位  p_{i,t} = tanh(Φ⁻¹(ts_percentile(S_{i,t}, 40)))
             —— 和 引擎/策略.py 的 建仓位() 用的是同一个映射公式,同一个窗口,
                只是套在单个因子自己的信号上,而不是套在组合信号上。
  cos_IC(trailing12月, 因子i, 指数x) = Σ p·r / sqrt(Σp² · Σr²)
             —— r 是该指数下一日收益,窗口内的日频观测,不按月分桶。

必须因果:每季度只能用截止到该季度cutoff为止、过去12个月的 (p,r) 历史算 cos IC,
不能用未来信息 —— 这样才跟"正典"用的 trailing 12月 rank IC 权重公平对照,两者
唯一的区别就是"用什么口径衡量因子过去12个月的表现",其余(组合方式、最终仓位
映射)完全不变。

用法: python -m 引擎.cos_ic权重 --输出 结果_cosIC
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd
from scipy.stats import norm

TRAILING月 = 12
映射窗 = 40


def 单因子仓位(S: pd.Series) -> pd.Series:
    """与 引擎/策略.py:建仓位 完全同一公式,套用在单个因子自己的信号上。"""
    π = (S.rolling(映射窗).rank(pct=True) - 0.5 / 映射窗).clip(1e-6, 1 - 1e-6)
    return pd.Series(np.tanh(norm.ppf(π)), index=S.index).where(S.abs().gt(1e-12), 0.0)


def cos_ic(p: pd.Series, r: pd.Series) -> float:
    p, r = p.align(r, join="inner")
    m = p.notna() & r.notna()
    p, r = p[m].to_numpy(), r[m].to_numpy()
    denom = np.sqrt((p ** 2).sum() * (r ** 2).sum())
    return float((p * r).sum() / denom) if denom > 0 else np.nan


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--存档", default="结果_挖矿")
    ap.add_argument("--输出", required=True)
    a = ap.parse_args()
    根 = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    存档目录 = a.存档 if os.path.isabs(a.存档) else os.path.join(根, a.存档)
    输出 = a.输出 if os.path.isabs(a.输出) else os.path.join(根, a.输出)

    import 引擎.指数日频引擎 as E
    E.输出目录 = 存档目录
    E.安装GA()
    eng = E.宽基引擎(E.load_all())
    assert eng.状态["done_q"], "存档里没有跑完的季度,请先跑 引擎.指数日频引擎"

    存 = pd.read_csv(os.path.join(存档目录, "因子逐日信号.csv.gz"), parse_dates=["date"])
    print(f"存档 {len(存):,} 行,{存['因子'].nunique()} 个因子,"
          f"{存['date'].min():%Y-%m-%d}~{存['date'].max():%Y-%m-%d}")

    # 每个(指数,因子)拼出连续的 S 序列 → 算单因子仓位 p(需要历史做40日滚动分位,
    # 所以整段先算好,而不是逐季切片算——逐季切片会在窗口头部丢精度)
    p表, 黄牌表 = {}, {}
    for (指数, 因子), g in 存.groupby(["指数", "因子"]):
        s = g.set_index("date")["信号"].sort_index()
        s = s[~s.index.duplicated(keep="last")]
        p表[(指数, 因子)] = 单因子仓位(s)
        黄牌表[(指数, 因子)] = g.set_index("date")["黄牌"].sort_index()
    print(f"已算 {len(p表)} 个(指数,因子)单因子仓位序列")

    次日收益 = eng.次日收益   # 与挖矿口径一致的指数下一日收益

    季日全 = pd.Series(存["date"].unique()).sort_values()
    季表 = eng.季度表
    指数仓 = pd.DataFrame(0.0, index=存["date"].unique(), columns=eng.指数表).sort_index()
    烧机最少 = E.烧机最少

    for q in 季表:
        if str(q) not in eng.状态["done_q"]:
            continue
        季日 = eng.日历[(eng.日历 >= q.start_time) & (eng.日历 <= q.end_time)]
        季日 = 季日.intersection(指数仓.index)
        if len(季日) == 0:
            continue
        cutoff = q.start_time - pd.Timedelta(days=1)
        起 = cutoff - pd.DateOffset(months=TRAILING月)
        本季存 = 存[(存["date"] >= q.start_time) & (存["date"] <= q.end_time)]
        for 指数, gi in 本季存.groupby("指数"):
            权重 = {}
            for 因子 in gi["因子"].unique():
                p = p表.get((指数, 因子))
                if p is None:
                    continue
                r = 次日收益[指数] if 指数 in 次日收益.columns else None
                if r is None:
                    continue
                c = cos_ic(p.loc[起:cutoff], r.loc[起:cutoff])
                if not np.isfinite(c) or c <= 0:
                    continue
                黄 = 黄牌表[(指数, 因子)]
                最近黄 = 黄.loc[:cutoff]
                是黄 = bool(最近黄.iloc[-1]) if len(最近黄) else False
                权重[因子] = c * (0.5 if 是黄 else 1.0)
            总 = sum(权重.values())
            if 总 <= 0:
                continue
            合成 = pd.Series(0.0, index=季日)
            for 因子, w in 权重.items():
                s = gi[gi["因子"] == 因子].set_index("date")["信号"].reindex(季日).fillna(0.0)
                合成 = 合成.add(s * (w / 总), fill_value=0.0)
            if len(权重) < 烧机最少:
                合成 *= 0.5
            指数仓.loc[季日, 指数] = 合成.clip(-1, 1).values

    print(f"方案=cosIC  日均绝对仓位 {指数仓.abs().mean().mean():.3f}")

    E.输出目录 = 输出
    os.makedirs(输出, exist_ok=True)
    eng.状态.update(daily=[], 资产仓位=[], 分资产=[], 前暴露={}, 前仓资产={})
    eng.组合切片 = []
    eng._结算(list(指数仓.index), 指数仓)
    eng._导出()
    print(f"已写出 {输出}")


if __name__ == "__main__":
    main()
