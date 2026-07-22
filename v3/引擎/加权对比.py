# -*- coding: utf-8 -*-
"""第一版(rank IC 挖矿)——五种加权方式的净值/回撤/逐年对比图与汇总表。

因子库是同一批(26 算子、rank IC 挖出的 1245 个,复现仓库发布版 20.7%),
只改"怎么把因子加权合成"这一件事,做受控对比:

  正典      筛除(IC≤0剔除) + IC加权     ← 发布版线上口径,复现 20.7%/夏普1.01
  截正等权  筛除            + 等权
  等权      不筛除          + 等权
  纯等权    不筛除          + 等权 + 黄牌不减半
  cosIC     cos IC 加权(方向×强度,对齐真实盈亏)← 最优

各档信号从挖矿存档 结果_26算子/因子逐日信号.csv.gz 重算得到(不重挖),
再套同一套仓位映射(40日 tanh∘Φ⁻¹)与结算(双边万5)出净值。

用法: python -m 引擎.加权对比
产出: 结果/加权对比/{净值对比,回撤对比,逐年对比}.png + 汇总表.csv
"""
from __future__ import annotations

import os
import subprocess
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import 引擎.策略 as st

plt.rcParams["font.family"] = ["Arial Unicode MS", "PingFang SC", "Heiti SC"]
plt.rcParams["axes.unicode_minus"] = False

根 = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
挖矿存档 = os.path.join(根, "结果_26算子")
出目录 = os.path.join(根, "结果", "加权对比")
os.makedirs(出目录, exist_ok=True)

# (显示名, 输出子目录, 生成命令 or None=正典已在存档里, 颜色)
档位 = [
    ("正典 (rank IC 加权)", "结果_正典",   ["-m", "引擎.重算权重", "--方案", "正典"],   "#7f8c8d"),
    ("截正等权",            "结果_截正等权", ["-m", "引擎.重算权重", "--方案", "截正等权"], "#27ae60"),
    ("等权",               "结果_等权",   ["-m", "引擎.重算权重", "--方案", "等权"],     "#2980b9"),
    ("纯等权",             "结果_纯等权",  ["-m", "引擎.重算权重", "--方案", "纯等权"],   "#8e44ad"),
    ("cos IC 加权",        "结果_cosIC",  ["-m", "引擎.cos_ic权重"],                  "#c0392b"),
]


def 生成信号():
    """从挖矿存档重算各档 分资产逐日.csv(已存在则跳过)。"""
    for 名, 子, 命令, _ in 档位:
        目标 = os.path.join(根, 子, "分资产逐日.csv")
        if os.path.exists(目标):
            continue
        r = subprocess.run([sys.executable, *命令, "--存档", "结果_26算子", "--输出", 子],
                           cwd=根, env=dict(os.environ, PYTHONPATH=根),
                           capture_output=True, text=True)
        assert r.returncode == 0, f"{名} 信号生成失败:\n{r.stderr[-600:]}"
        print(f"  已生成 {子}")


def 档位净值(子目录: str, 禁空=False) -> tuple[pd.Series, pd.Series, pd.Series]:
    """读某档 分资产逐日 → 叠熊增强 → 仓位映射 → 结算,返回(逐日收益, 仓位, 标的收益)。"""
    S = pd.read_csv(os.path.join(根, 子目录, "分资产逐日.csv"), index_col=0,
                    parse_dates=True)[f"{st.ETF}_仓位"]
    fp = os.path.join(st.数据, "熊增强逐日.csv")
    if os.path.exists(fp):
        ΔS = pd.read_csv(fp, parse_dates=["date"]).set_index("date")[f"ΔS_{st.ETF}"]
        S = S + ΔS.reindex(S.index).fillna(0)
    r = st.读价(f"etf_{st.ETF}").pct_change().reindex(S.index)
    rf = st.读价("etf_511880").pct_change().clip(lower=0).reindex(S.index).fillna(0)
    p = st.建仓位(S, r, 禁空=禁空)
    return st.结算(p, r, rf), p, r


def main():
    print("生成五档信号(从挖矿存档重算,不重挖):")
    生成信号()

    收益, 仓位, 标的收益 = {}, {}, None
    for 名, 子, _, _ in 档位:
        收益[名], 仓位[名], 标的收益 = 档位净值(子)

    起 = st.口径起
    def 净(名): return (1 + 收益[名].loc[起:]).cumprod()

    # ---- 图1 净值对比 ----
    fig, ax = plt.subplots(figsize=(12, 6))
    for 名, _, _, c in 档位:
        nav = 净(名)
        lw = 2.4 if "cos" in 名 or "正典" in 名 else 1.4
        ax.plot(nav, lw=lw, label=f"{名}  ({nav.iloc[-1]:.2f}×)", color=c)
    持 = (1 + 标的收益.loc[起:].fillna(0)).cumprod()
    ax.plot(持, lw=1.2, ls="--", alpha=0.7, color="#95a5a6", label=f"持有512100  ({持.iloc[-1]:.2f}×)")
    ax.set_title("五种加权方式净值对比(熊增强·多空,2018起)", fontsize=13)
    ax.set_ylabel("净值(初始=1)"); ax.legend(fontsize=10); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(os.path.join(出目录, "净值对比.png"), dpi=150); plt.close(fig)

    # ---- 图2 回撤对比 ----
    fig, ax = plt.subplots(figsize=(12, 5))
    for 名, _, _, c in 档位:
        nav = 净(名); dd = (nav / nav.cummax() - 1) * 100
        lw = 2.2 if "cos" in 名 or "正典" in 名 else 1.2
        ax.plot(dd, lw=lw, label=名, color=c)
    ax.set_title("回撤对比(%)", fontsize=13); ax.axhline(0, color="k", lw=0.5)
    ax.legend(fontsize=10); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(os.path.join(出目录, "回撤对比.png"), dpi=150); plt.close(fig)

    # ---- 图3 逐年收益柱状 ----
    年份 = list(range(2018, 2027))
    fig, ax = plt.subplots(figsize=(13, 5.5))
    n = len(档位); w = 0.8 / n
    for i, (名, _, _, c) in enumerate(档位):
        年收 = [收益[名].loc[str(y)].pipe(lambda s: (1 + s).prod() - 1) * 100
               if len(收益[名].loc[str(y):str(y)]) > 20 else 0 for y in 年份]
        ax.bar(np.arange(len(年份)) + (i - n / 2 + 0.5) * w, 年收, w, label=名, color=c)
    ax.set_xticks(range(len(年份))); ax.set_xticklabels(年份)
    ax.set_title("逐年收益对比(%)", fontsize=13); ax.axhline(0, color="k", lw=0.5)
    ax.legend(fontsize=9, ncol=5); ax.grid(alpha=0.3, axis="y")
    fig.tight_layout(); fig.savefig(os.path.join(出目录, "逐年对比.png"), dpi=150); plt.close(fig)

    # ---- 汇总表(多空 + 纯多 两口径) ----
    行 = []
    for 名, 子, _, _ in 档位:
        for 口径, 禁空 in (("多空", False), ("纯多", True)):
            收, 仓, r = 档位净值(子, 禁空=禁空)
            s = st.统计(收, st.读价("etf_511880").pct_change().clip(lower=0).fillna(0), 仓)
            行.append({"加权方式": 名, "口径": 口径,
                       "年化%": f"{s['年化']*100:.1f}", "夏普": f"{s['夏普']:.2f}",
                       "最大回撤%": f"{s['最大回撤']*100:.1f}", "卡玛": f"{s['卡玛']:.2f}",
                       "均|仓|%": f"{s['均|仓|']*100:.1f}"})
    表 = pd.DataFrame(行)
    表.to_csv(os.path.join(出目录, "汇总表.csv"), index=False)
    print("\n" + "=" * 70)
    for 口径 in ("多空", "纯多"):
        print(f"【熊增强·{口径}】")
        print(表[表.口径 == 口径].drop(columns="口径").to_string(index=False)); print()
    print("图与表已落:", 出目录)


if __name__ == "__main__":
    main()
