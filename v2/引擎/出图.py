# -*- coding: utf-8 -*-
"""中证1000 v1 出图:净值图 / 仓位轨迹 / 回撤对比 / 逐年柱状。
用法: python -m 引擎.出图   (cwd=中证1000/v1)
"""
from __future__ import annotations

import os
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
plt.rcParams["font.family"] = ["Arial Unicode MS", "PingFang SC", "Heiti SC"]
plt.rcParams["axes.unicode_minus"] = False

from 引擎.策略 import 结果, 口径起, 跑一遍, 统计

图目录 = os.path.join(结果, "图")
os.makedirs(图目录, exist_ok=True)


def main():
    收, p, r, rf = 跑一遍(True)
    收多, p多, _, _ = 跑一遍(True, 禁空=True)
    收基, _, _, _ = 跑一遍(False)
    seg = slice(口径起, None)
    nav = (1 + 收.loc[seg]).cumprod()
    nav多 = (1 + 收多.loc[seg]).cumprod()
    nav基 = (1 + 收基.loc[seg]).cumprod()
    nav持 = (1 + r.loc[seg].fillna(0)).cumprod()
    nav货 = (1 + rf.loc[seg]).cumprod()

    # ---- 图1:净值 + 仓位 ----
    fig, axes = plt.subplots(2, 1, figsize=(12, 8.5), height_ratios=[3, 1], sharex=True)
    axes[0].plot(nav, lw=2.0, label="择时·熊增强(定稿)", color="#c0392b")
    axes[0].plot(nav多, lw=1.6, label="择时·纯多(禁空)", color="#2980b9")
    axes[0].plot(nav基, lw=1.1, alpha=0.6, label="择时·仅量价(多空)", color="#e67e22")
    axes[0].plot(nav持, lw=1.3, alpha=0.8, label="持有512100", color="#7f8c8d")
    axes[0].plot(nav货, lw=1.0, ls="--", alpha=0.6, label="货基", color="#2c3e50")
    axes[0].set_yscale("log")
    axes[0].set_title("中证1000择时 v1 · 净值(对数轴,2018起,扣双边万5)")
    axes[0].legend(); axes[0].grid(alpha=0.3, which="both")
    axes[1].fill_between(p.loc[seg].index, p.loc[seg], 0, where=p.loc[seg] >= 0,
                         color="#c0392b", alpha=0.55, step="mid", label="多头")
    axes[1].fill_between(p.loc[seg].index, p.loc[seg], 0, where=p.loc[seg] < 0,
                         color="#2980b9", alpha=0.55, step="mid", label="空头")
    axes[1].set_ylim(-1.1, 1.1); axes[1].axhline(0, color="k", lw=0.5)
    axes[1].set_title("仓位轨迹"); axes[1].legend(ncol=2, fontsize=9); axes[1].grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(os.path.join(图目录, "净值图.png"), dpi=140); plt.close(fig)

    # ---- 图2:回撤对比 ----
    fig, ax = plt.subplots(figsize=(12, 5))
    for s, 名, c in [(nav, "择时·多空", "#c0392b"), (nav多, "择时·纯多", "#2980b9"),
                     (nav持, "持有512100", "#7f8c8d")]:
        ax.fill_between(s.index, (s / s.cummax() - 1) * 100, 0, alpha=0.45, color=c, label=名)
    ax.set_title("回撤对比(%)"); ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(os.path.join(图目录, "回撤图.png"), dpi=140); plt.close(fig)

    # ---- 图3:逐年柱状 ----
    年, 择, 多, 持 = [], [], [], []
    for y in range(2018, 2027):
        a = 收.loc[str(y)]
        if len(a) <= 20:
            continue
        年.append(str(y))
        择.append((1 + a).prod() ** (244 / len(a)) - 1)
        多.append((1 + 收多.loc[str(y)]).prod() ** (244 / len(收多.loc[str(y)])) - 1)
        持.append((1 + r.loc[str(y)].fillna(0)).prod() ** (244 / len(r.loc[str(y)])) - 1)
    x = np.arange(len(年)); w = 0.27
    fig, ax = plt.subplots(figsize=(13, 5))
    ax.bar(x - w, np.array(择) * 100, w, label="择时·多空", color="#c0392b")
    ax.bar(x, np.array(多) * 100, w, label="择时·纯多", color="#2980b9")
    ax.bar(x + w, np.array(持) * 100, w, label="持有512100", color="#7f8c8d")
    for i, (a, b, c_) in enumerate(zip(择, 多, 持)):
        for dx, v in ((-w, a), (0, b), (w, c_)):
            ax.text(i + dx, v * 100, f"{v:.0%}", ha="center",
                    va="bottom" if v >= 0 else "top", fontsize=7)
    ax.set_xticks(x); ax.set_xticklabels(年); ax.axhline(0, color="k", lw=0.8)
    ax.set_title("逐年收益对比(%)"); ax.legend(); ax.grid(alpha=0.3, axis="y")
    fig.tight_layout(); fig.savefig(os.path.join(图目录, "逐年图.png"), dpi=140); plt.close(fig)

    for 名, x, q in (("多空", 收, p), ("纯多", 收多, p多)):
        st = 统计(x, rf, q)
        print(f"{名} 年化{st['年化']:.1%} 夏普{st['夏普']:.2f} 回撤{st['最大回撤']:.1%} "
              f"卡玛{st['卡玛']:.2f} 均|仓|{st['均|仓|']:.0%}")
    print("落盘:", 图目录)


if __name__ == "__main__":
    main()
