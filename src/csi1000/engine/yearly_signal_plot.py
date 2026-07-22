# -*- coding: utf-8 -*-
"""中证1000 v1.2 · 逐年K线信号图(B/S标记 + 仓位轨迹)。

只标"显著"的仓位变动:自上一次标记以来累计变动 ≥ 标记阈值(默认0.5,即半仓)
才打点,避免把 1.0→0.9 这种噪声画满屏。
  B = 增加暴露(加多 或 平空)   S = 减少暴露(减多 或 加空)
一年一张图,落在 结果/逐年信号图/。
用法: python -m 引擎.逐年信号图   (cwd=中证1000/v1)
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

from csi1000.engine.strategy import ETF, 跑一遍, 结果, 缓存, 统计

标记阈值 = 0.5
图目录 = os.path.join(结果, "逐年信号图")

底色, 网格色, 字色 = "#131722", "#2a2e39", "#d1d4dc"
涨色, 跌色 = "#f23645", "#089981"          # A股惯例:红涨绿跌
B色, S色 = "#00e676", "#ff1744"


def 取标记(p: pd.Series, 阈值: float = 标记阈值) -> pd.DataFrame:
    """自上次标记以来累计变动≥阈值才记一次,兼顾突变与缓慢累积。"""
    行, 上次 = [], p.iloc[0]
    for d, v in p.items():
        if abs(v - 上次) >= 阈值:
            行.append({"date": d, "类": "B" if v > 上次 else "S",
                       "自": 上次, "至": v})
            上次 = v
    return pd.DataFrame(行).set_index("date") if 行 else pd.DataFrame()


def 画一年(年: int, px: pd.DataFrame, p: pd.Series, 收: pd.Series, r: pd.Series):
    d = px.loc[str(年)]
    if len(d) < 20:
        return None
    pos = p.reindex(d.index)
    mk = 取标记(pos)
    x = np.arange(len(d))
    o, h, l, c = (d[k].to_numpy(float) for k in ("open", "high", "low", "close"))
    涨 = c >= o

    fig, (ax, ax2) = plt.subplots(
        2, 1, figsize=(22, 11), height_ratios=[3.4, 1], sharex=True,
        gridspec_kw={"hspace": 0.06})
    fig.patch.set_facecolor(底色)
    for a in (ax, ax2):
        a.set_facecolor(底色)
        a.grid(color=网格色, lw=0.5, alpha=0.6)
        a.tick_params(colors=字色, labelsize=10)
        for s in a.spines.values():
            s.set_color(网格色)

    # ---- K线 ----
    体高 = np.maximum(np.abs(c - o), (h.max() - l.min()) * 0.0012)
    for 掩, 色 in ((涨, 涨色), (~涨, 跌色)):
        ax.vlines(x[掩], l[掩], h[掩], color=色, lw=0.9)
        ax.bar(x[掩], 体高[掩], bottom=np.minimum(o, c)[掩], width=0.68,
               color=色, edgecolor=色, lw=0.4)

    # ---- B/S 标记 ----
    幅 = h.max() - l.min()
    位 = {d.index[i]: i for i in range(len(d))}
    for dt, row in mk.iterrows():
        i = 位[dt]
        if row["类"] == "B":
            ax.text(i, l[i] - 幅 * 0.028, "B", color=B色, fontsize=13,
                    fontweight="bold", ha="center", va="top")
        else:
            ax.text(i, h[i] + 幅 * 0.028, "S", color=S色, fontsize=13,
                    fontweight="bold", ha="center", va="bottom")

    策年 = (1 + 收.loc[str(年)]).prod() - 1
    基年 = (1 + r.loc[str(年)].fillna(0)).prod() - 1
    nB = int((mk["类"] == "B").sum()) if len(mk) else 0
    nS = int((mk["类"] == "S").sum()) if len(mk) else 0
    ax.set_title(
        f"中证1000择时 v1.2 · {年}年     策略 {策年:+.1%}   持有{ETF} {基年:+.1%}   "
        f"均|仓| {pos.abs().mean():.0%}     B {nB} 次 / S {nS} 次"
        f"(仓位累计变动≥{标记阈值:.1f} 才标记)",
        color=字色, fontsize=15, pad=14)
    ax.set_ylabel(f"{ETF} 价格", color=字色, fontsize=11)
    ax.set_ylim(l.min() - 幅 * 0.09, h.max() + 幅 * 0.09)

    # ---- 仓位轨迹 ----
    v = pos.to_numpy(float)
    ax2.fill_between(x, v, 0, where=v >= 0, color=涨色, alpha=0.55, step="mid")
    ax2.fill_between(x, v, 0, where=v < 0, color="#2962ff", alpha=0.55, step="mid")
    ax2.plot(x, v, color=字色, lw=0.7, alpha=0.8)
    ax2.axhline(0, color=字色, lw=0.8)
    for dt in mk.index:
        ax2.axvline(位[dt], color="#787b86", lw=0.5, alpha=0.35)
    ax2.set_ylim(-1.15, 1.15)
    ax2.set_ylabel("仓位\n(红=多 蓝=空)", color=字色, fontsize=10)

    # ---- 月度刻度 ----
    月 = pd.Series(d.index.month, index=x)
    刻 = [i for i in x if i == 0 or 月[i] != 月[i - 1]]
    ax2.set_xticks(刻)
    ax2.set_xticklabels([d.index[i].strftime("%m月") for i in 刻], color=字色)
    ax2.set_xlim(-2, len(d) + 1)

    fig.tight_layout()
    fp = os.path.join(图目录, f"{年}.png")
    fig.savefig(fp, dpi=130, facecolor=底色)
    plt.close(fig)
    return fp, nB, nS, 策年, 基年


def main():
    os.makedirs(图目录, exist_ok=True)
    收, p, r, rf = 跑一遍(True)
    px = pd.read_csv(os.path.join(缓存, f"etf_{ETF}.csv"),
                     parse_dates=["date"]).set_index("date").sort_index()
    px = px.loc["2018-01-01":]
    p = p.loc["2018-01-01":]

    print(f"{'年':>6s}{'策略':>9s}{'持有':>9s}{'B':>5s}{'S':>5s}   文件")
    for 年 in range(2018, 2027):
        out = 画一年(年, px, p, 收, r)
        if out:
            fp, nB, nS, 策, 基 = out
            print(f"{年:>6d}{策:>9.1%}{基:>9.1%}{nB:>5d}{nS:>5d}   {os.path.basename(fp)}")
    print(f"\n落盘: {图目录}")


if __name__ == "__main__":
    main()
