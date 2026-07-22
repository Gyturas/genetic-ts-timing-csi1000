# -*- coding: utf-8 -*-
"""仓位映射函数对比:clip(定稿) vs tanh 及其缩放变体。

映射都作用在同一个信号上:π = 信号的滚动分位(映射窗,v1.2=40日), z = Φ⁻¹(π)
  clip(定稿):  p = clip(z, -1, 1)          —— 84.1分位即饱和,平顶
  tanh:       p = tanh(z)                 —— 平滑,永不饱和,处处小于clip
  tanh(k·z):  p = tanh(k·z)  k=1.5/2.0    —— 加陡,恢复仓位规模
理论均|仓|: clip≈0.68, tanh≈0.57(tanh(z)<min(|z|,1) 恒成立⇒仓位必然更小)
用法: python -m 引擎.映射对比   (cwd=中证1000/v1)
"""
from __future__ import annotations

import os
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import norm

warnings.filterwarnings("ignore")
plt.rcParams["font.family"] = ["Arial Unicode MS", "PingFang SC", "Heiti SC"]
plt.rcParams["axes.unicode_minus"] = False

from csi1000.engine.strategy import 建信号, 结算, 统计, 读价, 映射窗, 结果, 口径起, ETF

图目录 = os.path.join(结果, "图")
os.makedirs(图目录, exist_ok=True)

映射表 = {
    "clip(定稿)": lambda z: np.clip(z, -1, 1),
    "tanh": np.tanh,
    "tanh(1.5z)": lambda z: np.tanh(1.5 * z),
    "tanh(2z)": lambda z: np.tanh(2.0 * z),
}


def main():
    S = 建信号(True)
    r = 读价(f"etf_{ETF}").pct_change().reindex(S.index)
    rf = 读价("etf_511880").pct_change().clip(lower=0).reindex(S.index).fillna(0)
    π = (S.rolling(映射窗).rank(pct=True) - 0.5 / 映射窗).clip(1e-6, 1 - 1e-6)
    z = pd.Series(norm.ppf(π), index=S.index)

    行, 曲线, 仓集 = [], {}, {}
    for 名, f in 映射表.items():
        p = pd.Series(f(z.to_numpy()), index=S.index) \
            .where(S.abs().gt(1e-12), 0.0).where(r.notna(), 0.0).fillna(0)
        收 = 结算(p, r, rf)
        行.append((名, 统计(收, rf, p)))
        曲线[名], 仓集[名] = 收, p
    行.append(("", {}))
    行.append(("持有512100", 统计(r.fillna(0), rf)))
    行.append(("", {}))
    for y in range(2018, 2027):
        if len(曲线["clip(定稿)"].loc[str(y)]) > 20:
            for 名 in 映射表:
                行.append((f"{y}·{名}", 统计(曲线[名].loc[str(y)], rf, 起=f"{y}-01-01")))
            行.append(("", {}))

    表 = pd.DataFrame([v for _, v in 行], index=[k for k, _ in 行]) \
        .reindex(columns=["年化", "夏普", "最大回撤", "卡玛", "均|仓|"])
    fmt = 表.copy()
    for c in fmt.columns:
        fmt[c] = fmt[c].apply(lambda v: (f"{v:.2f}" if c in ("夏普", "卡玛") else f"{v*100:.1f}%")
                              if isinstance(v, float) and np.isfinite(v) else "")
    fmt.to_csv(os.path.join(结果, "映射对比表.csv"))
    print(fmt.head(8).to_string())

    # ---- 图:映射函数形状 + 净值 ----
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    zz = np.linspace(-3, 3, 400)
    for 名, f in 映射表.items():
        axes[0].plot(zz, f(zz), lw=1.8, label=名)
    axes[0].axhline(0, color="k", lw=0.5); axes[0].axvline(0, color="k", lw=0.5)
    axes[0].set_xlabel("z = Φ⁻¹(分位)"); axes[0].set_ylabel("仓位")
    axes[0].set_title("映射函数形状"); axes[0].legend(); axes[0].grid(alpha=0.3)
    seg = slice(口径起, None)
    for 名 in 映射表:
        axes[1].plot((1 + 曲线[名].loc[seg]).cumprod(), lw=1.6, label=名)
    axes[1].plot((1 + r.loc[seg].fillna(0)).cumprod(), lw=1.1, alpha=0.7,
                 color="#7f8c8d", label="持有512100")
    axes[1].set_yscale("log"); axes[1].set_title("净值对比(对数轴,2018起)")
    axes[1].legend(); axes[1].grid(alpha=0.3, which="both")
    fig.tight_layout(); fig.savefig(os.path.join(图目录, "映射对比图.png"), dpi=140)
    print("\n落盘:", os.path.join(图目录, "映射对比图.png"))


if __name__ == "__main__":
    main()
