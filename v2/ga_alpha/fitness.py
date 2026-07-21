"""适应度：cos IC（v2）。

与 v1 的区别只有一处——把"因子好坏"的尺子从 rank IC 换成 cos IC：

  v1(rank IC): 因子值ₜ 与 收益ₜ₊₁ 的秩相关。只看排序对不对得上,丢掉信号强弱。
  v2(cos IC):  把因子过一遍**生产同款仓位映射**得到仓位 p,再算 cos(p, 收益ₜ₊₁)
               = Σ p·r / sqrt(Σp²·Σr²)。分子 Σp·r 就是"这个因子单独交易赚多少钱",
               既看方向对错、也看下注大小,直接对齐真实盈亏。

仓位映射复刻 引擎/指数日频引擎.py(RANK窗=120)+ 引擎/策略.py(映射窗=40):
    p = tanh(Φ⁻¹( ts_percentile( rank₁₂₀(因子)·2−1 , 40 ) ))

关键实现点:**p 在完整序列上算一次**,再按 train/valid 掩码取子段。
若先切片再滚动,valid 段头部会丢掉 train 段的历史,滚动分位与生产口径不一致。

其余(取绝对值选双向因子、复杂度罚、覆盖率门槛、HOF去相关用原始因子)全部同 v1。
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.stats import ConstantInputWarning, norm

warnings.filterwarnings("ignore", category=ConstantInputWarning)

from ga_alpha import expr
from ga_alpha.expr import Node

# 必须与 引擎/指数日频引擎.py 的 RANK窗、引擎/策略.py 的 映射窗 一致
RANK窗 = 120
映射窗 = 40
_每列最少 = 60          # 单列有效点少于此不计入 cos 均值(rolling 头部损失后至少要有一段)


def forward_returns(close: pd.DataFrame, horizon: int) -> pd.DataFrame:
    return close.pct_change(horizon).shift(-horizon)


def ts_rank_ic(factor: pd.DataFrame, fwd_ret: pd.DataFrame) -> tuple[float, float]:
    """v1 口径的 rank IC，保留备查/对照（v2 打分不再用它）。"""
    valid = factor.notna() & fwd_ret.notna()
    coverage = valid.to_numpy().mean()
    ics = factor.corrwith(fwd_ret, method="spearman")
    return float(ics.mean()), float(coverage)


def 信号到仓位(S: pd.DataFrame) -> pd.DataFrame:
    """第二层映射:ts_percentile(映射窗) → tanh(Φ⁻¹)。输入 S 已是 rank₁₂₀ 信号。
    引擎入库/记账阶段的 S 已过第一层,直接调这个,避免重复做 rank₁₂₀。"""
    π = (S.rolling(映射窗).rank(pct=True) - 0.5 / 映射窗).clip(1e-6, 1 - 1e-6)
    return pd.DataFrame(np.tanh(norm.ppf(π.to_numpy())), index=S.index, columns=S.columns)


def 到仓位(factor: pd.DataFrame) -> pd.DataFrame:
    """因子 → 生产同款单因子仓位 p。两层:rank₁₂₀ 后接 信号到仓位。完整序列上算,含滚动历史。"""
    S = factor.rolling(RANK窗).rank(pct=True) * 2 - 1
    return 信号到仓位(S)


def cos_ic(p: pd.DataFrame, fwd: pd.DataFrame) -> tuple[float, float]:
    """逐列 cos(p, fwd) 取均值，返回 (均值, 覆盖率)。p 已是仓位(可含符号)。

    覆盖率用**相对**口径:分母是该列有收益数据的天数,不是总行数。与 v1 注入的
    相对覆盖率IC 一致。否则晚上市的指数(cyb 2010、hongli 2008)在回看10年窗口里
    前几年本就没数据,绝对覆盖率被拉到0.7以下,整个 pop 被 min_coverage 全拒,HOF 空。
    """
    cos_列, cov_列 = [], []
    for c in p.columns:
        a = p[c].to_numpy()
        b = fwd[c].to_numpy() if c in fwd.columns else np.full(len(a), np.nan)
        有收益 = int(np.isfinite(b).sum())
        m = np.isfinite(a) & np.isfinite(b)
        cov_列.append(m.sum() / 有收益 if 有收益 > 0 else 0.0)   # 相对覆盖率
        if m.sum() < _每列最少:
            continue
        pv, rv = a[m], b[m]
        den = np.sqrt((pv * pv).sum() * (rv * rv).sum())
        if den > 0:
            cos_列.append(float((pv * rv).sum() / den))
    return (float(np.mean(cos_列)) if cos_列 else np.nan,
            float(np.mean(cov_列)) if cov_列 else 0.0)


@dataclass
class Individual:
    tree: Node
    key: str = ""
    train_ic: float = np.nan
    fitness: float = -np.inf
    factor: pd.DataFrame | None = field(default=None, repr=False)   # 原始因子值，供 HOF 去相关
    p: pd.DataFrame | None = field(default=None, repr=False)        # 仓位序列，供 valid_ic 复用

    def __post_init__(self):
        self.key = str(self.tree)


class Evaluator:
    """持有面板与日期切分，给个体打分。挖掘阶段只暴露 train 段的信息。"""

    def __init__(self, panels: dict[str, pd.DataFrame], cfg: dict):
        self.panels = panels
        self.fwd = forward_returns(panels["close"], cfg["fitness"]["horizon"])
        self.min_coverage = cfg["fitness"]["min_coverage"]
        self.parsimony = cfg["ga"]["parsimony"]

        idx = panels["close"].index
        train_end = pd.Timestamp(cfg["split"]["train_end"])
        valid_end = pd.Timestamp(cfg["split"]["valid_end"])
        self.train_mask = idx <= train_end
        self.valid_mask = (idx > train_end) & (idx <= valid_end)
        self.oos_mask = idx > valid_end

    def evaluate(self, ind: Individual) -> Individual:
        try:
            factor = expr.evaluate(ind.tree, self.panels)
        except Exception:
            return ind
        if not isinstance(factor, pd.DataFrame):
            return ind
        factor = factor.replace([np.inf, -np.inf], np.nan)

        p = 到仓位(factor)                       # 完整序列上算一次
        ic, coverage = cos_ic(p[self.train_mask], self.fwd[self.train_mask])
        if not np.isfinite(ic) or coverage < self.min_coverage:
            return ind

        ind.factor = factor
        ind.p = p
        ind.train_ic = ic
        ind.fitness = abs(ic) - self.parsimony * expr.size(ind.tree)
        return ind

    def valid_ic(self, ind: Individual) -> float:
        ic, _ = cos_ic(ind.p[self.valid_mask], self.fwd[self.valid_mask])
        return ic

    def oos_ic(self, ind_factor: pd.DataFrame) -> float:
        ic, _ = cos_ic(到仓位(ind_factor)[self.oos_mask], self.fwd[self.oos_mask])
        return ic
