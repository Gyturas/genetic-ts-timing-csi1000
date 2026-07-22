"""适应度：逐资产时序 RankIC（因子_t vs 收益_{t+1..t+h}）取均值，带覆盖率与复杂度约束。"""
from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.stats import ConstantInputWarning

# 退化个体（如常量因子）会触发 spearman 警告，靠 fitness 门槛淘汰即可，不必刷屏
warnings.filterwarnings("ignore", category=ConstantInputWarning)

from ga_alpha import expr
from ga_alpha.expr import Node


def forward_returns(close: pd.DataFrame, horizon: int) -> pd.DataFrame:
    return close.pct_change(horizon).shift(-horizon)


def ts_rank_ic(factor: pd.DataFrame, fwd_ret: pd.DataFrame) -> tuple[float, float]:
    """返回 (逐资产 spearman IC 的均值, 有效格覆盖率)。"""
    valid = factor.notna() & fwd_ret.notna()
    coverage = valid.to_numpy().mean()
    ics = factor.corrwith(fwd_ret, method="spearman")
    return float(ics.mean()), float(coverage)


@dataclass
class Individual:
    tree: Node
    key: str = ""
    train_ic: float = np.nan
    fitness: float = -np.inf
    factor: pd.DataFrame | None = field(default=None, repr=False)

    def __post_init__(self):
        self.key = str(self.tree)


class Evaluator:
    """持有面板与日期切分，负责给个体打分。挖掘阶段只暴露 train 段的信息。"""

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
            return ind  # 数值异常的个体保持 -inf
        if not isinstance(factor, pd.DataFrame):
            return ind
        factor = factor.replace([np.inf, -np.inf], np.nan)

        ic, coverage = ts_rank_ic(factor[self.train_mask], self.fwd[self.train_mask])
        if not np.isfinite(ic) or coverage < self.min_coverage:
            return ind

        ind.factor = factor
        ind.train_ic = ic
        ind.fitness = abs(ic) - self.parsimony * expr.size(ind.tree)
        return ind

    def valid_ic(self, ind: Individual) -> float:
        ic, _ = ts_rank_ic(ind.factor[self.valid_mask], self.fwd[self.valid_mask])
        return ic

    def oos_ic(self, ind_factor: pd.DataFrame) -> float:
        ic, _ = ts_rank_ic(ind_factor[self.oos_mask], self.fwd[self.oos_mask])
        return ic
