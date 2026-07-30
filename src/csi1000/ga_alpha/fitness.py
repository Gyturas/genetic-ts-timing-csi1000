"""适应度：逐资产时序 RankIC（因子_t vs 收益_{t+1..t+h}）取均值，带覆盖率与复杂度约束。"""
from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.stats import ConstantInputWarning

# 退化个体（如常量因子）会触发 spearman 警告，靠 fitness 门槛淘汰即可，不必刷屏
warnings.filterwarnings("ignore", category=ConstantInputWarning)

from csi1000.ga_alpha import expr
from csi1000.ga_alpha.expr import Node


def forward_returns(close: pd.DataFrame, horizon: int) -> pd.DataFrame:
    return close.pct_change(horizon).shift(-horizon)


def ts_rank_ic(factor: pd.DataFrame, fwd_ret: pd.DataFrame) -> tuple[float, float]:
    """返回 (逐资产 spearman IC 的均值, 有效格覆盖率)。

    输入两张已对齐的 日期×资产 面板：
      - factor ：因子值面板（expr.evaluate 的产出），每格 = 某天某资产的因子值
      - fwd_ret：未来收益面板（forward_returns 的产出），已对齐成"今天 vs 今天之后的收益"
    衡量思路：逐只资产在时间轴上算"因子 vs 未来收益"的秩相关（IC），再对资产取平均，
    得到该因子的整体预测力；同时报告有多少格子真正可用（覆盖率）。
    """
    # 有效格：只有"因子和未来收益都不是 NaN"的格子才能参与计算（& 是逐格逻辑与）
    valid = factor.notna() & fwd_ret.notna()
    # 覆盖率：布尔面板求均值(True=1/False=0) = 有效格占全部格子的比例；太低说明因子大片为空、质量差
    coverage = valid.to_numpy().mean()
    # 核心：corrwith 默认逐列(逐资产)沿时间轴求相关；method="spearman"=秩相关(只看排序、抗异常值)
    # 结果 ics 是每只资产一个 IC 值的 Series，代表"该资产上因子对其未来收益的预测力"
    ics = factor.corrwith(fwd_ret, method="spearman")
    # 对所有资产的 IC 取平均 = 因子整体预测力；连同覆盖率一并返回（float() 把 numpy 标量转成原生 float）
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
