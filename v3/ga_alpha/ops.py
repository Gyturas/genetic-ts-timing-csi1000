"""算子集：全部作用在 日期×资产 的 DataFrame 面板上，纯向量化。

约定：任何算子都不许看未来（只用 t 及之前的行），保证挖出的表达式天然无前视。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd

EPS = 1e-8
WINDOWS = (3, 5, 10, 20, 40, 60, 120, 244)


def _protected_div(a: pd.DataFrame, b: pd.DataFrame) -> pd.DataFrame:
    """受保护除法：|分母| <= EPS 的位置结果记 NaN，而不是产生 inf。

    GA 随机拼出的表达式随时可能除零；返回 NaN 让 fitness 的覆盖率约束去惩罚，
    比经典 GP 里"除零返回 1"的做法更诚实，也避免 inf 污染下游 rolling 计算。
    """
    return a / b.where(b.abs() > EPS)


def _signed_log(a: pd.DataFrame) -> pd.DataFrame:
    """带符号对数：sign(a) * log(1+|a|)，保留正负、压缩量级。

    对负数和 0 都安全（普通 log 对负数无定义、在 0 处发散），
    用于把 volume/amount 这类量级悬殊的字段压到与价格类字段可比的范围。
    """
    return np.sign(a) * np.log1p(a.abs())


def _ts_zscore(a: pd.DataFrame, w: int) -> pd.DataFrame:
    """滚动 z-score：(当前值 - 滚动均值) / 滚动标准差。

    分母走受保护除法：窗口内数值恒定时 std 为 0，普通除法会爆 inf。
    """
    r = a.rolling(w)
    return _protected_div(a - r.mean(), r.std())


def _signed_sqrt(a: pd.DataFrame) -> pd.DataFrame:
    """保符号开方：sign(a)·sqrt(|a|)，与 slog 互补的量纲压缩。"""
    return np.sign(a) * np.sqrt(a.abs())


def _ts_decay_linear(a: pd.DataFrame, w: int) -> pd.DataFrame:
    """线性衰减加权均值（最近权重 w，最远权重 1）。闭式滚动实现，无逐窗循环。

    Σ_{j=0..w-1}(w-j)·a_{t-j} = (w-t)·P_t + M_t，其中 P=滚动和、M=位置加权滚动和。
    """
    pos = np.arange(len(a), dtype=float)
    P = a.rolling(w).sum()
    M = a.mul(pos, axis=0).rolling(w).sum()
    num = P.mul(w - pos, axis=0) + M
    return num / (w * (w + 1) / 2)


def _ts_argmax(a: pd.DataFrame, w: int) -> pd.DataFrame:
    """距窗口内最大值的天数（0=今天就是最高点）。逐列 sliding window，仍为 numpy 向量化。"""
    from numpy.lib.stride_tricks import sliding_window_view
    out = np.full(a.shape, np.nan)
    vals = a.to_numpy()
    for j in range(vals.shape[1]):
        col = vals[:, j]
        if len(col) < w:
            continue
        sw = sliding_window_view(col, w)
        with np.errstate(invalid="ignore"):
            am = np.nanargmax(np.where(np.isnan(sw), -np.inf, sw), axis=1)
        out[w - 1:, j] = (w - 1) - am
        out[np.isnan(col), j] = np.nan
    return pd.DataFrame(out, index=a.index, columns=a.columns)


def _ts_argmin(a: pd.DataFrame, w: int) -> pd.DataFrame:
    return _ts_argmax(-a, w)


def _ts_tsi(a: pd.DataFrame, w: int) -> pd.DataFrame:
    """华泰口径趋势强度：窗口内差分的 均值/标准差（漂移 t 统计量）。"""
    d = a.diff()
    r = d.rolling(w)
    return _protected_div(r.mean(), r.std())


def _ts_er(a: pd.DataFrame, w: int) -> pd.DataFrame:
    """带符号 Kaufman 效率系数：位移 / 路程。abs 后即经典 ER∈[0,1]。"""
    disp = a - a.shift(w)
    path = a.diff().abs().rolling(w).sum()
    return _protected_div(disp, path)


def _ts_slope(a: pd.DataFrame, w: int) -> pd.DataFrame:
    """窗口内对时间回归的斜率（趋势斜率），闭式解 Cov(y,k)/Var(k)。"""
    pos = np.arange(len(a), dtype=float)
    P = a.rolling(w).mean()
    M = a.mul(pos, axis=0).rolling(w).mean()
    kbar = pd.Series(pos, index=a.index).rolling(w).mean()
    cov = M.sub(P.mul(kbar, axis=0))
    var_k = (w * w - 1) / 12.0
    return cov / var_k


def _ts_ac1(a: pd.DataFrame, w: int) -> pd.DataFrame:
    """滚动滞后1自相关(局部AR(1)系数):测'接力/打脸'状态。"""
    return a.rolling(w).corr(a.shift(1))


def _ts_cj(a: pd.DataFrame, w: int) -> pd.DataFrame:
    """滚动符号持续率(CJ检验的滚动版):相邻两期差分同号的比例。"""
    s = np.sign(a.diff())
    same = (s * s.shift(1) > 0).astype(float).where(s.notna() & s.shift(1).notna())
    return same.rolling(w).mean()


def _ts_vr5(a: pd.DataFrame, w: int) -> pd.DataFrame:
    """滚动5期方差比(VR检验的滚动版):>1接力组织,≈1随机堆积,<1回归。"""
    v1 = a.diff().rolling(w).var()
    v5 = a.diff(5).rolling(w).var()
    return _protected_div(v5, 5.0 * v1)


@dataclass(frozen=True)
class OpSpec:
    arity: int          # 子节点个数
    windowed: bool      # 是否带回看窗口参数
    fn: Callable        # (子面板..., [window]) -> 面板


OPS: dict[str, OpSpec] = {
    # 一元
    "neg":       OpSpec(1, False, lambda a: -a),
    "abs":       OpSpec(1, False, lambda a: a.abs()),
    "sign":      OpSpec(1, False, np.sign),
    "slog":      OpSpec(1, False, _signed_log),
    # 二元
    "add":       OpSpec(2, False, lambda a, b: a + b),
    "sub":       OpSpec(2, False, lambda a, b: a - b),
    "mul":       OpSpec(2, False, lambda a, b: a * b),
    "div":       OpSpec(2, False, _protected_div),
    # 时序一元（带窗口）
    "ts_mean":   OpSpec(1, True, lambda a, w: a.rolling(w).mean()),
    "ts_std":    OpSpec(1, True, lambda a, w: a.rolling(w).std()),
    "ts_min":    OpSpec(1, True, lambda a, w: a.rolling(w).min()),
    "ts_max":    OpSpec(1, True, lambda a, w: a.rolling(w).max()),
    "ts_sum":    OpSpec(1, True, lambda a, w: a.rolling(w).sum()),
    "ts_rank":   OpSpec(1, True, lambda a, w: a.rolling(w).rank(pct=True)),
    "ts_zscore": OpSpec(1, True, _ts_zscore),
    "delay":     OpSpec(1, True, lambda a, w: a.shift(w)),
    "delta":     OpSpec(1, True, lambda a, w: a.diff(w)),
    "ema":       OpSpec(1, True, lambda a, w: a.ewm(span=w, min_periods=w).mean()),
    "ts_decay_linear": OpSpec(1, True, _ts_decay_linear),
    "ts_argmax": OpSpec(1, True, _ts_argmax),
    "ts_argmin": OpSpec(1, True, _ts_argmin),
    "ts_tsi":    OpSpec(1, True, _ts_tsi),
    "ts_er":     OpSpec(1, True, _ts_er),
    "ts_slope":  OpSpec(1, True, _ts_slope),
    # 结构传感（快慢手方案首发:滞后结构的滚动估计量）
    "ts_ac1":    OpSpec(1, True, _ts_ac1),
    "ts_cj":     OpSpec(1, True, _ts_cj),
    "ts_vr5":    OpSpec(1, True, _ts_vr5),
    # 一元（补充）
    "ssqrt":     OpSpec(1, False, _signed_sqrt),
    # 时序二元（带窗口）
    "ts_corr":   OpSpec(2, True, lambda a, b, w: a.rolling(w).corr(b)),
}

# 个股版叶子：turnover 来自东财换手率；entropy120 / beta120 为预计算面板（贵的统计量进叶子）
TERMINALS = ("open", "high", "low", "close", "volume", "amount", "returns",
             "turnover", "entropy120", "beta120")
