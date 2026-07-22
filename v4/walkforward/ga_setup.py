# -*- coding: utf-8 -*-
"""ga_alpha 引擎适配(仅本进程):
1) 叶子表换成 etf择时 版(去 turnover);
2) 预测目标换成 t+1 超额收益(减货基),与协议一致——
   monkeypatch ga_alpha.fitness.forward_returns,horizon 固定 1。
"""
from __future__ import annotations

import pandas as pd

import ga_alpha.expr as gexpr
import ga_alpha.fitness as gfit
import ga_alpha.ops as gops

from walkforward import config as C

_RF: pd.Series | None = None


def install(rf: pd.Series) -> None:
    global _RF
    _RF = rf
    gops.TERMINALS = C.TERMINALS
    gexpr.TERMINALS = C.TERMINALS

    def fwd_excess_close(close: pd.DataFrame, horizon: int) -> pd.DataFrame:
        assert horizon == 1, "协议锁定 t+1"
        ret = close.pct_change(fill_method=None)
        return ret.sub(_RF.reindex(close.index).fillna(0.0), axis=0).shift(-1)

    gfit.forward_returns = fwd_excess_close

    def ts_rank_ic_rel(factor: pd.DataFrame, fwd_ret: pd.DataFrame):
        """覆盖率改为相对口径:分母=有收益数据的格(品种起始日不齐,绝对口径会误杀)。"""
        valid = factor.notna() & fwd_ret.notna()
        denom = int(fwd_ret.notna().to_numpy().sum())
        coverage = valid.to_numpy().sum() / max(denom, 1)
        ics = factor.corrwith(fwd_ret, method="spearman")
        return float(ics.mean()), float(coverage)

    gfit.ts_rank_ic = ts_rank_ic_rel
