# -*- coding: utf-8 -*-
"""截面结构叶子注入:6个截面序列(广播到全资产列)+ skew120(逐资产原生)。
数据源 data/archive/截面叶子.csv(由个股面板构造,构造脚本存 scripts/构造截面叶子.py)。
挖矿与评估共用本模块,保证两侧面板一致。"""
from __future__ import annotations

import os

import pandas as pd

from csi1000 import paths

截面名 = ("breadth", "disp", "limit_net", "nhnl", "xskew", "amt_ent")
新叶子 = 截面名 + ("skew120",)


def 注入(面板: dict) -> dict:
    """就地把新叶子加进 A股宽基 panels;返回同一 dict。"""
    日历 = 面板["close"].index
    列 = 面板["close"].columns
    叶 = pd.read_csv(os.path.join(paths.存档, "截面叶子.csv"),
                    parse_dates=["date"]).set_index("date")
    for 名 in 截面名:
        s = 叶[名].reindex(日历)                      # 不ffill:缺日即NaN,交给覆盖率惩罚
        面板[名] = pd.DataFrame({c: s for c in 列})
    面板["skew120"] = 面板["returns"].rolling(120).skew()
    return 面板
