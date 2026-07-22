# -*- coding: utf-8 -*-
"""仓库内统一的行情目录解析:优先用仓库自带的 数据/行情/,缺失时回退到父项目 data_cache。

这样 clone 下来即可独立运行,同时在原项目树里也能用更全的公共缓存。
"""
from __future__ import annotations

import os

_本 = os.path.dirname(os.path.abspath(__file__))
根 = os.path.dirname(_本)                                    # …/中证1000/v1


def 行情目录() -> str:
    自带 = os.path.join(根, "数据", "行情")
    if os.path.isdir(自带):
        return 自带
    回退 = os.path.join(os.path.dirname(os.path.dirname(根)), "data_cache")
    if os.path.isdir(回退):
        return 回退
    raise FileNotFoundError("找不到行情目录:既无 数据/行情/ 也无父项目 data_cache/")


def 行情文件(tag: str) -> str:
    fp = os.path.join(行情目录(), tag + ".csv")
    if not os.path.exists(fp):
        raise FileNotFoundError(
            f"缺少行情文件 {tag}.csv(查找目录 {行情目录()})。"
            f"若在独立仓库中运行,请确认 数据/行情/ 已包含该文件。")
    return fp
