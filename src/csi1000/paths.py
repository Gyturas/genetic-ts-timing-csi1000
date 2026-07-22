# -*- coding: utf-8 -*-
"""全仓唯一的路径出口:目录约定集中在这里,其余代码一律 `from csi1000 import paths`。

仓库布局(src layout):
    src/csi1000/                 可导入代码(本文件所在)
    data/cache/                  行情缓存(原 v4/data_cache)
    data/archive/                信号与回测存档(原 v4/数据,含 行情/ 与 argarch/)
    results/final/               定稿成绩表与图(原 v4/结果)
    results/ops26/               26算子挖矿定稿存档(原 v4/结果_26算子)
    results/mining/              挖矿运行产物(原 结果_挖矿,gitignore)
    results/mine_no_elders/      无元老重挖产物(原 结果_无元老挖矿,gitignore)

根目录解析:本文件上溯三层(src/csi1000/paths.py → 仓库根);
可用环境变量 CSI1000_ROOT 覆盖(例如数据挪到仓库外时)。
"""
from __future__ import annotations

import os

根 = os.environ.get("CSI1000_ROOT") or os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

行情缓存 = os.path.join(根, "data", "cache")
存档 = os.path.join(根, "data", "archive")
结果 = os.path.join(根, "results", "final")
结果_26算子 = os.path.join(根, "results", "ops26")
挖矿输出 = os.path.join(根, "results", "mining")
无元老输出 = os.path.join(根, "results", "mine_no_elders")


def 行情目录() -> str:
    """优先用仓库自带的 data/archive/行情/(自包含),缺失时回退 data/cache/。"""
    自带 = os.path.join(存档, "行情")
    if os.path.isdir(自带):
        return 自带
    if os.path.isdir(行情缓存):
        return 行情缓存
    raise FileNotFoundError("找不到行情目录:既无 data/archive/行情/ 也无 data/cache/")


def 行情文件(tag: str) -> str:
    fp = os.path.join(行情目录(), tag + ".csv")
    if not os.path.exists(fp):
        raise FileNotFoundError(
            f"缺少行情文件 {tag}.csv(查找目录 {行情目录()})。"
            f"请确认 data/archive/行情/ 或 data/cache/ 已包含该文件。")
    return fp
