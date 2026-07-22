# -*- coding: utf-8 -*-
"""把熊概率序列 p_bear 延伸到指定日期,写入 数据/bear_prob.csv。

熊概率是 walk-forward 的:季频重估 elastic-net logistic,季内逐日预测,全程 OOS。
因此延伸到新日期不会改写历史值——本脚本会校验这一点,历史若被改动则报错退出。

p_bear 的用途:家族再倾斜(引擎/家族再倾斜.py)按熊强度 h 压反转/波动族、抬动量族。
p_bear 落在中性带内时 h=0,即熊增强不调制。

用法:
    python -m 引擎.更新熊概率                    # 延伸到行情数据的最后一天
    python -m 引擎.更新熊概率 --end 2026-07-17   # 延伸到指定日期
    python -m 引擎.更新熊概率 --check            # 只校验,不写盘
"""
from __future__ import annotations

import argparse
import os
import warnings

import pandas as pd

warnings.filterwarnings("ignore")

本目录 = os.path.dirname(os.path.abspath(__file__))
根 = os.path.dirname(本目录)
数据 = os.path.join(根, "数据")
存档 = os.path.join(数据, "bear_prob.csv")

历史最大容差 = 1e-3          # 重算与存档在重叠段允许的最大绝对差


def 行情末日() -> str:
    from csi1000 import paths  # 行情文件
    s = pd.read_csv(行情文件("idx_sh000300"), parse_dates=["date"])["date"]
    return str(s.max().date())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--end", default=None, help="延伸到该日期(默认取行情数据最后一天)")
    ap.add_argument("--check", action="store_true", help="只校验与存档是否一致,不写盘")
    a = ap.parse_args()
    end = a.end or 行情末日()

    from csi1000.engine.bear_factor import 熊概率序列
    print(f"重算熊概率至 {end} …")
    p = 熊概率序列(结束=end).dropna()

    if os.path.exists(存档):
        旧 = pd.read_csv(存档, parse_dates=["date"]).set_index("date")["p_bear"].dropna()
        共 = p.index.intersection(旧.index)
        差 = float((p[共] - 旧[共]).abs().max()) if len(共) else float("nan")
        print(f"与存档重叠 {len(共)} 天,最大绝对差 {差:.2e}")
        if len(共) and 差 > 历史最大容差:
            raise SystemExit(
                f"历史值被改动(最大差 {差:.2e} > 容差 {历史最大容差:.0e})。"
                f"walk-forward 下历史不应变化,请先排查行情数据是否被回改。")
        新增 = p.index.difference(旧.index)
        print(f"新增 {len(新增)} 天"
              + (f":{新增.min().date()} ~ {新增.max().date()}" if len(新增) else "(无)"))
    else:
        print("无存档,首次生成")

    if a.check:
        print("--check 模式,未写盘")
        return

    p.to_frame("p_bear").to_csv(存档)
    print(f"落盘: {存档}   末日 {p.index.max().date()}   p_bear={p.iloc[-1]:.4f}")
    print("近5日: " + "  ".join(f"{d.date()}={v:.4f}" for d, v in p.tail(5).items()))


if __name__ == "__main__":
    main()
