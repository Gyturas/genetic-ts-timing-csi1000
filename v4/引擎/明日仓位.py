# -*- coding: utf-8 -*-
"""v4 实盘:给出下一交易日开盘的中证1000(512100)推荐仓位。

v4 的定稿信号全量重算只需约 2 分钟,故实盘不做增量延算(v3 曾因手工延算的
黄牌口径/存档缺行两处不同源翻车),每天直接全量重跑 定稿信号,再读最后一行。
先更新行情(data_cache 与 数据/行情 的 13 个文件)再跑本脚本。

用法: PYTHONPATH=. python -m 引擎.明日仓位
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd
from scipy.stats import norm

根 = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    末 = pd.read_csv(os.path.join(根, "data_cache", "idx_sh000852.csv"),
                    usecols=["date"])["date"].max()
    print(f"行情末日 {末}(若不是最新收盘日,请先更新行情再跑)")

    from 引擎.定稿信号 import main as 生成
    生成()

    π = pd.read_csv(os.path.join(根, "数据", "映射分位.csv"),
                    parse_dates=["date"]).set_index("date")
    S = pd.read_csv(os.path.join(根, "数据", "分资产逐日.csv"), index_col=0,
                    parse_dates=True)["512100_仓位"]
    p = pd.Series(np.tanh(norm.ppf(π["π_增强"])), index=π.index) \
        .where(π["π_增强"].notna(), 0.0)

    print("\n" + "=" * 62)
    print(f"最新收盘日 {π.index[-1].date()}(信号) → 下一交易日开盘执行")
    print("=" * 62)
    print(f"  组合信号 comb = {float(S.iloc[-1]):+.4f}   40日分位 π = {float(π['π_增强'].iloc[-1]):.3f}")
    print(f"  多空口径推荐仓位: {float(p.iloc[-1]):+.1%}")
    print(f"  纯多口径推荐仓位: {max(float(p.iloc[-1]), 0.0):+.1%}")
    print("\n近8个交易日仓位轨迹(多空):")
    for d, v in p.tail(8).items():
        print(f"  {d.date()}  {v:+.1%}")


if __name__ == "__main__":
    main()
