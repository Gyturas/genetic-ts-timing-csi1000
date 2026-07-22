# -*- coding: utf-8 -*-
"""v3 定稿信号生成:原库(rank IC 挖矿) + cos IC 加权 + 元老明文剔除。

为什么是这个配置(全部有受控实验支撑,见 说明.md 汇总表):
  - 库用 rank IC 挖:cos 挖矿(v2)与无元老重挖都没有超过它——
    rank 是抗噪的选择压力,且元老在挖矿期充当残差闸的正交陪练(有它挖出的库更好);
  - 持仓用 cos IC 加权:cos 的分子就是因子单独交易的盈亏,对齐目标函数;
  - 元老明文剔除:9 个人工动量元老方向在 2013-14 预热期锁死为 -1,与 2015 后
    正向长动量相反,持仓期是纯拖累("好陪练、坏队员")。

流程:读 结果_26算子/因子逐日信号.csv.gz → 滤掉 elder/argarch → 临时存档目录
     → 调 引擎.cos_ic权重 重放 → 覆盖 数据/分资产逐日.csv。
之后 python -m 引擎.策略 应复现 年化24.9% / 夏普1.22 / 回撤-18.6% / 卡玛1.34。

用法: python -m 引擎.定稿信号
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile

import pandas as pd

根 = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
存档 = os.path.join(根, "结果_26算子")


def main():
    名 = pd.read_csv(os.path.join(根, "数据", "名册.csv"))
    老 = 名[名.kind.isin(["elder", "argarch"])]["编号"].tolist()
    存 = pd.read_csv(os.path.join(存档, "因子逐日信号.csv.gz"), parse_dates=["date"])
    n老 = int(存.因子.isin(老).sum())
    print(f"存档 {len(存):,} 行;剔除元老(全部13人,含已退役){n老:,} 行")

    tmp = tempfile.mkdtemp(prefix="v3定稿_")
    滤目录 = os.path.join(tmp, "滤后存档"); os.makedirs(滤目录)
    存[~存.因子.isin(老)].to_csv(os.path.join(滤目录, "因子逐日信号.csv.gz"),
                              index=False, compression="gzip")
    shutil.copy(os.path.join(存档, "state.pkl"), 滤目录)
    shutil.copytree(os.path.join(存档, "argarch"), os.path.join(滤目录, "argarch"))

    出 = os.path.join(tmp, "cos重放")
    r = subprocess.run([sys.executable, "-m", "引擎.cos_ic权重",
                        "--存档", 滤目录, "--输出", 出],
                       cwd=根, env=dict(os.environ, PYTHONPATH=根),
                       capture_output=True, text=True)
    assert r.returncode == 0, f"cos重放失败:\n{r.stderr[-800:]}"
    print([l for l in r.stdout.splitlines() if "日均" in l][0])

    目标 = os.path.join(根, "数据", "分资产逐日.csv")
    shutil.copy(os.path.join(出, "分资产逐日.csv"), 目标)
    print(f"已覆盖 {目标}")
    shutil.rmtree(tmp, ignore_errors=True)
    print("完成。请跑 python -m 引擎.策略 验证 24.9%/1.22。")


if __name__ == "__main__":
    main()
