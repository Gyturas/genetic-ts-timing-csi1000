# -*- coding: utf-8 -*-
"""个股日线滚动面板 + 截面叶子日更(v5.1 生产管线第一环)。

用途:①状态层的恐慌探测器 limit_net ②xsec 因子库求值所需的 16 叶子面板。
维护 data/cache/stocks_daily.parquet(滚动保留最近 ~400 交易日 × 全部历史成分),
每日增量拉当期成分股日线(cjpy,8线程约5分钟),再重算 data/archive/截面叶子.csv。

首次运行自动从个股项目的 parquet 母本引导(若存在),否则回补 400 个自然日。
用法: PYTHONPATH=src python -m csi1000.engine.update_stocks
"""
from __future__ import annotations

import os
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd

from csi1000 import paths

面板档 = os.path.join(paths.行情缓存, "stocks_daily.parquet")
叶子档 = os.path.join(paths.存档, "截面叶子.csv")
母本目录 = os.path.expanduser("~/Desktop/中证1000个股择时/中证1000数据/float32")
保留日 = 400            # 滚动窗:nhnl 需 250 日历史,留足余量


def _成分(cl_index=None) -> tuple[pd.DataFrame, list]:
    """返回(权重矩阵, 最新成分列表)。权重来自个股项目的 point-in-time 文件。"""
    wt = pd.read_parquet(os.path.join(母本目录, "中证1000权重.parquet"))
    wt.index = pd.to_datetime(wt.index)
    return wt, wt.iloc[-1].dropna().index.tolist()


def _引导() -> dict[str, pd.DataFrame]:
    """从母本 parquet 引导初始面板(只取最近 保留日)。"""
    出 = {}
    for k, f in [("close", "close"), ("open", "open"), ("vol", "vol"), ("amount", "amount")]:
        d = pd.read_parquet(os.path.join(母本目录, f + ".parquet"))
        d.index = pd.to_datetime(d.index)
        出[k] = d.tail(保留日).astype("float32")
    print(f"  从母本引导:{出['close'].shape[0]} 天 × {出['close'].shape[1]} 股"
          f"(至 {出['close'].index[-1].date()})")
    return 出


def _拉一只(args):
    code, 起 = args
    import cjpy
    for 试 in range(3):
        try:
            d = cjpy.get_market_data(code=code, start=起, end="2050-01-01",
                                     cycle="day", rate="后复权")   # 必须与母本同口径
            if "时间" not in d or not len(d):
                return code, None
            return code, pd.DataFrame({
                "date": pd.to_datetime(d["时间"]), "close": d["close"].astype(float),
                "open": d["open"].astype(float), "vol": d["vol"].astype(float),
                "amount": d["amount"].astype(float)}).set_index("date")
        except Exception:
            time.sleep(1 + 试)
    return code, None


def 更新面板() -> dict[str, pd.DataFrame]:
    """增量更新滚动面板并落盘。返回 {close/open/vol/amount: 日×股}。"""
    if os.path.exists(面板档):
        存 = pd.read_parquet(面板档)
        面 = {k: 存.xs(k, level=0, axis=1) for k in ("close", "open", "vol", "amount")}
        起日 = 面["close"].index.max() - pd.Timedelta(days=5)
    elif os.path.isdir(母本目录):
        面 = _引导()
        起日 = 面["close"].index.max() - pd.Timedelta(days=5)
    else:
        面 = {k: pd.DataFrame(dtype="float32") for k in ("close", "open", "vol", "amount")}
        起日 = pd.Timestamp.today() - pd.Timedelta(days=保留日 * 1.6)

    _, 成员 = _成分()
    起 = 起日.strftime("%Y-%m-%d")
    t0 = time.time()
    新 = {}
    with ThreadPoolExecutor(max_workers=8) as ex:
        for code, d in ex.map(_拉一只, [(c, 起) for c in 成员]):
            if d is not None and len(d):
                新[code] = d
    print(f"  拉取 {len(新)}/{len(成员)} 只({time.time()-t0:.0f}s)")

    for k in ("close", "open", "vol", "amount"):
        增 = pd.DataFrame({c: d[k] for c, d in 新.items()})
        旧 = 面[k]
        # 按【单元格】合并而非整行覆盖:新值优先,未拉取的股票/日期保留旧值。
        # 整行覆盖会把非当期成分股的历史数据抹成 NaN(掩码分母缩水 → 叶子断层)
        合 = 增.combine_first(旧) if len(旧) else 增
        面[k] = 合.sort_index().tail(保留日).astype("float32")
    宽 = pd.concat({k: v for k, v in 面.items()}, axis=1)
    宽.to_parquet(面板档)
    print(f"  面板 → {面['close'].shape[0]} 天 × {面['close'].shape[1]} 股,末日 {面['close'].index[-1].date()}")
    return 面


def 算叶子(面: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """按 docs/实验档案/02_截面叶子 的定稿口径重算 6 个截面叶子。"""
    cl, op, vo, am = 面["close"], 面["open"], 面["vol"], 面["amount"]
    wt, _ = _成分()
    列 = cl.columns
    成员 = wt.reindex(columns=列).notna().reindex(cl.index).ffill().fillna(False).astype(bool)
    持仓掩 = 成员.shift(1).fillna(False)                       # t 日用 t-1 日成分
    有效 = (vo > 0) & cl.notna() & cl.shift(1).notna()          # 剔停牌
    掩 = 持仓掩 & 有效
    r = cl.pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan)
    N = 掩.sum(axis=1).astype("float64")
    N = N.where(N > 1)                                         # N<=1 时熵/占比无定义
    # 涨跌停阈:主板 9.5%;创业板300/301(2020-08-24起)与科创688 为 19.5%
    阈 = pd.DataFrame(0.0949, index=cl.index, columns=列)
    创 = [c for c in 列 if c[:3] in ("300", "301")]
    科 = [c for c in 列 if c[:3] == "688"]
    if 创: 阈.loc["2020-08-24":, 创] = 0.1949
    if 科: 阈[科] = 0.1949
    高 = cl.rolling(250, min_periods=200).max(); 低 = cl.rolling(250, min_periods=200).min()
    w = am.where(掩).clip(lower=0)
    w总 = w.sum(axis=1).replace(0, np.nan)
    w = w.div(w总, axis=0).where(lambda x: x > 0)               # 0 权重不进熵(0·log0=0)
    叶 = pd.DataFrame({
        "breadth":   (r.gt(0) & 掩).sum(axis=1) / N,
        "disp":      r.where(掩).std(axis=1),
        "limit_net": ((r.ge(阈) & 掩).sum(axis=1) - (r.le(-阈) & 掩).sum(axis=1)) / N,
        "nhnl":      ((cl.ge(高 * 0.9999) & 掩).sum(axis=1) - (cl.le(低 * 1.0001) & 掩).sum(axis=1)) / N,
        "xskew":     r.where(掩).skew(axis=1),
        "amt_ent":   -(w * np.log(w)).sum(axis=1) / np.log(N),
    })
    叶 = 叶.replace([np.inf, -np.inf], np.nan)
    return 叶[N.fillna(0) > 500]


def main():
    面 = 更新面板()
    新叶 = 算叶子(面)
    if os.path.exists(叶子档):
        旧叶 = pd.read_csv(叶子档, parse_dates=["date"]).set_index("date")
        # 滚动面板只有 400 天,算不出自己最老那 250 天的 nhnl(250日新高新低需更长历史)。
        # 因此只信任【尾部】:存档末日往前留 5 个交易日的修订窗,更早一律以存档为准。
        旧尾 = 旧叶.index.max()
        分界 = 旧叶.index[max(0, 旧叶.index.get_loc(旧尾) - 5)]
        合 = pd.concat([旧叶[旧叶.index <= 分界], 新叶[新叶.index > 分界]]).sort_index()
    else:
        合 = 新叶
    合.index.name = "date"
    合.to_csv(叶子档)
    print(f"截面叶子 → {len(合)} 天,末日 {合.index[-1].date()}")
    print(f"  最新 limit_net={合['limit_net'].iloc[-1]:+.4f}  breadth={合['breadth'].iloc[-1]:.3f}")
    return str(合.index[-1].date())


if __name__ == "__main__":
    main()
