# -*- coding: utf-8 -*-
"""更新行情到最新交易日(data/cache 与 data/archive/行情 两处)。

指数用不复权拼接(close 对齐校验),ETF 用前复权收益率桥接(重叠段收益一致性校验),
成交量按整数倍单位对齐。接口:东方财富历史K线。周末/节假日拉取自然无新增。

用法: PYTHONPATH=src python -m csi1000.engine.update_market
"""
from __future__ import annotations

import json
import os
import subprocess

import pandas as pd

from csi1000 import paths

指数 = {"idx_sh000300": "1.000300", "idx_sh000905": "1.000905", "idx_sh000852": "1.000852",
       "idx_sz399006": "0.399006", "idx_sh000688": "1.000688", "idx_sh000922": "1.000922"}
ETF = {"etf_510300": "1.510300", "etf_510500": "1.510500", "etf_512100": "1.512100",
       "etf_159915": "0.159915", "etf_588000": "1.588000", "etf_510880": "1.510880",
       "etf_511880": "1.511880"}


def _拉(secid: str, fqt: int) -> pd.DataFrame:
    url = (f"https://push2his.eastmoney.com/api/qt/stock/kline/get?secid={secid}"
           f"&fields1=f1&fields2=f51,f52,f53,f54,f55,f56,f57&klt=101&fqt={fqt}&beg=20250101&end=20500101")
    out = subprocess.run(["curl", "-sm", "30", "-A", "Mozilla/5.0", url],
                         capture_output=True, text=True).stdout
    d = pd.DataFrame([x.split(",") for x in json.loads(out)["data"]["klines"]],
                     columns=["date", "open", "close", "high", "low", "volume", "amount"])
    d["date"] = pd.to_datetime(d["date"])
    for c in ["open", "close", "high", "low", "volume", "amount"]:
        d[c] = d[c].astype(float)
    return d.set_index("date")


def _更新(tag: str, secid: str, 是指数: bool, 报告: list) -> None:
    cache = paths.行情缓存
    arc = os.path.join(paths.存档, "行情")
    目录s = [cache] + ([arc] if os.path.exists(os.path.join(arc, tag + ".csv")) else [])
    末 = None
    for 目录 in 目录s:
        fp = os.path.join(目录, tag + ".csv")
        if not os.path.exists(fp):
            continue
        旧 = pd.read_csv(fp, parse_dates=["date"]).set_index("date")
        if 是指数:
            新 = _拉(secid, 0)
            共 = 旧.index.intersection(新.index)[-4:]
            if len(共) < 2 or (旧.loc[共, "close"] / 新.loc[共, "close"] - 1).abs().max() > 1e-3:
                报告.append(f"{tag}: ✗close口径不符,跳过"); return
            比v = (旧.loc[共, "volume"] / 新.loc[共, "volume"]).median() if "volume" in 旧 else 1
            if not (0.99 < 比v < 1.01) and 比v > 0 and 0.99 < 比v / round(比v) < 1.01:
                新["volume"] *= round(比v)
            增 = 新[新.index > 旧.index.max()][list(旧.columns)]
        else:
            权 = _拉(secid, 1)
            共 = 旧.index.intersection(权.index)[-4:]
            r旧 = 旧.loc[共, "close"].pct_change().dropna()
            r新 = 权.loc[共, "close"].pct_change().dropna()
            if len(r旧) < 2 or (r旧 - r新).abs().max() > 1e-3:
                报告.append(f"{tag}: ✗收益口径不符,跳过"); return
            m = 旧.index.max(); 锚 = float(旧.loc[m, "close"]); 行 = []
            for d in 权.index[权.index > m]:
                rr = float(权.loc[d, "close"]) / float(权["close"].shift(1).loc[d]) - 1
                锚 *= (1 + rr); b = 锚 / float(权.loc[d, "close"])
                row = {"date": d, "open": 权.loc[d, "open"] * b, "high": 权.loc[d, "high"] * b,
                       "low": 权.loc[d, "low"] * b, "close": 锚}
                if "volume" in 旧.columns:
                    row["volume"] = 权.loc[d, "volume"] * 100
                if "amount" in 旧.columns:
                    row["amount"] = 权.loc[d, "amount"]
                行.append(row)
            增 = pd.DataFrame(行).set_index("date")[list(旧.columns)] if 行 else pd.DataFrame()
        if len(增) == 0:
            末 = 旧.index.max().date()
            continue
        pd.concat([旧, 增]).to_csv(fp)
        末 = 增.index.max().date()
    报告.append(f"{tag}: → {末}")


def main() -> str:
    报告 = []
    for t, s in 指数.items():
        _更新(t, s, True, 报告)
    for t, s in ETF.items():
        _更新(t, s, False, 报告)
    末日 = pd.read_csv(os.path.join(paths.行情缓存, "idx_sh000852.csv"),
                     usecols=["date"])["date"].max()
    print(f"行情更新完毕,末日 {末日}")
    for r in 报告:
        print(" ", r)
    return 末日


if __name__ == "__main__":
    main()
