# -*- coding: utf-8 -*-
"""更新行情到最新交易日(data/cache 与 data/archive/行情 两处)。

指数用不复权拼接(close 对齐校验),ETF 用前复权收益率桥接(重叠段收益一致性校验),
成交量按整数倍单位对齐。接口:东方财富历史K线。周末/节假日拉取自然无新增。

用法: PYTHONPATH=src python -m csi1000.engine.update_market
"""
from __future__ import annotations

import http.client
import json
import os
import time
import urllib.request

import pandas as pd

from csi1000 import paths

指数 = {"idx_sh000300": "1.000300", "idx_sh000905": "1.000905", "idx_sh000852": "1.000852",
       "idx_sz399006": "0.399006", "idx_sh000688": "1.000688", "idx_sh000922": "1.000922"}
ETF = {"etf_510300": "1.510300", "etf_510500": "1.510500", "etf_512100": "1.512100",
       "etf_159915": "0.159915", "etf_588000": "1.588000", "etf_510880": "1.510880",
       "etf_511880": "1.511880"}


def _拉(secid: str, fqt: int) -> pd.DataFrame:
    """用 urllib(内置,不依赖外部 curl)拉行情——双击 .command 时 PATH 为空,curl 不可用。"""
    url = (f"https://push2his.eastmoney.com/api/qt/stock/kline/get?secid={secid}"
           f"&fields1=f1&fields2=f51,f52,f53,f54,f55,f56,f57&klt=101&fqt={fqt}&beg=20250101&end=20500101")
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0",
                                               "Connection": "close"})
    最后错 = None
    for _ in range(3):                       # 东财偶发 IncompleteRead / 空返回,重试3次
        try:
            with urllib.request.urlopen(req, timeout=30) as r:
                out = r.read().decode("utf-8")
            data = json.loads(out).get("data")
            if data and data.get("klines"):
                d = pd.DataFrame(
                    [x.split(",") for x in data["klines"]],
                    columns=["date", "open", "close", "high", "low", "volume", "amount"])
                d["date"] = pd.to_datetime(d["date"])
                for c in ["open", "close", "high", "low", "volume", "amount"]:
                    d[c] = d[c].astype(float)
                return d.set_index("date")
            最后错 = "接口返回空"
        except (http.client.IncompleteRead, urllib.error.URLError,
                json.JSONDecodeError, TimeoutError) as e:
            最后错 = type(e).__name__
        time.sleep(1.5)
    raise RuntimeError(f"拉取失败(secid={secid}): {最后错};可能非交易时段或网络受限")


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
    for t, s, 是指数 in ([(t, s, True) for t, s in 指数.items()]
                       + [(t, s, False) for t, s in ETF.items()]):
        try:
            _更新(t, s, 是指数, 报告)
        except Exception as e:                # 单标的失败不中断全局(信号可退回已有数据)
            报告.append(f"{t}: ✗ {type(e).__name__}")
    末日 = pd.read_csv(os.path.join(paths.行情缓存, "idx_sh000852.csv"),
                     usecols=["date"])["date"].max()
    print(f"行情更新完毕,中证1000末日 {末日}")
    for r in 报告:
        print(" ", r)
    return 末日


if __name__ == "__main__":
    main()
