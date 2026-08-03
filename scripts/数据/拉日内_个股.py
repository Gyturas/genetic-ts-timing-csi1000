# -*- coding: utf-8 -*-
"""拉个股 5m 日内数据(2019-2026)→ data/cache/intraday/stocks/{code}.parquet
6线程并行、逐股落盘可断点续传、失败重试3次。个股含订单流字段(buy_vol/sale_vol)。"""
import cjpy, pandas as pd, os, time, sys
from concurrent.futures import ThreadPoolExecutor

出 = "data/cache/intraday/stocks"
os.makedirs(出, exist_ok=True)
N股 = int(sys.argv[1]) if len(sys.argv) > 1 else 150
列 = ["时间","open","high","low","close","vol","amount","yclose","buy_vol","sale_vol"]

wt = pd.read_parquet("/Users/yilinzhou/Desktop/中证1000个股择时/中证1000数据/float32/中证1000权重.parquet")
wt.index = pd.to_datetime(wt.index)
在册 = wt.loc["2019-01-01":].notna().sum()
池 = 在册[在册 > 1000].sort_values(ascending=False).index.tolist()
import numpy as np
rng = np.random.default_rng(42)
选 = list(rng.choice(池, min(N股, len(池)), replace=False))     # 随机抽,避免"长期在册=大票"偏置
print(f"候选{len(池)}只 → 随机抽{len(选)}只", flush=True)

def 一只(code):
    fp = f"{出}/{code.replace('.','_')}.parquet"
    if os.path.exists(fp): return "skip"
    块 = []
    for y in range(2019, 2027):
        for 试 in range(3):
            try:
                x = cjpy.get_market_data(code=code, start=f"{y}-01-01", end=f"{y}-12-31", cycle="5m")
                if len(x): 块.append(x[[c for c in 列 if c in x.columns]])
                break
            except Exception:
                time.sleep(2 + 试 * 3)
    if not 块: return "fail"
    d = pd.concat(块, ignore_index=True)
    d["时间"] = pd.to_datetime(d["时间"])
    d.to_parquet(fp)
    return len(d)

t0 = time.time(); done = 0
with ThreadPoolExecutor(max_workers=6) as ex:
    for i, r in enumerate(ex.map(一只, 选)):
        done += 1
        if done % 10 == 0:
            el = time.time() - t0
            print(f"  {done}/{len(选)}  用时{el/60:.0f}min  预计剩余{el/done*(len(选)-done)/60:.0f}min", flush=True)
print(f"DONE 共{len(os.listdir(出))}只 用时{(time.time()-t0)/60:.0f}min", flush=True)
