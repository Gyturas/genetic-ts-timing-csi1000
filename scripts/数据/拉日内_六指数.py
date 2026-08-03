# -*- coding: utf-8 -*-
"""拉六指数 5m 日内数据 2018-2026 → data/cache/intraday/{tag}_5m.parquet"""
import cjpy, pandas as pd, os, time
指 = {"hs300":"000300.SH","zz500":"000905.SH","zz1000":"000852.SH",
     "cyb":"399006.SZ","kc50":"000688.SH","hongli":"000922.SH"}
out = "data/cache/intraday"
os.makedirs(out, exist_ok=True)
for tag, code in 指.items():
    fp = f"{out}/{tag}_5m.parquet"
    if os.path.exists(fp):
        print(f"{tag}: 已存在,跳过", flush=True); continue
    块 = []
    for y in range(2018, 2027):
        for 试 in range(3):
            try:
                x = cjpy.get_market_data(code=code, start=f"{y}-01-01", end=f"{y}-12-31", cycle="5m")
                块.append(x); break
            except Exception as e:
                if 试 == 2: print(f"  {tag} {y} 失败: {str(e)[:40]}", flush=True)
                time.sleep(3)
    if not 块: continue
    d = pd.concat(块, ignore_index=True)
    d["时间"] = pd.to_datetime(d["时间"])
    d[["时间","open","high","low","close","vol","amount","yclose"]].to_parquet(fp)
    print(f"{tag}: {len(d)}行 {d['时间'].min().date()}~{d['时间'].max().date()}", flush=True)
print("DONE", flush=True)
