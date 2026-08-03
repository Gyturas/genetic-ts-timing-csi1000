# -*- coding: utf-8 -*-
"""从5m数据造日内结构特征,测对已定死方程 E[r]=φ₁r+β₁rv1+β₅rv5 的增量。"""
import numpy as np, pandas as pd, warnings, os, time
warnings.filterwarnings("ignore")
import cjpy
from arch.univariate import ARX, GARCH

缓 = "data/cache/intraday/zz1000_5m.parquet"
if os.path.exists(缓):
    d = pd.read_parquet(缓)
else:
    块 = []
    for y in range(2018, 2027):
        t0 = time.time()
        x = cjpy.get_market_data(code="000852.SH", start=f"{y}-01-01", end=f"{y}-12-31", cycle="5m")
        块.append(x); print(f"  {y}: {len(x)}行 {time.time()-t0:.0f}s", flush=True)
    d = pd.concat(块, ignore_index=True)
    d["时间"] = pd.to_datetime(d["时间"])
    d.to_parquet(缓)
d["时间"] = pd.to_datetime(d["时间"]); d["日"] = d["时间"].dt.date
d = d[d["时间"].dt.time > pd.Timestamp("09:25").time()]        # 去集合竞价bar
print(f"总计 {len(d)}行, {d['日'].nunique()}个交易日")

g = d.groupby("日")
r5 = d.groupby("日")["close"].pct_change()
d["r5"] = r5
特 = pd.DataFrame({
    "RV5m":   np.sqrt(g.apply(lambda x: (x["r5"].dropna()**2).sum())*1e4),      # 已实现波动(%)
    "BV":     np.sqrt(g.apply(lambda x: (x["r5"].abs()*x["r5"].abs().shift(1)).sum()*np.pi/2)*1e4),
    "OFI":    g.apply(lambda x: (x["buy_vol"].sum()-x["sale_vol"].sum())/(x["buy_vol"].sum()+x["sale_vol"].sum()+1)),
    "OFI尾":  g.apply(lambda x: (x["buy_vol"].tail(6).sum()-x["sale_vol"].tail(6).sum())/(x["buy_vol"].tail(6).sum()+x["sale_vol"].tail(6).sum()+1)),
    "日内收益": g.apply(lambda x: (x["close"].iloc[-1]/x["open"].iloc[0]-1)*100),
    "隔夜收益": g.apply(lambda x: (x["open"].iloc[0]/x["yclose"].iloc[0]-1)*100),
    "尾盘量占": g.apply(lambda x: x["vol"].tail(6).sum()/(x["vol"].sum()+1)),
    "委比":    g["wb"].mean(),
})
特.index = pd.to_datetime(特.index)
特["跳跃"] = (特.RV5m**2 - 特.BV**2).clip(lower=0)**0.5
特.to_csv("data/cache/intraday/日内特征.csv")

# —— 对已定死方程的增量检验 ——
zz = pd.read_csv("data/cache/idx_sh000852.csv", parse_dates=["date"]).set_index("date")
r = (np.log(zz["close"]).diff()*100).loc["2018":]; r = r[r.abs()<25].dropna()
标 = lambda x: (x-x.mean())/x.std()
基 = {"rv1": 标(np.sqrt(r**2).shift(1)), "rv5": 标(np.sqrt((r**2).rolling(5).mean()).shift(1))}
def 星(t): a=abs(t); return "***" if a>2.576 else "**" if a>1.96 else "*" if a>1.645 else ""
def 估(名, 额外):
    X = dict(基); X.update(额外)
    dd = pd.concat([r.rename("y"), pd.DataFrame(X)], axis=1).dropna()
    m = ARX(dd["y"], lags=1, x=dd[list(X)]); m.volatility = GARCH(1,1)
    f = m.fit(disp="off"); p,t = f.params, f.tvalues
    k1 = next(k for k in p.index if "[1]" in k and k not in X)
    出 = f"  {名:22s} φ₁={p[k1]:+.3f}({t[k1]:+.1f}) rv1={p['rv1']*100:+.0f}bp({t['rv1']:+.1f})"
    for k in 额外: 出 += f"  {k}={p[k]*100:+.1f}bp({t[k]:+.2f}){星(t[k])}"
    print(出, flush=True)
print("\n═══ 日内特征对基准方程的增量(2018起,GARCH扰动)═══")
估("基准(仅rv1+rv5)", {})
for k in ["RV5m","OFI","OFI尾","跳跃","隔夜收益","日内收益","尾盘量占","委比"]:
    估(f"+{k}", {k: 标(特[k].shift(1)).reindex(r.index)})
估("+全部微结构", {k: 标(特[k].shift(1)).reindex(r.index) for k in ["RV5m","OFI","OFI尾","跳跃"]})
# RV5m 能否取代 rv1
dd = pd.concat([r.rename("y"), pd.DataFrame({"RV5m":标(特["RV5m"].shift(1)).reindex(r.index),
                                            "rv5":基["rv5"]})], axis=1).dropna()
m = ARX(dd["y"], lags=1, x=dd[["RV5m","rv5"]]); m.volatility=GARCH(1,1); f=m.fit(disp="off")
p,t = f.params,f.tvalues
k1 = next(k for k in p.index if "[1]" in k and k not in ("RV5m","rv5"))
print(f"\n  用RV5m替代rv1: φ₁={p[k1]:+.3f}({t[k1]:+.1f})  RV5m={p['RV5m']*100:+.0f}bp(t={t['RV5m']:+.2f}){星(t['RV5m'])}")
print(f"  对照:原rv1 t=+5.12 (Spec D)")
print(f"  corr(RV5m, rv1日线) = {特['RV5m'].corr(np.sqrt(r**2).reindex(特.index)):+.3f}")
