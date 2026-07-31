# -*- coding: utf-8 -*-
"""构造截面结构叶子(计划v1):宽度/离散度/涨停温度计/新高新低差/截面偏度/成交额熵。
池:2014-11起中证1000成分(t用t-1成员);之前用代理池(流通市值=amount/(turn/100),
月末全A排801-1800)。停牌剔除;涨跌停阈分板块分年代。输出→主repo data/archive/截面叶子.csv"""
import numpy as np, pandas as pd, os
本 = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
D = os.path.join(本, "中证1000数据", "float32")
载 = lambda f: pd.read_parquet(os.path.join(D, f + ".parquet"))
cl, vo, am, tu, wt = 载("close"), 载("vol"), 载("amount"), 载("turn"), 载("中证1000权重")
for x in (cl, vo, am, tu, wt): x.index = pd.to_datetime(x.index)
列 = cl.columns
vo, am, tu = vo.reindex(columns=列), am.reindex(columns=列), tu.reindex(columns=列)
vo, am, tu = [x.reindex(cl.index) for x in (vo, am, tu)]

r = cl.pct_change(fill_method=None)
有效 = (vo > 0) & cl.notna() & cl.shift(1).notna()

# —— 池掩码:真实成分(2014-11+) + 代理池(之前,流通市值月末排801-1800) ——
真 = wt.reindex(columns=列).notna().reindex(cl.index).ffill().fillna(False)
起真 = pd.Timestamp("2014-11-03")
mcap = (am / (tu / 100)).where(有效)
月末 = cl.index.to_series().groupby(cl.index.to_period("M")).last()
代理 = pd.DataFrame(False, index=cl.index, columns=列)
上月单 = None
for i, d in enumerate(月末):
    if d >= 起真: break
    v = mcap.loc[d].dropna()
    if len(v) < 1200: continue
    池 = v.sort_values(ascending=False).index[800:1800]
    if i + 1 < len(月末):
        lo = cl.index[cl.index > d][0] if (cl.index > d).any() else None
        hi = 月末.iloc[i + 1]
        if lo is not None:
            代理.loc[lo:hi, 池] = True
掩 = 代理.where(cl.index.to_series().lt(起真), 真).astype(bool)
成员 = 掩.shift(1).fillna(False) & 有效

def 行统(x, m, f):
    return f(x.where(m))

N = 成员.sum(axis=1)
breadth = (r.gt(0) & 成员).sum(axis=1) / N
disp = r.where(成员).std(axis=1)
xskew = r.where(成员).skew(axis=1)
# 涨跌停阈:主板9.5%;创业板300/301自2020-08-24、科创688全程 19.5%
阈 = pd.DataFrame(0.0949, index=cl.index, columns=列)
创 = [c for c in 列 if c[:3] in ("300", "301")]
科 = [c for c in 列 if c[:3] == "688"]
阈.loc["2020-08-24":, 创] = 0.1949
阈[科] = 0.1949
up = (r.ge(阈) & 成员).sum(axis=1); dn = (r.le(-阈) & 成员).sum(axis=1)
limit_net = (up - dn) / N
高250 = cl.rolling(250, min_periods=200).max()
低250 = cl.rolling(250, min_periods=200).min()
nh = (cl.ge(高250 * 0.9999) & 成员).sum(axis=1)
nl = (cl.le(低250 * 1.0001) & 成员).sum(axis=1)
nhnl = (nh - nl) / N
w = am.where(成员); w = w.div(w.sum(axis=1), axis=0)
amt_ent = -(w * np.log(w)).sum(axis=1) / np.log(N.replace(0, np.nan))

叶 = pd.DataFrame({"breadth": breadth, "disp": disp, "limit_net": limit_net,
                  "nhnl": nhnl, "xskew": xskew, "amt_ent": amt_ent})
叶 = 叶[N > 500]
叶.index.name = "date"
出 = os.path.expanduser("~/Desktop/中证1000择时/data/archive/截面叶子.csv")
叶.to_csv(出)
print(f"叶子 {叶.index[0].date()}~{叶.index[-1].date()} {len(叶)}天 → {出}")
print(f"拼接日前后(±3天)均值检查:")
边 = 叶.loc["2014-10-20":"2014-11-14"]
print(边.round(4).to_string())
