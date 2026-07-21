# -*- coding: utf-8 -*-
"""cos IC 打分的正确性单元测试。全量挖矿前必须全绿。

跑: PYTHONPATH=.. python tests/test_cos_fitness.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from ga_alpha import fitness as F

失败 = []
def 断言(名, 条件, 详=""):
    print(f"  {'✓' if 条件 else '✗'} {名}" + ("" if 条件 else f"   {详}"))
    if not 条件: 失败.append(名)


# ① cos_ic 数学:手算一个已知答案
def 手算cos(p, r):
    p, r = np.array(p, float), np.array(r, float)
    return (p*r).sum() / np.sqrt((p*p).sum() * (r*r).sum())

p = [0.5, -0.3, 0.8, -0.1, 0.2]
r = [0.02, -0.01, 0.03, 0.00, -0.02]
# 单列 DataFrame,给足 _每列最少 需要绕过 → 临时调小
old = F._每列最少; F._每列最少 = 3
val, cov = F.cos_ic(pd.DataFrame({"x": p}), pd.DataFrame({"x": r}))
F._每列最少 = old
断言("cos_ic 与手算一致", abs(val - 手算cos(p, r)) < 1e-12, f"{val} vs {手算cos(p,r)}")

# ② 方向性:p 取负,cos 反号
F._每列最少 = 3
v1, _ = F.cos_ic(pd.DataFrame({"x": p}), pd.DataFrame({"x": r}))
v2, _ = F.cos_ic(pd.DataFrame({"x": [-x for x in p]}), pd.DataFrame({"x": r}))
F._每列最少 = old
断言("仓位取负 → cos 反号", abs(v1 + v2) < 1e-12, f"{v1} vs {v2}")

# ③ 完美正相关 cos=1,完美反相关 cos=-1
F._每列最少 = 3
vp,_ = F.cos_ic(pd.DataFrame({"x":[1,2,3,4.]}), pd.DataFrame({"x":[1,2,3,4.]}))
vn,_ = F.cos_ic(pd.DataFrame({"x":[1,2,3,4.]}), pd.DataFrame({"x":[-1,-2,-3,-4.]}))
F._每列最少 = old
断言("同向 cos=+1", abs(vp-1)<1e-12, str(vp))
断言("反向 cos=-1", abs(vn+1)<1e-12, str(vn))

# ④ 到仓位:常数因子 → π=0.5 → ppf=0 → p=0
c = pd.DataFrame({"x": np.ones(300)})
pc = F.到仓位(c)
断言("常数因子 → 仓位≈0(非NaN段)", np.nanmax(np.abs(pc.values)) < 1e-9,
     f"max|p|={np.nanmax(np.abs(pc.values)):.2e}")

# ⑤ 到仓位:纯单调因子过 rank120 会饱和成常数 → 仓位退化为0
#    (这是正确的:价格水平这类纯趋势没有择时信息,rank窗内每天都是最大值)
mono = pd.DataFrame({"x": np.arange(300.0)})
pm = F.到仓位(mono)
断言("单调因子饱和 → 仓位退化≈0", np.nanmax(np.abs(pm.values)) < 1e-9,
     f"max|p|={np.nanmax(np.abs(pm.values)):.2e}")

# ⑤b 有周期的因子(正弦)→ 仓位随因子起伏(有正有负、与因子正相关),且恒在(-1,1)
sin = pd.DataFrame({"x": np.sin(np.arange(600.0) / 20)})
ps = F.到仓位(sin)
尾 = ps["x"].dropna()
断言("周期因子仓位有正有负", (尾 > 0.1).any() and (尾 < -0.1).any(),
     f"范围 {尾.min():.2f}~{尾.max():.2f}")
共 = pd.concat([sin["x"], ps["x"]], axis=1).dropna()
断言("仓位与因子正相关", 共.iloc[:, 0].corr(共.iloc[:, 1]) > 0.2,
     f"corr={共.iloc[:, 0].corr(共.iloc[:, 1]):.2f}")
断言("仓位恒在(-1,1)", np.nanmax(np.abs(ps.values)) < 1.0, f"max|p|={np.nanmax(np.abs(ps.values)):.4f}")

# ⑥ 陷阱验证:先算p再切片 ≠ 先切片再算p;evaluate 走的是前者(正确)
rng = np.random.default_rng(0)
f = pd.DataFrame({"x": rng.standard_normal(600)})
p_full = F.到仓位(f)
valid_mask = np.zeros(600, bool); valid_mask[450:] = True     # 模拟 valid 段在尾部
p_对 = p_full[valid_mask]                                     # evaluate/valid_ic 的做法
p_错 = F.到仓位(f[valid_mask])                                # 先切再滚(会丢历史)
头部错NaN = p_错["x"].iloc[:F.RANK窗+F.映射窗].isna().mean()
断言("先切片再滚 → 头部大量NaN(证明该做法错)", 头部错NaN > 0.9, f"{头部错NaN:.2f}")
断言("先算p再切 → valid段几乎无NaN(evaluate正确)",
     p_对["x"].isna().mean() < 0.05, f"NaN比例={p_对['x'].isna().mean():.2f}")

# ⑦ 覆盖率:rolling 头部损失后,train 段覆盖率仍应 > min_coverage(0.7)
train_mask = np.ones(2000, bool)
ff = pd.DataFrame({"a": rng.standard_normal(2000), "b": rng.standard_normal(2000)})
pp = F.到仓位(ff)
fwd = pd.DataFrame({"a": rng.standard_normal(2000), "b": rng.standard_normal(2000)})
_, cov = F.cos_ic(pp, fwd)
断言("2000天序列覆盖率 > 0.7", cov > 0.7, f"cov={cov:.3f}")

print("\n" + ("全部通过 ✓" if not 失败 else f"✗ {len(失败)} 项失败: {失败}"))
sys.exit(1 if 失败 else 0)
