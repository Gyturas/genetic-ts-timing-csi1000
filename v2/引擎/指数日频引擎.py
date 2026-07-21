# -*- coding: utf-8 -*-
"""指数择时·日频·genetic-cta完全复制×资产定制(指数/方案.md 锁定版)。

双轨结算:分资产满额账(每指数 货基+仓位×(ETF收益−货基)−万5成本)+六ETF等权组合账。

制度照抄 genetic-cta 第三轮定版 config:
  信号 = 因子值.rolling(120).rank(pct)×2−1×方向号 ∈[-1,1];仓位=全库加权截断[-1,1],可做空;
  入库三关:月度IC均值≥0.015 / 残差IC≥0.01 / 相关筛0.9;库无上限;ICIR禁用;
  退役:trailing12月IC<0黄牌,连2季退役,<-0.02立即;宽限12月;复职:冷却4季+场外0.0225+限1次;
  在役<8仓位减半;无趋势闸/无平滑/无不交易带。
本版扩展(用户裁定):两级挖矿——六个指数各一个专属库(单指数面板挖,考自己)+一个类库(整体挖,
考混合月度IC);资产定制——类库因子也逐指数记账,每个指数用"对自己的trailing12月IC"全库加权。
用法: python -m walkforward.宽基日频引擎 [--smoke] [--end YYYY-MM-DD]
"""
from __future__ import annotations

import argparse
import os
import pickle
import time
import warnings

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

warnings.filterwarnings("ignore")

import ga_alpha.expr as gexpr
import ga_alpha.fitness as gfit
import ga_alpha.ops as gops
from ga_alpha.evolve import run_ga

from walkforward import config as C
from walkforward.data import load_all
from walkforward.elders import ARGARCH_NAMES, ArgarchRecorder, trend_elder_values
from walkforward.engine import quarter_list

# ---------------- genetic-cta 照抄参数 ----------------
# v2 阈值:cos IC 尺度,按 v1 存档实测的月度cos分布(均值+0.028 std0.227;
# trailing12m 均值+0.000 std0.060 >0占47%)与 rank 阈值按 σ 对齐标定。★起始值,smoke 后锁★
入库线, 残差线, 相关筛 = 0.01, 0.01, 0.9      # 入库线cos(原rank0.015);残差线仍rank口径不变
TRAILING月 = 12
黄牌线, 连黄退役, 立即退役线 = -0.03, 2, -0.05  # cos(原 0.0/-0.02);trailing均值中位≈0故门槛下移
最少IC月数, 宽限月, 回归自变量上限, 烧机最少 = 18, 12, 40, 8
复职冷却季, 复职线, 复职场外最少月 = 4, 0.03, 12  # 复职线cos(原rank0.0225)
RANK窗 = 120
回看年, 验证年 = 10, 2
GA参数 = {"population": 200, "generations": 20, "tournament_k": 5, "p_crossover": 0.7,
          "p_mutation": 0.25, "max_depth": 6, "init_depth_min": 2, "init_depth_max": 4,
          "elitism": 8, "parsimony": 0.001, "seed": 42}
HOF参数 = {"size": 20, "min_train_ic": 0.01, "min_valid_ic": 0.01, "max_corr": 0.7}  # v2:整段cos尺度,起始值smoke后校
单月最少对数 = 30
成本 = C.COST

# 挖矿产物写到 结果_挖矿/,不覆盖仓库自带的 结果/(回测成绩表与图存档在那里)
输出目录 = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "结果_挖矿")


def 安装GA():
    gops.TERMINALS = C.TERMINALS
    gexpr.TERMINALS = C.TERMINALS

    def 原始次日收益(close, horizon):
        return close.pct_change(fill_method=None).shift(-1)   # 照抄:不减货基

    gfit.forward_returns = 原始次日收益
    # v2 打分走 fitness.cos_ic(算在仓位 p 上),不再经 ts_rank_ic,故不注入它
    # (v1 曾在此注入"相对覆盖率IC"替换 rank IC 打分;cos 版直接在 fitness 内实现)


def 混合月度IC(signal: pd.DataFrame, fwd: pd.DataFrame, 最少对: int | None = None) -> dict:
    """逐月堆叠全资产(信号_t,收益_{t+1})对做一个Spearman——rank IC 记账口径。
    v2 只有残差闸仍用它(残差是去重/增量检查,留 rank 当稳健锚);记账/入库改用 混合月度cosIC。
    最少对数按列数自适应:单列面板一个月只有约20个交易日,30的门槛会让单指数账本永远为空。"""
    if 最少对 is None:
        最少对 = 单月最少对数 if signal.shape[1] >= 2 else 15
    df = pd.concat([signal.stack().rename("s"), fwd.stack().rename("r")], axis=1).dropna()
    out = {}
    for m, g in df.groupby(df.index.get_level_values(0).to_period("M")):
        if len(g) >= 最少对:
            ic = spearmanr(g["s"], g["r"])[0]
            if np.isfinite(ic):
                out[str(m)] = float(ic)
    return out


def 混合月度cosIC(signal: pd.DataFrame, fwd: pd.DataFrame, 最少对: int | None = None) -> dict:
    """v2 记账/入库口径:先把信号(已 rank₁₂₀)过第二层映射成仓位 p,再逐月算 cos(p, 收益)。
    cos = Σp·r / sqrt(Σp²·Σr²),分子就是该因子当月单独交易的毛盈亏,方向×强度都算。
    信号已含 sign(录取时乘过),故 cos 带方向。最少对数口径与 混合月度IC 一致。"""
    if 最少对 is None:
        最少对 = 单月最少对数 if signal.shape[1] >= 2 else 15
    p = gfit.信号到仓位(signal)
    df = pd.concat([p.stack().rename("p"), fwd.stack().rename("r")], axis=1).dropna()
    out = {}
    for m, g in df.groupby(df.index.get_level_values(0).to_period("M")):
        if len(g) >= 最少对:
            den = np.sqrt((g["p"] ** 2).sum() * (g["r"] ** 2).sum())
            if den > 0:
                out[str(m)] = float((g["p"] * g["r"]).sum() / den)
    return out


def 尾均值(hist: dict, n: int) -> float:
    vals = [hist[k] for k in sorted(hist)[-n:]]
    return float(np.mean(vals)) if vals else np.nan


class 宽基引擎:
    def __init__(self, 数据: dict, log=print):
        self.log = lambda *a: log("[指数日频]", *a)
        类 = 数据["classes"]["A股宽基"]
        self.面板 = 类["panels"]
        self.指数表 = list(self.面板["close"].columns)          # 6个指数
        self.映射 = 类["spec"]["etf_map"]                       # ETF -> [指数]
        self.rf = 数据["rf"]
        self.etf收益 = 数据["etf_ret"][list(self.映射)]
        self.日历 = self.面板["close"].index
        self.次日收益 = self.面板["returns"].shift(-1)
        os.makedirs(输出目录, exist_ok=True)
        self.元老记录器 = ArgarchRecorder(self.面板["returns"], os.path.join(输出目录, "argarch"))
        self.季度表 = quarter_list(C.WARMUP_START, C.END)
        self.状态 = self._载入()
        self.组合切片: list[pd.DataFrame] = []      # 逐季逐指数的因子信号+权重存档
        self.信号缓存: dict[int, pd.DataFrame] = {}
        for 编号, m in self.状态["members"].items():
            self.信号缓存[编号] = self._算信号(m)

    # ---------- 状态 ----------
    def _载入(self):
        fp = os.path.join(输出目录, "state.pkl")
        if os.path.exists(fp):
            with open(fp, "rb") as f:
                st = pickle.load(f)
            assert st["done_q"] == [str(q) for q in self.季度表[:len(st["done_q"])]], "存档不连续"
            self.元老记录器.load()
            self.log(f"续跑 {len(st['done_q'])} 季")
            return st
        return {"members": {}, "编号": 0, "done_q": [], "daily": [], "资产仓位": [],
                "分资产": [], "events": [], "元老已定向": False, "前仓资产": {}}

    def _存(self):
        fp = os.path.join(输出目录, "state.pkl")
        with open(fp + ".tmp", "wb") as f:
            pickle.dump(self.状态, f)
        os.replace(fp + ".tmp", fp)
        self.元老记录器.save()

    def _算信号(self, m: dict) -> pd.DataFrame:
        """全史因果信号:值→rank120→[-1,1]×方向号。专属库因子只有本指数一列。"""
        if m["kind"] == "argarch":
            v = self.元老记录器.values(m["name"]).reindex(self.日历)
        elif m["kind"] == "elder":
            v = trend_elder_values(self.面板)[m["name"]]
        else:
            面板 = self.面板 if m["专属"] is None else \
                {k: p[[m["专属"]]] for k, p in self.面板.items()}
            v = gexpr.evaluate(m["node"], 面板).replace([np.inf, -np.inf], np.nan)
        return (v.rolling(RANK窗).rank(pct=True) * 2 - 1) * m["sign"]

    def _在役(self):
        return [i for i, m in self.状态["members"].items() if m["status"] == "在役"]

    def _可用(self, 指数: str):
        return [i for i in self._在役()
                if self.状态["members"][i]["专属"] in (None, 指数)]

    # ---------- 记账 ----------
    def _补账(self, 编号: int, cutoff: pd.Timestamp):
        m = self.状态["members"][编号]
        起 = cutoff - pd.DateOffset(months=16)
        S = self.信号缓存[编号].loc[起:cutoff]
        F = self.次日收益.loc[起:cutoff]
        if m["专属"] is not None:
            F = F[[m["专属"]]]
            S = S[[m["专属"]]] if m["专属"] in S.columns else S
        for 月, ic in 混合月度cosIC(S, F).items():
            m["总账"].setdefault(月, ic)
        for 指数 in (self.指数表 if m["专属"] is None else [m["专属"]]):
            if 指数 not in S.columns:
                continue
            for 月, ic in 混合月度cosIC(S[[指数]], F[[指数]] if 指数 in F.columns
                                    else self.次日收益[[指数]].loc[起:cutoff]).items():
                m["分账"].setdefault(指数, {}).setdefault(月, ic)

    # ---------- 主循环 ----------
    def run(self):
        建库起 = pd.Timestamp(C.BUILD_START)
        for qi, q in enumerate(self.季度表):
            if str(q) in self.状态["done_q"]:
                continue
            季日 = self.日历[(self.日历 >= q.start_time) & (self.日历 <= q.end_time)]
            前 = self.日历[self.日历 < q.start_time]
            if len(季日) == 0 or len(前) == 0:
                self.状态["done_q"].append(str(q)); continue
            cutoff = 前[-1]
            t0 = time.time()
            self.元老记录器.extend(cutoff, 季日)
            if not self.状态["members"]:
                self._元老入册(q)
            for 编号, m in self.状态["members"].items():   # AR-GARCH值面板逐季增长,缓存必须刷新
                if m["kind"] == "argarch":
                    self.信号缓存[编号] = self._算信号(m)
            for 编号 in list(self.状态["members"]):
                self._补账(编号, cutoff)
            if q.start_time >= 建库起:
                self._季度步(q, qi, cutoff)
            self._仓位与结算(q, cutoff, 季日)
            self.状态["done_q"].append(str(q))
            self._存()
            在役 = len(self._在役())
            self.log(f"{q} 完成({time.time()-t0:.0f}s) 在役{在役}")
        self._导出()

    def _元老入册(self, q):
        趋势 = trend_elder_values(self.面板)
        for 名, kind in [(n, "elder") for n in 趋势] + [(n, "argarch") for n in ARGARCH_NAMES]:
            编号 = self.状态["编号"]; self.状态["编号"] += 1
            self.状态["members"][编号] = {
                "编号": 编号, "name": 名, "expr_str": 名, "node": None, "kind": kind,
                "sign": 1.0, "专属": None, "elder": True, "admit_q": str(q),
                "status": "在役", "yellow": 0, "retire_q": None, "reenlist_used": False,
                "总账": {}, "分账": {}}
            self.信号缓存[编号] = self._算信号(self.状态["members"][编号])
        self.log("元老13人入册(预热记账)")

    def _季度步(self, q, qi, cutoff):
        if not self.状态["元老已定向"]:
            for 编号 in self._在役():
                m = self.状态["members"][编号]
                均 = np.mean(list(m["总账"].values())) if m["总账"] else 0.0
                if 均 < 0:
                    m["sign"] = -1.0
                    m["总账"] = {k: -v for k, v in m["总账"].items()}
                    m["分账"] = {i: {k: -v for k, v in h.items()} for i, h in m["分账"].items()}
                    self.信号缓存[编号] = -self.信号缓存[编号]
            self.状态["元老已定向"] = True
            self.log("元老按预热IC定向完毕")

        self._体检(q)
        self._复职(q, cutoff)
        起挖 = cutoff - pd.DateOffset(years=回看年)
        验证起 = cutoff - pd.DateOffset(years=验证年)
        # 类库整体挖 + 六个专属库分别挖
        任务 = [(None, self.面板)] + [
            (i, {k: p[[i]] for k, p in self.面板.items()}) for i in self.指数表]
        for 专属, 面板 in 任务:
            self._挖与录取(q, qi, cutoff, 起挖, 验证起, 专属, 面板)

    def _体检(self, q):
        for 编号 in self._在役():
            m = self.状态["members"][编号]
            if len(m["总账"]) < 宽限月:
                continue
            均 = 尾均值(m["总账"], TRAILING月)
            if np.isfinite(均) and 均 < 立即退役线:
                m.update(status="退役", retire_q=str(q), yellow=0)
                self.状态["events"].append((str(q), "立即退役", m["name"][:60], f"IC={均:+.3f}"))
            elif (not np.isfinite(均)) or 均 < 黄牌线:
                m["yellow"] += 1
                if m["yellow"] >= 连黄退役:
                    m.update(status="退役", retire_q=str(q), yellow=0)
                    self.状态["events"].append((str(q), "黄牌退役", m["name"][:60], f"IC={均:+.3f}"))
            else:
                m["yellow"] = 0

    def _复职(self, q, cutoff):
        for 编号, m in self.状态["members"].items():
            if m["status"] != "退役" or m["reenlist_used"] or m["retire_q"] is None:
                continue
            if (pd.Period(str(q)) - pd.Period(m["retire_q"])).n < 复职冷却季:
                continue
            退月 = pd.Period(m["retire_q"], freq="Q").asfreq("M", "start")
            场外 = [v for k, v in sorted(m["总账"].items()) if pd.Period(k, freq="M") >= 退月]
            if len(场外) < 复职场外最少月:
                continue
            均 = float(np.mean(场外[-24:]))
            if not np.isfinite(均) or 均 < 复职线:
                continue
            范围 = self.指数表 if m["专属"] is None else [m["专属"]]
            if not self._残差过关(编号, cutoff, 范围[0] if m["专属"] else None):
                continue
            m.update(status="在役", reenlist_used=True, yellow=0, retire_q=None)
            self.状态["events"].append((str(q), "复职", m["name"][:60], f"场外IC={均:+.3f}"))

    def _残差过关(self, 候选编号, cutoff, 专属) -> bool:
        """相关筛0.9 + 对现役(权重前40)回归残差月度IC≥0.01。候选编号可为dict(未入册)。"""
        验证起 = cutoff - pd.DateOffset(years=验证年)
        if isinstance(候选编号, dict):
            候选S = 候选编号["S"]
        else:
            候选S = self.信号缓存[候选编号]
        列 = [专属] if 专属 else self.指数表
        候选S = 候选S[[c for c in 列 if c in 候选S.columns]].loc[验证起:cutoff]
        现役 = self._可用(专属) if 专属 else self._在役()
        if not 现役:
            return True
        现役 = 现役[:回归自变量上限]
        y = 候选S.stack().rename("y")
        xs = pd.concat([self.信号缓存[i][[c for c in 列 if c in self.信号缓存[i].columns]]
                        .loc[验证起:cutoff].stack().rename(f"x{i}") for i in 现役], axis=1)
        j = pd.concat([y, xs], axis=1).dropna()
        if len(j) < 200:
            return False
        if j.corr(method="spearman")["y"].drop("y").abs().max() > 相关筛:
            return False
        X = np.column_stack([np.ones(len(j)), j.iloc[:, 1:].to_numpy()])
        beta, *_ = np.linalg.lstsq(X, j["y"].to_numpy(), rcond=None)
        残差 = pd.Series(j["y"].to_numpy() - X @ beta, index=j.index).unstack()
        F = self.次日收益[列].loc[验证起:cutoff]
        ics = 混合月度IC(残差, F)
        return bool(ics) and float(np.mean(list(ics.values()))) >= 残差线

    def _挖与录取(self, q, qi, cutoff, 起挖, 验证起, 专属, 面板):
        切片 = {k: v.loc[起挖:cutoff] for k, v in 面板.items()}
        cfg = {"ga": dict(GA参数, seed=GA参数["seed"] * 1_000_000
                          + int(f"{q.year}{q.quarter}") * 10
                          + (0 if 专属 is None else 1 + self.指数表.index(专属))),
               "hall_of_fame": dict(HOF参数),
               "fitness": {"horizon": 1, "min_coverage": 0.7},
               "split": {"train_end": str(验证起.date()), "valid_end": str(cutoff.date())}}
        hof = run_ga(切片, cfg, log=lambda *a: None)
        已有 = {m["expr_str"] for m in self.状态["members"].values()}
        列 = [专属] if 专属 else self.指数表
        统计 = {"考过": 0, "残差拒": 0, "录取": 0}
        for e in sorted(hof.entries, key=lambda x: -x["fitness"]):
            if e["expr"] in 已有:
                continue
            候选 = {"name": e["expr"][:60], "expr_str": e["expr"], "node": e["_tree"],
                    "kind": "ga", "sign": float(np.sign(e["train_ic"]) or 1.0), "专属": 专属,
                    "elder": False, "admit_q": str(q), "status": "在役", "yellow": 0,
                    "retire_q": None, "reenlist_used": False, "总账": {}, "分账": {}}
            v = gexpr.evaluate(e["_tree"], 面板 if 专属 is None else
                               {k: p[[专属]] for k, p in self.面板.items()}) \
                .replace([np.inf, -np.inf], np.nan)
            S = (v.rolling(RANK窗).rank(pct=True) * 2 - 1) * 候选["sign"]
            ics = 混合月度cosIC(S.loc[验证起:cutoff], self.次日收益[列].loc[验证起:cutoff])
            if len(ics) < 最少IC月数:
                continue
            均 = float(np.mean(list(ics.values())))
            if not np.isfinite(均) or 均 < 入库线:
                continue
            统计["考过"] += 1
            if not self._残差过关({"S": S}, cutoff, 专属):
                统计["残差拒"] += 1
                continue
            统计["录取"] += 1
            编号 = self.状态["编号"]; self.状态["编号"] += 1
            候选["编号"] = 编号
            候选["总账"] = dict(ics)
            self.状态["members"][编号] = 候选
            self.信号缓存[编号] = S
            self._补账(编号, cutoff)
            已有.add(e["expr"])
            库名 = "类库" if 专属 is None else f"专属{专属}"
            self.状态["events"].append((str(q), "入库", e["expr"][:80],
                                        f"{库名} IC={均:+.3f} 方向={候选['sign']:+.0f}"))
        self.log(f"  {'类库' if 专属 is None else '专属'+专属}: HOF={len(hof.entries)} "
                 f"考过={统计['考过']} 残差拒={统计['残差拒']} 录取={统计['录取']}")

    # ---------- 仓位与结算 ----------
    def _仓位与结算(self, q, cutoff, 季日):
        self._结算(季日, self._建指数仓(q, 季日))

    def _建指数仓(self, q, 季日):
        """把在役因子的逐日信号按权重合成为各指数仓位。权重方案在这里,结算在 _结算。"""
        指数仓 = pd.DataFrame(0.0, index=季日, columns=self.指数表)
        if not self.状态["元老已定向"]:
            return 指数仓
        for 指数 in self.指数表:
            可用 = self._可用(指数)
            权重 = {}
            for i in 可用:
                m = self.状态["members"][i]
                w = 尾均值(m["分账"].get(指数, {}), TRAILING月)
                w = max(w, 0.0) if np.isfinite(w) else 0.0
                权重[i] = w * (0.5 if m["yellow"] > 0 else 1.0)
            self._存组合切片(q, 指数, 季日, 可用, 权重)
            总 = sum(权重.values())
            if 总 <= 0:
                continue
            合成 = pd.Series(0.0, index=季日)
            for i, w in 权重.items():
                if w == 0:
                    continue
                s = self.信号缓存[i]
                if 指数 not in s.columns:
                    continue
                合成 = 合成.add(s[指数].reindex(季日).fillna(0.0) * (w / 总), fill_value=0.0)
            if len(可用) < 烧机最少:
                合成 *= 0.5
            指数仓[指数] = 合成.clip(-1, 1)
        return 指数仓

    def _结算(self, 季日, 指数仓):
        r_etf = self.etf收益.reindex(季日)
        rf_q = self.rf.reindex(季日)
        前暴露 = self.状态.get("前暴露", {})
        前仓资产 = self.状态.get("前仓资产", {})
        for d in 季日:
            上市 = [e for e in self.映射 if pd.notna(r_etf.at[d, e])]
            房间 = 1.0 / len(上市) if 上市 else 0.0
            暴露, 当日收益, 成本额 = {}, float(rf_q.at[d]), 0.0
            for e in self.映射:
                p = float(np.mean([指数仓.at[d, i] for i in self.映射[e]])) if e in 上市 else 0.0
                暴露[e] = p * (房间 if e in 上市 else 0.0)
            for e in self.映射:
                前 = 前暴露.get(e, 0.0)
                r = r_etf.at[d, e]
                当日收益 += 前 * ((0.0 if pd.isna(r) else float(r)) - float(rf_q.at[d]))
                成本额 += abs(暴露[e] - 前) * 成本
            当日收益 -= 成本额
            基准 = float(np.nanmean([r_etf.at[d, e] for e in 上市])) if 上市 else float(rf_q.at[d])
            self.状态["daily"].append((d, 当日收益, sum(abs(v) for v in 暴露.values()), 基准, 成本额))
            self.状态["资产仓位"].append((d, dict(暴露)))
            # 分资产满额账:每ETF独立记账(货基+仓位×超额−成本),空仓吃货基
            资产行 = {}
            for e in self.映射:
                有价 = pd.notna(r_etf.at[d, e])
                仓 = float(np.mean([指数仓.at[d, i] for i in self.映射[e]])) if 有价 else 0.0
                前仓 = 前仓资产.get(e, 0.0)
                r = float(r_etf.at[d, e]) if 有价 else 0.0
                收 = float(rf_q.at[d]) + 前仓 * (r - float(rf_q.at[d])) - abs(仓 - 前仓) * 成本
                资产行[e + "_择时收益"] = 收
                资产行[e + "_仓位"] = 仓
                前仓资产[e] = 仓
            self.状态["分资产"].append((d, 资产行))
            前暴露 = 暴露
        self.状态["前暴露"] = 前暴露
        self.状态["前仓资产"] = 前仓资产

    def _存组合切片(self, q, 指数, 季日, 可用, 权重):
        """存下"这一季、这个指数、每个在役因子的逐日信号与正典权重"。

        有了它,换任何加权方案(等权/IC加权/风险平价…)都只是读表重算,秒级出结果,
        不必重挖一遍因子——因子库本身跟权重无关。
        """
        # 每个"可用"因子都要存,信号取不到就存 NaN —— 不能整行跳过。
        # 引擎合成时分母是 sum(所有可用因子的权重),而信号缺失的因子分子上按 0 计
        # (fillna(0.0)),等于让它照占权重、稀释组合。漏存这些行,重放的分母就偏小、
        # 信号偏大。2018Q4 起 hs300/cyb 对不上 5~6%,就是这么来的。
        for i in 可用:
            s = self.信号缓存[i]
            v = (s[指数].reindex(季日) if 指数 in s.columns
                 else pd.Series(np.nan, index=季日))
            self.组合切片.append(pd.DataFrame({
                "date": 季日, "指数": 指数, "因子": i,
                "信号": v.values, "权重": 权重.get(i, 0.0),
                "黄牌": int(self.状态["members"][i]["yellow"] > 0),
                "烧机减半": int(len(可用) < 烧机最少)}))

    def _导出组合切片(self):
        if not self.组合切片:
            return
        fp = os.path.join(输出目录, "因子逐日信号.csv.gz")
        块 = list(self.组合切片)
        if os.path.exists(fp):        # 续跑:并回之前季度已写的部分
            块.insert(0, pd.read_csv(fp, parse_dates=["date"]))
        df = pd.concat(块, ignore_index=True)
        df = df.drop_duplicates(subset=["date", "指数", "因子"], keep="last") \
               .sort_values(["date", "指数", "因子"])
        df.to_csv(fp, index=False, compression="gzip")   # 全精度:截断会让重算对不上
        self.log(f"因子逐日信号存档: {len(df):,} 行 -> {os.path.basename(fp)}")

    def _导出(self):
        self._导出组合切片()
        df = pd.DataFrame(self.状态["daily"],
                          columns=["date", "择时收益", "总暴露", "持有基准收益", "换手成本"]) \
            .set_index("date")
        df = df[~df.index.duplicated(keep="last")].sort_index()
        df.to_csv(os.path.join(输出目录, "组合逐日.csv"))
        ex = pd.DataFrame({d: v for d, v in self.状态["资产仓位"]}).T.sort_index()
        ex[~ex.index.duplicated(keep="last")].to_csv(os.path.join(输出目录, "资产暴露.csv"))
        fz = pd.DataFrame({d: v for d, v in self.状态["分资产"]}).T.sort_index()
        fz[~fz.index.duplicated(keep="last")].to_csv(os.path.join(输出目录, "分资产逐日.csv"))
        pd.DataFrame(self.状态["events"], columns=["季度", "事件", "名称", "明细"]) \
            .to_csv(os.path.join(输出目录, "事件.csv"), index=False)
        名册 = [{k: m.get(k) for k in ("编号", "name", "kind", "sign", "专属", "elder",
                                      "admit_q", "status", "retire_q", "yellow")}
               for m in self.状态["members"].values()]
        pd.DataFrame(名册).to_csv(os.path.join(输出目录, "名册.csv"), index=False)
        self.log("导出:", 输出目录)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--end", default=None)
    args = ap.parse_args()
    if args.smoke:
        GA参数.update(population=40, generations=5)
    if args.end:
        C.END = args.end
    安装GA()
    宽基引擎(load_all()).run()


if __name__ == "__main__":
    main()
