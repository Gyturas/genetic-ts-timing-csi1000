# -*- coding: utf-8 -*-
"""etf择时 锁定配置(PROTOCOL v1.0)。所有协议数值集中于此,跑后不改。"""
from __future__ import annotations

import os

from csi1000 import paths

ROOT = paths.根                                # 仓库根
CACHE = paths.行情缓存                          # data/cache/
RESULTS = os.path.join(ROOT, "results")
STATE_DIR = os.path.join(RESULTS, "state")     # 挖矿运行时状态(gitignore)

# ---------------- 五大类:挖掘面板(信号层) 与 交易ETF(执行层) ----------------
# panel: 面板品种 -> data_cache 文件tag ; etf_map: ETF -> 其信号来源面板品种列表
# offshore=True 的面板序列整体 +1 个A股交易日错位(收盘时点晚于A股收盘)
CLASSES = {
    "A股宽基": {
        "panel": {"hs300": "idx_sh000300", "zz500": "idx_sh000905", "zz1000": "idx_sh000852",
                  "cyb": "idx_sz399006", "kc50": "idx_sh000688", "hongli": "idx_sh000922"},
        "etf_map": {"510300": ["hs300"], "510500": ["zz500"], "512100": ["zz1000"],
                    "159915": ["cyb"], "588000": ["kc50"], "510880": ["hongli"]},
        "offshore": False,
        "bench": "hs300",           # beta120 基准(面板品种名或 "EW"=面板等权)
    },
    "债券": {
        "panel": {"gz": "idx_sh000012", "qz": "idx_sh000013", "zhuan": "idx_sh000832"},
        "etf_map": {"511010": ["gz"], "511260": ["gz"], "511380": ["zhuan"]},
        "offshore": False,
        "bench": "gz",
    },
    "贵金属": {
        "panel": {"au": "fut_au0", "ag": "fut_ag0"},
        "etf_map": {"518880": ["au"], "161226": ["ag"]},
        "offshore": False,
        "bench": "au",
    },
    "商品": {
        "panel": {"cu": "fut_cu0", "al": "fut_al0", "zn": "fut_zn0", "m": "fut_m0",
                  "ta": "fut_ta0", "ma": "fut_ma0", "rb": "fut_rb0"},
        "etf_map": {"159980": ["cu", "al", "zn"], "159985": ["m"], "159981": ["ta", "ma"]},
        "offshore": False,
        "bench": "EW",
    },
    "跨境": {
        "panel": {"ndx": "idx_ndx", "spx": "idx_spx", "hsi": "idx_hsi", "hstech": "idx_hstech",
                  "oil": "etf_501018"},          # 原油无海外面板,用其LOF自身(有量额)
        "etf_map": {"513100": ["ndx"], "513500": ["spx"], "159920": ["hsi"],
                    "513130": ["hstech"], "501018": ["oil"]},
        "offshore": True,
        "bench": "spx",
    },
}
QDII_ETFS = {"513100", "513500", "159920", "513130", "501018"}

# ---------------- 阶段 ----------------
WARMUP_START = "2013-01-01"      # 元老记账起点(定方向用)
BUILD_START  = "2015-01-01"      # 建库/挖矿/选用/记仓起点
PAPER_START  = "2018-01-01"      # 模拟盘(报表参考段)
LIVE_START   = "2020-01-01"      # 实盘计成绩
END          = "2026-07-16"

# ---------------- 成本 / 仓位 ----------------
COST = 0.0005                    # ETF 双边万5(单边计提于|Δ仓|)
COST_QDII = 0.0008
MAP_WIN = 120                    # π = 组合信号120日滚动百分位
MAP_SHIFT = 0.0                  # p = clip(Φ⁻¹(π)+c, 0, 1), c 默认0
EMA_SMOOTH = 5                   # 先映射后 EMA5 平滑
TREND_GATE_WIN = 200             # 标的(面板品种)200日线下 p=0;闸后置,零仓无视缓冲带
NO_TRADE_BAND = 0.05             # 执行层不交易带(对每品种 p)
MIN_HISTORY = 364                # 品种面板史不足364天 → 闸关(p=0)

# ---------------- 因子评价与库制度 ----------------
Z_WIN = 250                      # 信号标准化: z250(因子值), 方向号入库时冻结
SEL_WIN = 250                    # 选用/体检 trailing 窗(≈12月)
ICIR_MONTHS = 36                 # ICIR: trailing 36 个月度IC
VALID_DAYS = 488                 # 入库考期 = validation 2年日对
TRAIN_MIN_DAYS = 750             # 挖矿 train 段最少样本,不足则本季不挖

ADMIT_IC = 0.03                  # 类均IC(validation)入库线
ADMIT_T = 2.0                    # 月度类均IC 的 NW t 硬门槛
ADMIT_XS_POS = 0.60              # 跨品种 IC>0 占比
ADMIT_RESID_IC = 0.015           # 对现役(相关最高≤6个)回归残差IC
ADMIT_MAX_CORR = 0.90            # 与现役最大相关
ADMIT_PER_Q = 3                  # 每类每季新入库上限
FAMILY_CAP = 0.40                # 行为家族库内占比上限(库≥5才启用;"其他"不设限)
FAMILY_TAG_MIN = 0.30            # 原型|corr|<0.3 → 标"其他"
EVENT_TOP = 0.20                 # 高光日=信号 validation 内前20%
EVENT_TRIM = 0.05                # 事件收益去尾5%

SEATS = 6                        # 每类选用席位
SCORE_W = {"ic": 0.35, "icir": 0.20, "pnl": 0.20, "hit": 0.15, "dscorr": 0.10}
IC_SHRINK_OWN = 0.4              # 收缩IC = 0.4本品种 + 0.6跨品种(类均)
HYSTERESIS_Q = 2                 # 挑战者连续2季胜出才换座

RETIRE_IC = 0.0                  # trailing 12月(250日)带号类均IC<0 → 黄牌
YELLOW_Q = 2                     # 连续2季黄牌 → 退役
RETIRE_HARD_IC = -0.03           # 立即退役线
RETIRE_HARD_T = -1.5
GRACE_DAYS = 250                 # 入库不足12月不体检
YELLOW_PENALTY = 0.5             # 黄牌期选用综合分×0.5

REENLIST_Q = 4                   # 退役≥4季可申请复员
REENLIST_IC = ADMIT_IC * 1.5     # 场外24月(500日)IC 线
REENLIST_WIN = 500
TRANSFER_TOP = 10                # 转会:其他类现役按本类综合分前10为候选
TRANSFER_ENABLED = True          # smoke 单类跑时置 False(否则等兄弟类快照)
TRANSFER_TIMEOUT = 7200

# ---------------- GA ----------------
GA = {
    "population": 200, "generations": 20, "tournament_k": 5,
    "p_crossover": 0.70, "p_mutation": 0.25, "elitism": 8,
    "max_depth": 7, "init_depth_min": 2, "init_depth_max": 5,
    "parsimony": 0.001, "seed": 42,
}
HOF = {"size": 20, "min_train_ic": 0.02, "min_valid_ic": 0.02, "max_corr": 0.70}
FIT_MIN_COVERAGE = 0.55          # 面板有效格覆盖率下限(品种起始日不齐,不能设太高)

TERMINALS = ("open", "high", "low", "close", "volume", "amount", "returns",
             "entropy120", "beta120")

# ---------------- 元老(13,无豁免) ----------------
ARGARCH = {"refit_window": 750, "min_obs": 400, "ret_scale": 100.0}
