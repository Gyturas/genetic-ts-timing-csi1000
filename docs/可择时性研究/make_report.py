# -*- coding: utf-8 -*-
"""统一重跑全部资产的五项 AR-GARCH 回归,生成中文(tables.tex)与英文(tables_en.tex)
计量格式表格。所有数字同源。星号: |t|>2.576*** >1.96** >1.645*。"""
from __future__ import annotations

import json, os, warnings
import numpy as np, pandas as pd
from scipy.stats import spearmanr

warnings.filterwarnings("ignore")
from arch.univariate import ARX, GARCH

本 = os.path.dirname(os.path.abspath(__file__))
年 = 244

# (源, 中文名, 英文名, 实测等级)
组表 = {
    "A": [("csv:idx_sh000300", "沪深300", "CSI 300", 2), ("csv:idx_sh000905", "中证500", "CSI 500", 3),
          ("csv:idx_sh000852", "中证1000", "CSI 1000", 4), ("csv:idx_sz399006", "创业板指", "ChiNext", np.nan),
          ("dc:0.399303", "国证2000", "SZSE 2000", np.nan), ("dc:2.932000", "中证2000", "CSI 2000", np.nan),
          ("dc:0.899050", "北证50", "BSE 50", np.nan)],
    "B": [("csv:au9999", "黄金现货", "Gold spot", 2), ("csv:fut_au0", "黄金期货", "Gold fut.", 2),
          ("csv:fut_ag0", "白银", "Silver", 1), ("csv:fut_rb0", "螺纹钢", "Rebar", 1),
          ("csv:fut_cu0", "铜", "Copper", 1), ("csv:fut_m0", "豆粕", "Soy meal", 1),
          ("csv:fut_ta0", "PTA", "PTA", 1), ("csv:fut_ma0", "甲醇", "Methanol", 1)],
    "C": [("csv:idx_ndx", "纳斯达克100", "Nasdaq 100", 2), ("csv:idx_spx", "标普500", "S\\&P 500", np.nan),
          ("stk:1.600519", "贵州茅台", "Moutai", np.nan), ("stk:1.601318", "中国平安", "Ping An", np.nan),
          ("stk:1.600426", "华鲁恒升", "Hualu", np.nan), ("stk:0.002131", "利欧股份", "Leo Grp", np.nan),
          ("stk:0.300308", "中际旭创", "Innolight", np.nan), ("stk:0.002741", "光华科技", "Guanghua", np.nan)],
}


def 读(src: str) -> pd.Series:
    kind, key = src.split(":", 1)
    if kind == "csv":
        d = pd.read_csv(os.path.join(本, "data", key + ".csv"), parse_dates=["date"])
        return d.set_index("date")["close"].dropna()
    fp = os.path.join(本, ("raw_" if kind == "dc" else "rawstk_") + key + ".json")
    k = json.load(open(fp))["data"]["klines"]
    d = pd.DataFrame([x.split(",") for x in k], columns=["date", "close"])
    return pd.Series(d["close"].astype(float).values, index=pd.to_datetime(d["date"]))


def 回归(px: pd.Series) -> dict | None:
    r = (np.log(px).diff().dropna() * 100).loc["2010":]
    r = r[r.abs() < 25]
    if len(r) < 600:
        return None
    m1 = ARX(r, lags=1); m1.volatility = GARCH(1, 1)
    f1 = m1.fit(disp="off")
    sig = f1.conditional_volatility.reindex(r.index)
    z = ((sig - sig.mean()) / sig.std()).shift(1)
    momW = min(244, len(r) // 6)
    mom = r.rolling(momW).sum().shift(1); zmom = (mom - mom.mean()) / mom.std()
    d = pd.concat([r.rename("r"), pd.DataFrame({"zmom": zmom, "z": z, "zr": z * r.shift(1)})], axis=1).dropna()
    m2 = ARX(d["r"], lags=1, x=d[["zmom", "z", "zr"]]); m2.volatility = GARCH(1, 1)
    f2 = m2.fit(disp="off")
    p, t = f2.params, f2.tvalues
    k1 = next(k for k in p.index if "[1]" in k and k not in ("zmom", "z", "zr"))
    ka = next(k for k in p.index if k.lower().startswith("alpha"))
    kb = next(k for k in p.index if k.lower().startswith("beta"))
    s2 = f2.conditional_volatility.reindex(d.index)
    mz = pd.concat([(d["r"] ** 2).rename("y"), (s2 ** 2).rename("x")], axis=1).dropna()
    return dict(N=len(d), momW=momW, 起=str(r.index[0].date()),
                mu=float(p["Const"]) / 100 * 年 * 100, t_mu=float(t["Const"]),
                phi1=float(p[k1]), t1=float(t[k1]),
                phiL=float(p["zmom"]) * 100, tL=float(t["zmom"]),
                phi2=float(p["z"]) * 100, t2=float(t["z"]),
                phi3=float(p["zr"]), t3=float(t["zr"]),
                persist=float(p[ka] + p[kb]),
                MZ=float(np.corrcoef(mz["y"], mz["x"])[0, 1] ** 2),
                VR5=float(r.rolling(5).sum().dropna().var() / (5 * r.var())),
                SR=float(r.mean() / r.std() * np.sqrt(年)),
                vol=float(r.std() * np.sqrt(年)))


def 星(t):
    a = abs(t)
    return "^{***}" if a > 2.576 else ("^{**}" if a > 1.96 else ("^{*}" if a > 1.645 else ""))


结果, 等级, 英名 = {}, {}, {}
for 组, lst in 组表.items():
    for src, 名, en, 级 in lst:
        o = 回归(读(src))
        if o:
            结果[名] = o; 英名[名] = en
            if np.isfinite(级):
                等级[名] = 级
json.dump(结果, open(os.path.join(本, "report_meta.json"), "w"), ensure_ascii=False, indent=1)

文案 = {
    "cn": dict(
        行序=[("$\\hat\\mu$~常数项 (年化\\%)", "mu", "t_mu", ".1f"),
             ("$\\hat\\varphi_1$~($r_{t-1}$)", "phi1", "t1", ".4f"),
             ("$\\hat\\varphi_L$~($\\tilde m_{t-1}$, bp)", "phiL", "tL", ".1f"),
             ("$\\hat\\varphi_2$~($\\tilde\\sigma_{t-1}$, bp)", "phi2", "t2", ".1f"),
             ("$\\hat\\varphi_3$~($\\tilde\\sigma_{t-1}\\!\\cdot\\! r_{t-1}$)", "phi3", "t3", ".4f")],
        统计=[("$\\hat\\alpha+\\hat\\beta$", "persist", ".3f"), ("MZ $R^2$", "MZ", ".3f"),
             ("VR(5)", "VR5", ".3f"), ("买持夏普", "SR", ".2f"), ("年化波动(\\%)", "vol", ".0f")],
        N行="$N$", 起行="样本起点",
        注1="括号内为 Bollerslev--Wooldridge 稳健 $t$ 值;$^{***}$、$^{**}$、$^{*}$ 分别表示 1\\%、5\\%、10\\% 显著水平(双侧)。",
        注2="$\\hat\\varphi_L,\\hat\\varphi_2$ 单位为 bp/日(状态变量每高一个标准差);长动量窗为 $\\min(244,N/6)$ 日。",
        标题=["A股宽基指数:五项均值方程估计(GARCH(1,1) 扰动)", "贵金属与商品期货(主力连续)",
             "海外指数与 A 股个股(前复权)", "模型参数与实测策略效果的 Spearman 秩相关($n=%d$)"],
        秩行=[("phi1", "$\\hat\\varphi_1$~($r_{t-1}$,短动量)"), ("phiL", "$\\hat\\varphi_L$~($\\tilde m_{t-1}$,长动量)"),
             ("phi2", "$\\hat\\varphi_2$~($\\tilde\\sigma_{t-1}$,波动溢价)"),
             ("phi3", "$\\hat\\varphi_3$~($\\tilde\\sigma_{t-1}\\!\\cdot\\! r_{t-1}$,交互)"),
             ("persist", "$\\hat\\alpha+\\hat\\beta$(GARCH持续性)"), ("MZ", "MZ $R^2$(波动可预测)"),
             ("VR5", "VR(5)(方差比)"), ("SR", "买持夏普")],
        秩注="实测效果序数化:中证1000=4,中证500=3,弱/一般=2,商品=1;未实测资产不参与。",
        秩头="参数 & $\\hat\\rho$ & $p$ 值", 标签后缀=""),
    "en": dict(
        行序=[("$\\hat\\mu$~const.\\ (ann.\\ \\%)", "mu", "t_mu", ".1f"),
             ("$\\hat\\varphi_1$~($r_{t-1}$)", "phi1", "t1", ".4f"),
             ("$\\hat\\varphi_L$~($\\tilde m_{t-1}$, bp)", "phiL", "tL", ".1f"),
             ("$\\hat\\varphi_2$~($\\tilde\\sigma_{t-1}$, bp)", "phi2", "t2", ".1f"),
             ("$\\hat\\varphi_3$~($\\tilde\\sigma_{t-1}\\!\\cdot\\! r_{t-1}$)", "phi3", "t3", ".4f")],
        统计=[("$\\hat\\alpha+\\hat\\beta$", "persist", ".3f"), ("MZ $R^2$", "MZ", ".3f"),
             ("VR(5)", "VR5", ".3f"), ("B\\&H Sharpe", "SR", ".2f"), ("Ann.\\ vol (\\%)", "vol", ".0f")],
        N行="$N$", 起行="Sample start",
        注1="Bollerslev--Wooldridge robust $t$-statistics in parentheses; $^{***}$, $^{**}$, $^{*}$ denote significance at the 1\\%, 5\\%, 10\\% levels (two-sided).",
        注2="$\\hat\\varphi_L,\\hat\\varphi_2$ are in bp/day per one-s.d.\\ move of the state variable; the long-momentum window is $\\min(244,N/6)$ days.",
        标题=["Chinese broad equity indices: five-term mean equation with GARCH(1,1) errors",
             "Precious metals and commodity futures (front-month continuous)",
             "Overseas indices and individual A-share stocks (adjusted prices)",
             "Spearman rank correlation between model parameters and realized strategy performance ($n=%d$)"],
        秩行=[("phi1", "$\\hat\\varphi_1$~($r_{t-1}$, short momentum)"), ("phiL", "$\\hat\\varphi_L$~($\\tilde m_{t-1}$, long momentum)"),
             ("phi2", "$\\hat\\varphi_2$~($\\tilde\\sigma_{t-1}$, volatility premium)"),
             ("phi3", "$\\hat\\varphi_3$~($\\tilde\\sigma_{t-1}\\!\\cdot\\! r_{t-1}$, interaction)"),
             ("persist", "$\\hat\\alpha+\\hat\\beta$ (GARCH persistence)"), ("MZ", "MZ $R^2$ (vol.\\ predictability)"),
             ("VR5", "VR(5) (variance ratio)"), ("SR", "B\\&H Sharpe")],
        秩注="Realized-performance ranks: CSI 1000 = 4, CSI 500 = 3, weak/mediocre = 2, commodities = 1; assets without live results are excluded.",
        秩头="Parameter & $\\hat\\rho$ & $p$-value", 标签后缀="en"),
}


def 做表(names, caption, label, L语):
    cols = [n for n in names if n in 结果]
    头 = [(英名[c] if L语["标签后缀"] else c) for c in cols]
    Ls = ["\\begin{table}[htbp]\\centering", f"\\caption{{{caption}}}\\label{{{label}}}",
          "\\footnotesize\\setlength{\\tabcolsep}{4pt}", "\\begin{threeparttable}",
          "\\begin{tabular}{l" + "c" * len(cols) + "}", "\\toprule",
          " & " + " & ".join(头) + " \\\\", "\\midrule"]
    for 名t, k, kt, fmt in L语["行序"]:
        a = [f"${结果[c][k]:{fmt}}{星(结果[c][kt])}$" for c in cols]
        b = [f"$({结果[c][kt]:.2f})$" for c in cols]
        Ls.append(名t + " & " + " & ".join(a) + " \\\\")
        Ls.append(" & " + " & ".join(b) + " \\\\[2pt]")
    Ls.append("\\midrule")
    for 名t, k, fmt in L语["统计"]:
        Ls.append(名t + " & " + " & ".join(f"${结果[c][k]:{fmt}}$" for c in cols) + " \\\\")
    Ls.append(L语["N行"] + " & " + " & ".join(f"${结果[c]['N']}$" for c in cols) + " \\\\")
    Ls.append(L语["起行"] + " & " + " & ".join(f"\\scriptsize {结果[c]['起'][:7]}" for c in cols) + " \\\\")
    Ls += ["\\bottomrule", "\\end{tabular}", "\\begin{tablenotes}\\scriptsize",
           "\\item " + L语["注1"], "\\item " + L语["注2"],
           "\\end{tablenotes}\\end{threeparttable}\\end{table}"]
    return "\n".join(Ls)


for lang, L语 in 文案.items():
    sfx = L语["标签后缀"]
    lab = lambda base: f"tab:{base}{'_en' if sfx else ''}"
    T1 = 做表([x[1] for x in 组表["A"]], L语["标题"][0], lab("ashare"), L语)
    T2 = 做表([x[1] for x in 组表["B"]], L语["标题"][1], lab("comm"), L语)
    T3 = 做表([x[1] for x in 组表["C"]], L语["标题"][2], lab("stk"), L语)
    名单 = [n for n in 等级 if n in 结果]
    T4 = ["\\begin{table}[htbp]\\centering",
          "\\caption{" + L语["标题"][3] % len(名单) + "}\\label{" + lab("rank") + "}",
          "\\small\\begin{threeparttable}\\begin{tabular}{lcc}\\toprule",
          L语["秩头"] + " \\\\ \\midrule"]
    for k, 名t in L语["秩行"]:
        x = [结果[n][k] for n in 名单]; y = [等级[n] for n in 名单]
        rho, pv = spearmanr(x, y)
        s = "$^{***}$" if pv < .01 else ("$^{**}$" if pv < .05 else ("$^{*}$" if pv < .1 else ""))
        T4.append(f"{名t} & ${rho:+.3f}${s} & ${pv:.3f}$ \\\\")
    T4 += ["\\bottomrule\\end{tabular}", "\\begin{tablenotes}\\scriptsize\\item " + L语["秩注"],
           "\\end{tablenotes}\\end{threeparttable}\\end{table}"]
    fn = "tables_en.tex" if sfx else "tables.tex"
    open(os.path.join(本, fn), "w").write(T1 + "\n\n" + T2 + "\n\n" + T3 + "\n\n" + "\n".join(T4) + "\n")
    print(fn, "完成")
print("资产数:", len(结果))
