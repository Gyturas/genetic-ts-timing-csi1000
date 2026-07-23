# -*- coding: utf-8 -*-
"""每日一键面板:更新行情 → 生成定稿信号 → 出明日仓位 → 自包含 HTML 到桌面。

用法: PYTHONPATH=src python -m csi1000.engine.daily_dashboard
产出: ~/Desktop/中证1000_每日面板.html(自动在浏览器打开)
"""
from __future__ import annotations

import base64
import io
import os
import datetime as dt

import numpy as np
import pandas as pd
from scipy.stats import norm

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = ["Arial Unicode MS", "PingFang SC"]
plt.rcParams["axes.unicode_minus"] = False

from csi1000 import paths
import csi1000.engine.update_market as um
import csi1000.engine.final_signal as fs
import csi1000.engine.strategy as st

桌面 = os.path.expanduser("~/Desktop")


def _png(fig) -> str:
    b = io.BytesIO(); fig.savefig(b, format="png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    return "data:image/png;base64," + base64.b64encode(b.getvalue()).decode()


def _指标(ret, rf, 起):
    r = ret.dropna().loc[起:]
    if len(r) < 20:
        return None
    nav = (1 + r).cumprod(); 年 = len(r) / 244
    ex = r - rf.reindex(r.index).fillna(0)
    dd = (nav / nav.cummax() - 1).min()
    ann = nav.iloc[-1] ** (1 / 年) - 1
    return dict(年化=ann, 夏普=ex.mean() / ex.std() * np.sqrt(244) if ex.std() > 0 else 0,
                回撤=dd, 卡玛=ann / abs(dd) if dd < 0 else np.nan, 累计=nav.iloc[-1] - 1)


def main():
    print("① 更新行情 …"); 末日 = um.main()
    print("② 生成定稿信号(约2分钟)…"); fs.main()

    # 明日仓位
    π = pd.read_csv(os.path.join(paths.存档, "映射分位.csv"),
                    parse_dates=["date"]).set_index("date")
    comb = pd.read_csv(os.path.join(paths.存档, "分资产逐日.csv"),
                       index_col=0, parse_dates=True)["512100_仓位"]
    仓 = pd.Series(np.tanh(norm.ppf(π["π_增强"])), index=π.index).where(π["π_增强"].notna(), 0.0)
    今 = 仓.index[-1]; 明仓 = float(仓.iloc[-1])
    πv = float(π["π_增强"].iloc[-1])
    combs = comb.dropna()
    combv = float(combs.iloc[-1]); comb日 = combs.index[-1]   # comb 列可能比 π 晚一日更新

    # 历史回测(拿净值、仓位序列)
    收, p, r, rf = st.跑一遍(True)
    收 = 收.dropna()
    nav = (1 + 收.loc["2018":]).cumprod()
    持nav = (1 + r.reindex(收.index).fillna(0).loc["2018":]).cumprod()
    今年 = str(今.year)

    全 = _指标(收, rf, "2018-01-01"); 年内 = _指标(收, rf, f"{今年}-01-01")
    持全 = _指标(r.reindex(收.index).fillna(0), rf, "2018-01-01")
    当前回撤 = float(nav.iloc[-1] / nav.cummax().iloc[-1] - 1)

    # 图1:净值
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(nav, lw=2, color="#c0392b", label=f"择时策略 ({nav.iloc[-1]:.1f}×)")
    ax.plot(持nav, lw=1.3, ls="--", color="#95a5a6", label=f"持有512100 ({持nav.iloc[-1]:.1f}×)")
    ax.set_yscale("log"); ax.legend(fontsize=10); ax.grid(alpha=0.3)
    ax.set_title("净值(对数轴,2018起)", fontsize=12)
    净值图 = _png(fig)

    # 图2:近60日仓位
    fig, ax = plt.subplots(figsize=(9, 2.6))
    近 = 仓.tail(60)
    ax.fill_between(近.index, 近.values * 100, 0, color="#2c6fbb", alpha=0.5, step="mid")
    ax.axhline(0, color="k", lw=0.5); ax.grid(alpha=0.3)
    ax.set_title("近60交易日仓位 (%)", fontsize=12)
    仓位图 = _png(fig)

    def 卡(标, 指, 色="#1a1a1a"):
        if 指 is None:
            return ""
        return f"""<div class=card><div class=t>{标}</div>
        <div class=g><b style="color:{色}">{指['年化']*100:.1f}%</b><span>年化</span></div>
        <div class=g><b>{指['夏普']:.2f}</b><span>夏普</span></div>
        <div class=g><b>{指['回撤']*100:.1f}%</b><span>回撤</span></div>
        <div class=g><b>{指['卡玛']:.2f}</b><span>卡玛</span></div>
        <div class=g><b>{指['累计']*100:.0f}%</b><span>累计</span></div></div>"""

    方向 = "做多" if 明仓 > 0.02 else ("做空" if 明仓 < -0.02 else "空仓")
    色 = "#c0392b" if 明仓 > 0.02 else ("#27ae60" if 明仓 < -0.02 else "#7f8c8d")

    # ---- 实盘跟踪:仓位确定性可复现,按模型执行时 模型仓位=实盘仓位 ----
    # 时序:仓位 p[T](T日收盘算,T+1生效)→ 贡献到 结算收益 收[T+1]
    r_etf = r.reindex(收.index)                         # 512100 当日收益
    本周一 = (今 - pd.Timedelta(days=今.weekday())).normalize()

    # (A) 本周实盘战绩:本周各交易日,策略实吃日收益 vs 持有
    本周 = 收.loc[本周一:]
    本周持 = r_etf.reindex(本周.index).fillna(0)
    战绩行 = ""
    策累, 持累 = 1.0, 1.0
    for d in 本周.index:
        策累 *= (1 + 收[d]); 持累 *= (1 + 本周持[d])
        c = "#c0392b" if 收[d] >= 0 else "#27ae60"
        战绩行 += (f"<tr><td>{d.date()}</td>"
                 f"<td style='text-align:right;color:{c}'>{收[d]*100:+.2f}%</td>"
                 f"<td style='text-align:right'>{本周持[d]*100:+.2f}%</td>"
                 f"<td style='text-align:right;font-weight:600'>{(策累-1)*100:+.2f}%</td></tr>")
    周策收 = (策累 - 1) * 100; 周持收 = (持累 - 1) * 100

    # (B) 滚动明细:信号日仓位 → 次日ETF收益 + 次日策略收益(最新信号日次日未到,留空)
    近信号 = 仓.tail(9)                                  # 含今天(最新信号,收益空)
    滚动行 = ""
    交易日 = list(收.index)
    for T, posT in 近信号.items():
        # posT 在 T 的下一交易日生效,吃那天的收益
        次日 = 交易日[交易日.index(T) + 1] if T in 交易日 and 交易日.index(T) + 1 < len(交易日) else None
        pc = "#c0392b" if posT > 0.02 else ("#27ae60" if posT < -0.02 else "#7f8c8d")
        if 次日 is not None:
            er, sr = float(r_etf.get(次日, np.nan)), float(收.get(次日, np.nan))
            sc = "#c0392b" if sr >= 0 else "#27ae60"
            右 = (f"<td style='text-align:right'>{次日.date()}</td>"
                 f"<td style='text-align:right'>{er*100:+.2f}%</td>"
                 f"<td style='text-align:right;color:{sc};font-weight:600'>{sr*100:+.2f}%</td>")
        else:
            右 = ("<td style='text-align:right;color:#bbb'>次日</td>"
                 "<td style='text-align:right;color:#bbb'>—</td>"
                 "<td style='text-align:right;color:#bbb'>待结算</td>")
        滚动行 += (f"<tr><td>{T.date()}</td>"
                 f"<td style='text-align:right;color:{pc};font-weight:600'>{posT*100:+.1f}%</td>{右}</tr>")

    html = f"""<!doctype html><html><head><meta charset=utf-8>
<title>中证1000 每日面板</title><style>
*{{margin:0;padding:0;box-sizing:border-box}}
body{{font-family:-apple-system,"PingFang SC",sans-serif;background:#f4f5f7;color:#1a1a1a;padding:24px;max-width:1000px;margin:auto}}
h1{{font-size:20px;margin-bottom:4px}} .sub{{color:#888;font-size:13px;margin-bottom:20px}}
.hero{{background:#fff;border-radius:14px;padding:26px;text-align:center;box-shadow:0 1px 4px #0001;margin-bottom:18px}}
.hero .pos{{font-size:56px;font-weight:700;color:{色};line-height:1}}
.hero .dir{{font-size:18px;color:{色};margin:6px 0}}
.hero .meta{{color:#888;font-size:13px}}
.cards{{display:flex;gap:14px;margin-bottom:18px;flex-wrap:wrap}}
.card{{background:#fff;border-radius:12px;padding:16px 18px;flex:1;min-width:220px;box-shadow:0 1px 4px #0001}}
.card .t{{font-size:13px;color:#888;margin-bottom:10px;font-weight:600}}
.g{{display:inline-block;width:19%;text-align:center;vertical-align:top}}
.g b{{display:block;font-size:16px}} .g span{{font-size:11px;color:#aaa}}
img{{width:100%;border-radius:12px;background:#fff;box-shadow:0 1px 4px #0001;margin-bottom:14px}}
.row{{display:flex;gap:14px;flex-wrap:wrap}} .row>*{{flex:1;min-width:280px}}
table{{width:100%;background:#fff;border-radius:12px;border-collapse:collapse;overflow:hidden;box-shadow:0 1px 4px #0001;font-size:14px}}
td{{padding:7px 14px;border-bottom:1px solid #f0f0f0}}
.lbl{{font-size:13px;font-weight:600;color:#555;margin-bottom:6px}}
.note{{color:#aaa;font-size:12px;margin-top:16px;line-height:1.6}}
.row{{margin-bottom:16px;align-items:flex-start}}
</style></head><body>
<h1>中证1000 时序择时 · 每日面板</h1>
<div class=sub>v4 定稿(双40窗+同质参照系+cos加权+剔元老) · 信号日 {今.date()} · 生成 {dt.datetime.now():%Y-%m-%d %H:%M}</div>

<div class=hero>
  <div style="font-size:14px;color:#888;margin-bottom:8px">下一交易日开盘 · 512100 推荐仓位</div>
  <div class=pos>{明仓*100:+.1f}%</div>
  <div class=dir>{方向}</div>
  <div class=meta>组合信号 comb = {combv:+.3f} &nbsp;·&nbsp; 40日分位 π = {πv:.3f} &nbsp;·&nbsp; 当前回撤 {当前回撤*100:.1f}%</div>
</div>

<div class=cards>{卡("策略 · 2018至今", 全, 色)}{卡(f"策略 · {今年}年内", 年内)}{卡("持有512100 · 2018至今", 持全, "#7f8c8d")}</div>

<div class=row>
  <div style="flex:1.15">
    <div class=lbl>本周实盘战绩(自 {本周一.date()} 周一起 · 按模型仓位每日实吃)</div>
    <table>
      <tr style="color:#888;font-size:12px"><td>日期</td><td style="text-align:right">策略当日</td><td style="text-align:right">持有当日</td><td style="text-align:right">本周累计</td></tr>
      {战绩行}
      <tr style="font-weight:700;background:#fafafa"><td>本周合计</td><td style="text-align:right;color:{'#c0392b' if 周策收>=0 else '#27ae60'}">{周策收:+.2f}%</td><td style="text-align:right">{周持收:+.2f}%</td><td style="text-align:right">超额 {周策收-周持收:+.2f}%</td></tr>
    </table>
  </div>
  <div style="flex:1">
    <div class=lbl>信号 → 次日收益(前一日仓位,次日吃到的收益)</div>
    <table>
      <tr style="color:#888;font-size:12px"><td>信号日</td><td style="text-align:right">仓位</td><td style="text-align:right">次日</td><td style="text-align:right">标的</td><td style="text-align:right">策略</td></tr>
      {滚动行}
    </table>
  </div>
</div>

<img src="{净值图}">
<img src="{仓位图}">

<div class=note>
口径:回测按信号日收盘成交,实盘为次日开盘(半天实施偏差约 −0.7pp);双边万5成本。
选择偏差已知,保守预期夏普 1.0~1.2(见 v4 审计记录)。此面板为策略输出,非投资建议,自行决策与执行。
</div>
</body></html>"""

    out = os.path.join(桌面, "中证1000_每日面板.html")
    open(out, "w", encoding="utf-8").write(html)
    print(f"\n★ 明日推荐仓位 {明仓*100:+.1f}%（{方向}）")
    print(f"★ 面板已生成: {out}")
    if not os.environ.get("CSI1000_NO_OPEN"):   # 定时任务里不弹浏览器(launchd 设了该变量)
        os.system(f'open "{out}"')


if __name__ == "__main__":
    main()
