# -*- coding: utf-8 -*-
"""每日面板(v5.1):更新行情+个股 → 六库合奏+状态层 → 明日仓位 → 桌面 HTML。

流程:①更新13标的行情 ②更新个股面板与截面叶子(状态层所需)③重算六库信号
      ④状态层调制 ⑤生成面板
用法: PYTHONPATH=src python -m csi1000.engine.daily_dashboard [--k 1.2]
环境: CSI1000_NO_OPEN=1 时不自动打开浏览器(定时任务用)
"""
from __future__ import annotations

import argparse
import base64
import datetime as dt
import io
import os

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = ["Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False

from csi1000 import paths

桌面 = os.path.expanduser("~/Desktop")
成本 = 0.0005          # 512100 ETF 双边万5(ETF 无印花税,已属保守)


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
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=float, default=1.2)
    ap.add_argument("--skip-data", action="store_true", help="跳过数据更新(调试用)")
    a = ap.parse_args()

    if not a.skip_data:
        print("① 更新指数/ETF 行情 …")
        import csi1000.engine.update_market as um
        um.main()
        print("② 更新个股面板与截面叶子(状态层所需)…")
        import csi1000.engine.update_stocks as us
        us.main()
    print("③ 重算六库信号 + 状态层(约5分钟)…")
    from csi1000.engine.live_v51 import 算
    d = 算(a.k)

    import csi1000.engine.strategy as st
    zz = pd.read_csv(os.path.join(paths.行情缓存, "idx_sh000852.csv"),
                     parse_dates=["date"]).set_index("date")
    指数收 = zz["close"].pct_change()                       # 同花顺口径:指数收→收

    信号日 = d.index[-1]
    明仓 = float(d["仓位"].iloc[-1])
    p6 = float(d["p六库"].iloc[-1])
    恐慌 = bool(d["恐慌"].iloc[-1])
    收益, rf = d["收益"], d["货基"]
    ret = 收益.dropna()
    nav = (1 + ret.loc["2018":]).cumprod()
    持nav = (1 + d["标的开→开"].reindex(ret.index).fillna(0).loc["2018":]).cumprod()
    今年 = str(信号日.year)
    全 = _指标(ret, rf, "2018-01-01"); 年内 = _指标(ret, rf, f"{今年}-01-01")
    持全 = _指标(d["标的开→开"].reindex(ret.index).fillna(0), rf, "2018-01-01")
    当前回撤 = float(nav.iloc[-1] / nav.cummax().iloc[-1] - 1)

    方向 = "做多" if 明仓 > 0.02 else ("做空" if 明仓 < -0.02 else "空仓")
    色 = "#c0392b" if 明仓 > 0.02 else ("#27ae60" if 明仓 < -0.02 else "#7f8c8d")

    # ---- 滚动战绩:从上周一起 ----
    上周一 = (信号日 - pd.Timedelta(days=信号日.weekday() + 7)).normalize()
    段 = d.loc[上周一:].copy()
    段["实际持仓"] = d["仓位"].shift(2).reindex(段.index)     # T日结算的是T-2日信号
    策累, 持累, 行 = 1.0, 1.0, ""
    for t, row in 段.iterrows():
        if not np.isfinite(row["收益"]):
            continue
        策累 *= (1 + row["收益"]); 持累 *= (1 + row["标的开→开"])
        c = "#c0392b" if row["收益"] >= 0 else "#27ae60"
        ix = 指数收.get(t, np.nan)
        行 += (f"<tr><td>{t.date()}</td>"
              f"<td class=r>{row['实际持仓']*100:+.0f}%</td>"
              f"<td class=r>{row['标的开→开']*100:+.2f}%</td>"
              f"<td class=r style='color:{c};font-weight:500'>{row['收益']*100:+.2f}%</td>"
              f"<td class=r style='font-weight:500'>{(策累-1)*100:+.2f}%</td>"
              f"<td class=r style='color:#aaa'>{ix*100:+.2f}%</td></tr>")
    周策 = (策累 - 1) * 100; 周持 = (持累 - 1) * 100

    # ---- 待结算队列:已定但还没到结算日的仓位 ----
    队 = ""
    for i, t in enumerate(d.index[-2:]):
        队 += (f"<tr><td>{t.date()}</td><td class=r style='color:{色 if i==1 else '#555'};font-weight:500'>"
              f"{d['仓位'].loc[t]*100:+.1f}%</td>"
              f"<td class=r style='color:#bbb'>{'次日开盘建仓' if i==1 else '持有中'}</td></tr>")

    # ---- 图 ----
    fig, ax = plt.subplots(figsize=(9, 3.6))
    ax.plot(nav, lw=2, color="#c0392b", label=f"v5.1 策略 ({nav.iloc[-1]:.1f}×)")
    ax.plot(持nav, lw=1.2, ls="--", color="#95a5a6", label=f"持有512100 ({持nav.iloc[-1]:.1f}×)")
    ax.set_yscale("log"); ax.legend(fontsize=9); ax.grid(alpha=.3)
    ax.set_title("净值(对数轴,2018起,open→open 含成本)", fontsize=11)
    净值图 = _png(fig)

    fig, ax = plt.subplots(figsize=(9, 2.4))
    近 = d["仓位"].tail(60)
    ax.fill_between(近.index, 近.values * 100, 0, color="#2c6fbb", alpha=.5, step="mid")
    恐 = d["恐慌"].tail(60).astype(bool)
    if 恐.any():
        ax.fill_between(近.index, -100, 100, where=恐.values, color="#e74c3c", alpha=.08, step="mid")
    ax.axhline(0, color="k", lw=.5); ax.grid(alpha=.3); ax.set_ylim(-105, 105)
    ax.set_title("近60日目标仓位 %(红色底纹=恐慌态)", fontsize=11)
    仓位图 = _png(fig)

    def 卡(标, 指, c="#1a1a1a"):
        if 指 is None:
            return ""
        return (f"<div class=card><div class=t>{标}</div>"
                f"<div class=g><b style='color:{c}'>{指['年化']*100:.1f}%</b><span>年化</span></div>"
                f"<div class=g><b>{指['夏普']:.2f}</b><span>夏普</span></div>"
                f"<div class=g><b>{指['回撤']*100:.1f}%</b><span>回撤</span></div>"
                f"<div class=g><b>{指['卡玛']:.2f}</b><span>卡玛</span></div>"
                f"<div class=g><b>{指['累计']*100:.0f}%</b><span>累计</span></div></div>")

    html = f"""<!doctype html><html><head><meta charset=utf-8>
<title>中证1000 每日面板</title><style>
*{{margin:0;padding:0;box-sizing:border-box}}
body{{font-family:-apple-system,"PingFang SC",sans-serif;background:#f4f5f7;color:#1a1a1a;
     padding:24px;max-width:1020px;margin:auto}}
h1{{font-size:20px;margin-bottom:4px}} .sub{{color:#888;font-size:13px;margin-bottom:18px}}
.hero{{background:#fff;border-radius:14px;padding:24px;text-align:center;
      box-shadow:0 1px 4px #0001;margin-bottom:16px}}
.hero .pos{{font-size:54px;font-weight:700;color:{色};line-height:1}}
.hero .dir{{font-size:17px;color:{色};margin:6px 0}}
.hero .meta{{color:#888;font-size:12.5px;margin-top:8px}}
.badge{{display:inline-block;padding:2px 10px;border-radius:10px;font-size:12px;margin-left:6px}}
.cards{{display:flex;gap:12px;margin-bottom:16px;flex-wrap:wrap}}
.card{{background:#fff;border-radius:12px;padding:14px 16px;flex:1;min-width:230px;box-shadow:0 1px 4px #0001}}
.card .t{{font-size:12.5px;color:#888;margin-bottom:9px;font-weight:600}}
.g{{display:inline-block;width:19%;text-align:center;vertical-align:top}}
.g b{{display:block;font-size:15px}} .g span{{font-size:10.5px;color:#aaa}}
img{{width:100%;border-radius:12px;background:#fff;box-shadow:0 1px 4px #0001;margin-bottom:12px}}
.row{{display:flex;gap:12px;flex-wrap:wrap;margin-bottom:16px;align-items:flex-start}}
table{{width:100%;background:#fff;border-radius:12px;border-collapse:collapse;overflow:hidden;
      box-shadow:0 1px 4px #0001;font-size:13px}}
td{{padding:6px 12px;border-bottom:1px solid #f0f0f0}} .r{{text-align:right}}
thead td{{color:#888;font-size:11.5px;font-weight:600}}
.lbl{{font-size:12.5px;font-weight:600;color:#555;margin-bottom:6px}}
.note{{background:#fff;border-radius:12px;padding:14px 18px;box-shadow:0 1px 4px #0001;
      font-size:12.5px;color:#555;line-height:1.75;margin-bottom:14px}}
.note b{{color:#c0392b}}
</style></head><body>

<h1>中证1000 时序择时 · 每日面板</h1>
<div class=sub>v5.1(六库合奏 + 状态层风险预算回收,k={a.k}) · 信号日 {信号日.date()} · 生成 {dt.datetime.now():%Y-%m-%d %H:%M}</div>

<div class=hero>
  <div style="font-size:13px;color:#888;margin-bottom:6px">下一交易日 <b>开盘</b> · 512100 目标仓位</div>
  <div class=pos>{明仓*100:+.1f}%</div>
  <div class=dir>{方向}{'<span class=badge style="background:#fdecea;color:#c0392b">恐慌态·多头已折半</span>' if 恐慌 else ''}</div>
  <div class=meta>六库原始信号 {p6*100:+.1f}% &nbsp;·&nbsp; ×{a.k} 预算回收后截断至 {明仓*100:+.1f}%
       &nbsp;·&nbsp; 当前回撤 {当前回撤*100:.1f}%</div>
</div>

<div class=cards>{卡("v5.1 策略 · 2018至今", 全, 色)}{卡(f"策略 · {今年}年内", 年内)}{卡("持有512100 · 2018至今", 持全, "#7f8c8d")}</div>

<div class=note>
<b>为什么这里的日收益和同花顺对不上?</b> 同花顺显示的是<b>指数收盘→收盘</b>,本策略在
<b>开盘</b>成交,结算的是<b>ETF开盘→开盘</b>。二者只共享"隔夜"那一段:<br>
&nbsp;&nbsp;开→开[t] = 隔夜[t] + <b>昨日</b>日内 &nbsp;&nbsp;|&nbsp;&nbsp; 收→收[t] = 隔夜[t] + <b>今日</b>日内<br>
A股日内波动占总方差 85% 且前后日不相关,所以两个口径的<b>单日</b>数字几乎正交(近250日相关仅 0.07),
但<b>长期累计一致</b>(近两年 +51.1% vs +52.2%)。表格最后一列给出指数收→收供对照。
</div>

<div class=row>
  <div style="flex:1.55">
    <div class=lbl>滚动战绩(自上周一 {上周一.date()} 起 · 按模型仓位实吃)</div>
    <table><thead><tr><td>结算日</td><td class=r>实际持仓</td><td class=r>标的开→开</td>
      <td class=r>策略当日</td><td class=r>累计</td><td class=r>指数收→收</td></tr></thead>
      {行}
      <tr style="font-weight:700;background:#fafafa"><td>合计</td><td class=r>—</td>
        <td class=r>{周持:+.2f}%</td>
        <td class=r style="color:{'#c0392b' if 周策>=0 else '#27ae60'}">{周策:+.2f}%</td>
        <td class=r>超额 {周策-周持:+.2f}%</td><td class=r style="color:#aaa">—</td></tr>
    </table>
  </div>
  <div style="flex:1">
    <div class=lbl>待结算队列(收盘出信号 → 次日开盘建仓 → 隔日开盘结算)</div>
    <table><thead><tr><td>信号日</td><td class=r>目标仓位</td><td class=r>状态</td></tr></thead>{队}</table>
    <div class=note style="margin-top:10px;font-size:12px">
      成本:双边万5(512100 ETF 无印花税)· 年换手约 5800% → 年成本约 2.9%,<b>已计入</b>上方全部数字。<br>
      仓位经 ×{a.k} 后截断至 ±100%,<b>无需融资或杠杆</b>;空头腿以 IM 中证1000 股指期货实现。<br>
      空仓部分吃货基(511880)。
    </div>
  </div>
</div>

<img src="{净值图}">
<img src="{仓位图}">

<div class=note>
口径:信号 T 日<b>收盘</b>生成 → T+1 <b>开盘</b>建仓 → T+2 开盘结算(结算滞后2格),回测与实盘同口径。
状态层:limit_net(涨停−跌停占比)20日均的滚动2年分位 &lt;10% 判恐慌(占比约11%交易日),
恐慌时多头×0.5,整体×{a.k} 回收风险预算。<br>
选择偏差已知,保守预期夏普 1.2~1.5。此面板为策略输出,非投资建议,自行决策与执行。
</div>
</body></html>"""

    out = os.path.join(桌面, "中证1000_每日面板.html")
    open(out, "w", encoding="utf-8").write(html)
    print(f"\n★ 明日开盘目标仓位 {明仓*100:+.1f}%({方向}){' · 恐慌态' if 恐慌 else ''}")
    print(f"★ 本周(自{上周一.date()})策略 {周策:+.2f}% vs 持有 {周持:+.2f}%")
    print(f"★ 面板: {out}")
    if not os.environ.get("CSI1000_NO_OPEN"):
        os.system(f'open "{out}"')


if __name__ == "__main__":
    main()
