# -*- coding: utf-8 -*-
"""单个大类的季度 walk-forward 引擎。

每季在 cutoff(季首前最后一个交易日)执行,只见 ≤cutoff 数据:
  1) AR-GARCH 元老逐季扩展  2) 体检/退役(元老无豁免)  3) 复员
  4) GA 挖矿  5) 跨类转会(读兄弟类上一季快照,as-of 安全)  6) 入库考试
  7) 选用(综合分+滞回)  8) 信号合成→映射→闸→平滑  9) 日结算(等权房间+货基+成本)

因子值面板:表达式全因果 ⇒ 全史一次求值与逐日 as-of 计算逐位相同(前两版已验证),
故 state 只存表达式树,重启时重建值缓存;AR-GARCH 例外(参数季度重估),由 Recorder 持久化。
断点续跑:done_q 必须是季度表连续前缀,否则拒绝启动(防存档污染,前版教训)。
"""
from __future__ import annotations

import os
import pickle
import time

import numpy as np
import pandas as pd
from scipy.stats import norm

from csi1000.ga_alpha import expr as gexpr
from csi1000.ga_alpha.evolve import run_ga

from csi1000.walkforward import config as C
from csi1000.walkforward import metrics as M
from csi1000.walkforward.elders import ARGARCH_NAMES, ArgarchRecorder, trend_elder_values
from csi1000.walkforward import ga_setup


def quarter_list(start: str, end: str) -> list[pd.Period]:
    return list(pd.period_range(pd.Timestamp(start), pd.Timestamp(end), freq="Q"))


class ClassEngine:
    def __init__(self, cname: str, data: dict, log=print):
        self.cname = cname
        self.log = lambda *a: log(f"[{cname}]", *a)
        cd = data["classes"][cname]
        self.panels = cd["panels"]
        self.spec = cd["spec"]
        self.rf = data["rf"]
        self.etf_ret = data["etf_ret"][list(self.spec["etf_map"])]
        self.cal = self.panels["close"].index
        self.fwd = M.fwd_excess(self.panels["returns"], self.rf)
        self.arch = M.archetypes(self.panels)
        ga_setup.install(self.rf)

        self.cls_idx = list(C.CLASSES).index(cname)
        self.outdir = os.path.join(C.STATE_DIR, cname)
        self.shared = os.path.join(C.STATE_DIR, "shared")
        os.makedirs(self.outdir, exist_ok=True)
        os.makedirs(self.shared, exist_ok=True)
        self.recorder = ArgarchRecorder(self.panels["returns"],
                                        os.path.join(self.outdir, "argarch"))
        self.qs_all = quarter_list(C.WARMUP_START, C.END)      # 含预热季(只记AR-GARCH)
        self.qs_build = [q for q in self.qs_all if q.start_time >= pd.Timestamp(C.BUILD_START)]
        self.state = self._load_state()
        self.values: dict[int, pd.DataFrame] = {}
        self._rebuild_values()

    # ---------------- 状态 ----------------
    def _load_state(self) -> dict:
        fp = os.path.join(self.outdir, "state.pkl")
        if os.path.exists(fp):
            with open(fp, "rb") as f:
                st = pickle.load(f)
            done = st["done_q"]
            expect = [str(q) for q in self.qs_all[:len(done)]]
            assert done == expect, f"存档不连续: {done[-3:]} vs {expect[-3:]}, 拒绝续跑"
            assert self.recorder.load() or not done, "argarch 存档缺失"
            self.log(f"续跑: 已完成 {len(done)} 季 (至 {done[-1] if done else '-'})")
            return st
        return {"factors": {}, "next_fid": 0, "seats": [], "prev_scores": {},
                "p_exec": {}, "daily": [], "daily_expo": [], "events": [],
                "done_q": [], "selections": [], "elder_inited": False}

    def _save_state(self) -> None:
        fp = os.path.join(self.outdir, "state.pkl")
        tmp = fp + ".tmp"
        with open(tmp, "wb") as f:
            pickle.dump(self.state, f)
        os.replace(tmp, fp)
        self.recorder.save()

    def _rebuild_values(self) -> None:
        for fid, fac in self.state["factors"].items():
            self.values[fid] = self._compute_values(fac)

    def _compute_values(self, fac: dict) -> pd.DataFrame:
        if fac["kind"] == "argarch":
            return self.recorder.values(fac["name"])
        if fac["kind"] == "elder":
            return trend_elder_values(self.panels)[fac["name"]]
        v = gexpr.evaluate(fac["node"], self.panels)
        return v.replace([np.inf, -np.inf], np.nan)

    # ---------------- 主循环 ----------------
    def run(self) -> None:
        for qi, q in enumerate(self.qs_all):
            if str(q) in self.state["done_q"]:
                continue
            qdays = self.cal[(self.cal >= q.start_time) & (self.cal <= q.end_time)]
            pre = self.cal[self.cal < q.start_time]
            if len(qdays) == 0 or len(pre) == 0:
                self.state["done_q"].append(str(q))
                continue
            cutoff = pre[-1]
            t0 = time.time()
            self.recorder.extend(cutoff, qdays)
            if q.start_time >= pd.Timestamp(C.BUILD_START):
                self._quarter_step(q, qi, cutoff, qdays)
            self.state["done_q"].append(str(q))
            self._save_state()
            self.log(f"{q} 完成 ({time.time()-t0:.0f}s) 库={self._n_active()} "
                     f"席={len(self.state['seats'])}")
        self._export()

    def _n_active(self) -> int:
        return sum(f["status"] == "active" for f in self.state["factors"].values())

    def _actives(self) -> list[int]:
        return [i for i, f in self.state["factors"].items() if f["status"] == "active"]

    # ---------------- 季度步 ----------------
    def _quarter_step(self, q, qi, cutoff, qdays) -> None:
        if not self.state["elder_inited"]:
            self._init_elders(q, cutoff)
        self._physical(q, cutoff)
        self._reenlist(q, cutoff)
        pool = self._mine(qi, cutoff) + self._transfer_pool(q)
        self._admit(q, cutoff, pool)
        scores = self._select(q, cutoff)
        self._signals_and_settle(q, cutoff, qdays, scores)
        self._snapshot(q, scores)

    def _init_elders(self, q, cutoff) -> None:
        warm0 = pd.Timestamp(C.WARMUP_START)
        trend = trend_elder_values(self.panels)
        elder_specs = [(n, "elder") for n in trend] + [(n, "argarch") for n in ARGARCH_NAMES]
        for name, kind in elder_specs:
            vals = (trend[name] if kind == "elder" else self.recorder.values(name))
            S = M.zsig(vals).loc[warm0:cutoff]
            ic = M.daily_ic(S, self.fwd.loc[S.index]).mean()
            sign = 1.0 if not np.isfinite(ic) or ic >= 0 else -1.0
            fid = self.state["next_fid"]; self.state["next_fid"] += 1
            self.state["factors"][fid] = {
                "fid": fid, "name": name, "kind": kind, "node": None, "sign": sign,
                "born_q": str(q), "status": "active", "yellow": 0, "retired_q": None,
                "reenlisted": False, "family": "元老", "home": self.cname}
            self.values[fid] = vals
            self.state["events"].append((str(q), "元老入库", name, f"sign={sign:+.0f} 预热IC={ic:+.3f}"))
        self.state["elder_inited"] = True
        self.log(f"元老13人入库,方向由预热段(2013-14)冻结")

    def _signed_S(self, fid: int, upto=None, tail: int | None = None) -> pd.DataFrame:
        v = self.values[fid]
        if upto is not None:
            v = v.loc[:upto]
        if tail:
            v = v.tail(tail + C.Z_WIN)
        S = M.zsig(v, self.state["factors"][fid]["sign"])
        return S.tail(tail) if tail else S

    def _physical(self, q, cutoff) -> None:
        """体检/退役:trailing 250日带号类均IC + 12月NW t;宽限期250日。"""
        age_days = {}
        for fid in self._actives():
            fac = self.state["factors"][fid]
            born = pd.Period(fac["born_q"]).start_time
            age = int(((self.cal > born) & (self.cal <= cutoff)).sum())
            if age < C.GRACE_DAYS:
                continue
            S = self._signed_S(fid, cutoff, C.SEL_WIN)
            F = self.fwd.loc[S.index]
            ic = float(M.daily_ic(S, F).mean())
            t = M.nw_t(M.monthly_ic_series(S, F))
            if (np.isfinite(ic) and ic < C.RETIRE_HARD_IC) or (np.isfinite(t) and t < C.RETIRE_HARD_T):
                fac.update(status="retired", retired_q=str(q), yellow=0)
                self.state["events"].append((str(q), "立即退役", fac["name"], f"ic={ic:+.3f} t={t:+.1f}"))
            elif np.isfinite(ic) and ic < C.RETIRE_IC:
                fac["yellow"] += 1
                if fac["yellow"] >= C.YELLOW_Q:
                    fac.update(status="retired", retired_q=str(q), yellow=0)
                    self.state["events"].append((str(q), "黄牌退役", fac["name"], f"ic={ic:+.3f}"))
            else:
                fac["yellow"] = 0

    def _reenlist(self, q, cutoff) -> None:
        act_S = [self._signed_S(f, cutoff, C.SEL_WIN) for f in self._actives()]
        for fid, fac in self.state["factors"].items():
            if fac["status"] != "retired" or fac["reenlisted"] or fac["retired_q"] is None:
                continue
            if (pd.Period(str(q)) - pd.Period(fac["retired_q"])).n < C.REENLIST_Q:
                continue
            S = self._signed_S(fid, cutoff, C.REENLIST_WIN)
            F = self.fwd.loc[S.index]
            ic = float(M.daily_ic(S, F).mean())
            if not np.isfinite(ic) or ic < C.REENLIST_IC:
                continue
            S250 = S.tail(C.SEL_WIN)
            if M.residual_ic(S250, self.fwd.loc[S250.index], act_S) < C.ADMIT_RESID_IC:
                continue
            fac.update(status="active", reenlisted=True, yellow=0)
            self.state["events"].append((str(q), "复员", fac["name"], f"场外24月IC={ic:+.3f}"))

    def _mine(self, qi, cutoff) -> list[dict]:
        pre = self.cal[self.cal <= cutoff]
        if len(pre) < C.TRAIN_MIN_DAYS + C.VALID_DAYS:
            return []
        train_end, valid_end = pre[-C.VALID_DAYS - 1], cutoff
        sliced = {k: v.loc[:cutoff] for k, v in self.panels.items()}
        cfg = {"ga": dict(C.GA, seed=C.GA["seed"] + qi * 7 + self.cls_idx),
               "hall_of_fame": dict(C.HOF),
               "fitness": {"horizon": 1, "min_coverage": C.FIT_MIN_COVERAGE},
               "split": {"train_end": str(train_end.date()), "valid_end": str(valid_end.date())}}
        hof = run_ga(sliced, cfg, log=lambda *a: None)
        return [{"src": "GA", "expr": e["expr"], "node": e["_tree"],
                 "rank": abs(e["valid_ic"])}
                for e in sorted(hof.entries, key=lambda x: -abs(x["valid_ic"]))]

    def _transfer_pool(self, q) -> list[dict]:
        """读兄弟类上一季快照(等待,含超时放弃);候选=其现役综合分前10。"""
        if not C.TRANSFER_ENABLED:
            return []
        prev = str(pd.Period(str(q)) - 1)
        out, t0 = [], time.time()
        for sib in C.CLASSES:
            if sib == self.cname:
                continue
            fp = os.path.join(self.shared, f"{sib}_{prev}.pkl")
            while not os.path.exists(fp):
                if pd.Period(prev).start_time < pd.Timestamp(C.BUILD_START):
                    break
                if time.time() - t0 > C.TRANSFER_TIMEOUT:
                    self.log(f"等{sib}快照超时,本季放弃其转会候选")
                    break
                time.sleep(20)
            if not os.path.exists(fp):
                continue
            try:
                with open(fp, "rb") as f:
                    snap = pickle.load(f)
            except Exception:
                continue
            for cand in snap[:C.TRANSFER_TOP]:
                if cand["node"] is None:
                    continue                      # 元老/AR-GARCH 不跨类转会
                out.append({"src": f"转会:{sib}", "expr": cand["expr"],
                            "node": cand["node"], "rank": cand.get("score", 0.0)})
        return out

    def _admit(self, q, cutoff, pool) -> None:
        if not pool:
            return
        known = {f.get("expr_str") or f["name"] for f in self.state["factors"].values()}
        val_days = self.cal[self.cal <= cutoff][-C.VALID_DAYS:]
        act_ids = self._actives()
        act_S = [self._signed_S(f, cutoff, C.VALID_DAYS) for f in act_ids]
        lib_fams = pd.Series([self.state["factors"][f]["family"] for f in act_ids])
        admitted = 0
        cost = C.COST_QDII if self.cname == "跨境" else C.COST
        for cand in sorted(pool, key=lambda x: -x["rank"]):
            if admitted >= C.ADMIT_PER_Q or cand["expr"] in known or cand["node"] is None:
                continue
            vals = gexpr.evaluate(cand["node"], self.panels).replace([np.inf, -np.inf], np.nan)
            S0 = M.zsig(vals).loc[val_days]
            F = self.fwd.loc[val_days]
            ic0 = float(M.daily_ic(S0, F).mean())
            if not np.isfinite(ic0):
                continue
            sign = 1.0 if ic0 >= 0 else -1.0
            S = sign * S0
            sc = M.scorecard(S, F)
            reasons = []
            if sc["ic"] < C.ADMIT_IC: reasons.append(f"ic={sc['ic']:.3f}")
            if not (np.isfinite(sc["t"]) and sc["t"] >= C.ADMIT_T): reasons.append(f"t={sc['t']:.1f}")
            if not (np.isfinite(sc["xs_pos"]) and sc["xs_pos"] >= C.ADMIT_XS_POS): reasons.append("跨品种")
            if not (np.isfinite(sc["fast_ic"]) and sc["fast_ic"] >= 0): reasons.append("快分量")
            if not (np.isfinite(sc["event"]) and sc["event"] > cost): reasons.append("事件收益")
            if act_S and M.max_corr_vs(S, act_S) > C.ADMIT_MAX_CORR: reasons.append("相关")
            if act_S and M.residual_ic(S, F, act_S) < C.ADMIT_RESID_IC: reasons.append("残差")
            fam = M.family_tag(S, self.arch)
            if (fam != "其他" and len(act_ids) >= 5
                    and (lib_fams.eq(fam).sum() + 1) / (len(act_ids) + 1) > C.FAMILY_CAP):
                reasons.append("家族配额")
            if reasons:
                continue
            fid = self.state["next_fid"]; self.state["next_fid"] += 1
            self.state["factors"][fid] = {
                "fid": fid, "name": cand["expr"][:60], "expr_str": cand["expr"],
                "kind": "ga", "node": cand["node"], "sign": sign, "born_q": str(q),
                "status": "active", "yellow": 0, "retired_q": None,
                "reenlisted": False, "family": fam, "home": cand["src"]}
            self.values[fid] = vals
            act_ids.append(fid)
            act_S.append(S)
            lib_fams = pd.concat([lib_fams, pd.Series([fam])], ignore_index=True)
            admitted += 1
            self.state["events"].append(
                (str(q), "入库", cand["expr"][:80],
                 f"{cand['src']} ic={sc['ic']:+.3f} t={sc['t']:.1f} 族={fam} sign={sign:+.0f}"))

    def _select(self, q, cutoff) -> dict[int, float]:
        """综合分(组内z加权)+ 滞回换届;黄牌×0.5。返回 fid→分数。"""
        act = self._actives()
        if not act:
            self.state["seats"] = []
            return {}
        rows = {}
        for fid in act:
            S = self._signed_S(fid, cutoff, C.SEL_WIN)
            F = self.fwd.loc[S.index]
            sc = M.scorecard(S, F)
            rows[fid] = {"ic": sc["ic"], "pnl": sc["pnl"], "hit": sc["hit_gain"],
                         "dscorr": sc["dscorr"],
                         "icir": M.icir36(self.values[fid], self.state["factors"][fid]["sign"],
                                          self.fwd, cutoff)}
        df = pd.DataFrame(rows).T
        z = (df - df.mean()) / df.std().replace(0, np.nan)
        z = z.fillna(0.0)
        score = sum(C.SCORE_W[k] * z[k] for k in C.SCORE_W)
        for fid in act:
            if self.state["factors"][fid]["yellow"] > 0:
                score[fid] *= C.YELLOW_PENALTY
        scores = score.to_dict()

        prev = self.state["prev_scores"]
        seats = [f for f in self.state["seats"] if f in scores]     # 退役者自动腾位
        ranked = sorted(scores, key=lambda f: -scores[f])
        for cand_f in ranked:                                        # 空位按榜补齐
            if len(seats) >= C.SEATS:
                break
            if cand_f not in seats:
                seats.append(cand_f)
        changed = True
        while changed:                                               # 滞回:连续2季胜出才换
            changed = False
            challengers = [f for f in ranked if f not in seats]
            if not challengers or not seats:
                break
            ch = challengers[0]
            weakest = min(seats, key=lambda f: scores[f])
            if (scores[ch] > scores[weakest]
                    and ch in prev and weakest in prev and prev[ch] > prev[weakest]):
                seats[seats.index(weakest)] = ch
                self.state["events"].append((str(q), "换届", str(ch), f"顶替{weakest}"))
                changed = True
        self.state["seats"] = seats[:C.SEATS]
        self.state["prev_scores"] = scores
        self.state["selections"].append(
            (str(q), [(f, self.state["factors"][f]["name"], round(scores[f], 3))
                      for f in self.state["seats"]]))
        return scores

    # ---------------- 信号→仓位→结算 ----------------
    def _signals_and_settle(self, q, cutoff, qdays, scores) -> None:
        seats = self.state["seats"]
        insts = list(self.panels["close"].columns)
        end = qdays[-1]
        hist = self.cal[self.cal <= end]
        need = hist[-(C.Z_WIN + C.MAP_WIN + len(qdays) + 300):]

        if seats:
            ws = {}
            for fid in seats:
                S = self._signed_S(fid, cutoff, C.SEL_WIN)
                ic = float(M.daily_ic(S, self.fwd.loc[S.index]).mean())
                ws[fid] = max(ic, 0.0)
            tot = sum(ws.values())
            ws = ({f: w / tot for f, w in ws.items()} if tot > 1e-9
                  else {f: 1 / len(seats) for f in seats})
            comb = None
            for fid, w in ws.items():
                Sf = M.zsig(self.values[fid].reindex(need), self.state["factors"][fid]["sign"])
                comb = Sf * w if comb is None else comb.add(Sf * w, fill_value=0.0)
        else:
            comb = pd.DataFrame(0.0, index=need, columns=insts)

        pi = comb.rolling(C.MAP_WIN).rank(pct=True) - 0.5 / C.MAP_WIN
        with np.errstate(invalid="ignore"):
            p_raw = pd.DataFrame(norm.ppf(pi.clip(1e-6, 1 - 1e-6)) + C.MAP_SHIFT,
                                 index=pi.index, columns=pi.columns).clip(0, 1)
        close = self.panels["close"].loc[need]
        gate = close.ge(close.rolling(C.TREND_GATE_WIN).mean())
        enough = close.notna().rolling(C.MIN_HISTORY, min_periods=1).sum() >= C.MIN_HISTORY
        p_gated = p_raw.where(gate & enough, 0.0).fillna(0.0)
        p_sm = p_gated.ewm(span=C.EMA_SMOOTH, min_periods=1).mean()
        p_sm = p_sm.where(p_gated > 0, 0.0)          # 闸零即时清零,平滑不拖尾
        p_q = p_sm.loc[qdays]

        etf_map = self.spec["etf_map"]
        cost = {e: (C.COST_QDII if e in C.QDII_ETFS else C.COST) for e in etf_map}
        p_exec_prev = dict(self.state["p_exec"])
        r_etf = self.etf_ret.loc[qdays]
        rf_q = self.rf.loc[qdays]
        expo_prev_map = self.state.get("expo_prev", {})

        for d in qdays:
            listed = [e for e in etf_map if pd.notna(r_etf.at[d, e])]
            room = 1.0 / len(listed) if listed else 0.0
            expo, ret_d, turn_cost = {}, 0.0, 0.0
            for e in etf_map:
                if e not in listed:
                    p_exec_prev[e] = 0.0
                    expo[e] = 0.0
                    continue
                tgt = float(np.mean([p_q.at[d, i] for i in etf_map[e]]))
                prev_p = p_exec_prev.get(e, 0.0)
                if tgt < 1e-9:
                    newp = 0.0
                elif abs(tgt - prev_p) < C.NO_TRADE_BAND:
                    newp = prev_p
                else:
                    newp = tgt
                p_exec_prev[e] = newp
                expo[e] = newp * room
            for e in etf_map:
                pe = expo_prev_map.get(e, 0.0)
                r = r_etf.at[d, e]
                ret_d += pe * (0.0 if pd.isna(r) else float(r))
                turn_cost += abs(expo[e] - pe) * cost[e]
            cash = max(0.0, 1.0 - sum(expo_prev_map.get(e, 0.0) for e in etf_map))
            ret_d += cash * float(rf_q.at[d]) - turn_cost
            bench = np.nanmean([r_etf.at[d, e] for e in listed]) if listed else float(rf_q.at[d])
            self.state["daily"].append(
                (d, ret_d, sum(expo.values()), float(bench), turn_cost))
            self.state["daily_expo"].append((d, dict(expo)))
            expo_prev_map = expo
        self.state["p_exec"] = p_exec_prev
        self.state["expo_prev"] = expo_prev_map

    def _snapshot(self, q, scores) -> None:
        cands = []
        for fid in sorted(self._actives(), key=lambda f: -scores.get(f, -9)):
            fac = self.state["factors"][fid]
            if fac["node"] is None:
                continue
            cands.append({"expr": fac.get("expr_str", fac["name"]), "node": fac["node"],
                          "score": scores.get(fid, 0.0)})
        tmp = os.path.join(self.shared, f"{self.cname}_{q}.pkl.tmp")
        fp = os.path.join(self.shared, f"{self.cname}_{q}.pkl")
        with open(tmp, "wb") as f:
            pickle.dump(cands, f)
        os.replace(tmp, fp)

    # ---------------- 导出 ----------------
    def _export(self) -> None:
        df = pd.DataFrame(self.state["daily"],
                          columns=["date", "ret", "expo", "bench", "cost"]).set_index("date")
        df = df[~df.index.duplicated(keep="last")].sort_index()
        df.to_csv(os.path.join(self.outdir, "daily.csv"))
        pd.DataFrame(self.state["events"],
                     columns=["quarter", "event", "name", "detail"]) \
            .to_csv(os.path.join(self.outdir, "events.csv"), index=False)
        rows = [(qq, i + 1, fid, name, sc)
                for qq, lst in self.state["selections"]
                for i, (fid, name, sc) in enumerate(lst)]
        pd.DataFrame(rows, columns=["quarter", "seat", "fid", "name", "score"]) \
            .to_csv(os.path.join(self.outdir, "selections.csv"), index=False)
        ex = pd.DataFrame({d: e for d, e in self.state["daily_expo"]}).T.sort_index()
        ex = ex[~ex.index.duplicated(keep="last")]
        ex.to_csv(os.path.join(self.outdir, "etf_expo.csv"))
        self.log("导出完成:", self.outdir)
