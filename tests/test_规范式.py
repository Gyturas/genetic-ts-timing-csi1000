# -*- coding: utf-8 -*-
"""表达式规范化的单元测试。
正例来自审计实测的 13 对 |Spearman|=1.0000 重复因子;反例保证不误折叠。"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from csi1000.ga_alpha.expr import Node, 规范式, 去向规范式

def L(f): return Node(f"field:{f}")
def U(op, a, w=None): return Node(op, (a,), w)
def B(op, a, b, w=None): return Node(op, (a, b), w)
r, c, h, v, am = L("returns"), L("close"), L("high"), L("volume"), L("amount")

同 = [  # (说明, 树甲, 树乙) —— 去向规范式必须相同
 ("A+I 实测对: slog(ts_max(returns,10)) ≡ ts_min(neg(returns),10)",
  U("slog", U("ts_max", r, 10)), U("ts_min", U("neg", r), 10)),
 ("B 实测对: ts_std(high,5) ≡ neg(ts_std(high,5))",
  U("ts_std", h, 5), U("neg", U("ts_std", h, 5))),
 ("C 实测组: ts_std(close,5) ≡ abs(ts_std(close,5)) ≡ ts_std(abs(close),5)",
  U("ts_std", c, 5), U("abs", U("ts_std", c, 5))),
 ("C 恒正叶子: ts_std(abs(close),5) ≡ ts_std(close,5)",
  U("ts_std", U("abs", c), 5), U("ts_std", c, 5)),
 ("D: ts_sum(a,20) ≡ ts_mean(a,20)(正仿射,rank下恒等)",
  U("ts_sum", r, 20), U("ts_mean", r, 20)),
 ("C+D 实测对: ts_mean(abs(disp),5) ≡ ts_sum(disp,5)",
  U("ts_mean", U("abs", L("disp")), 5), U("ts_sum", L("disp"), 5)),
 ("E: delta(a,5) ≡ sub(a,delay(a,5))", U("delta", c, 5), B("sub", c, U("delay", c, 5))),
 ("F: delay∘delay 可加", U("delay", U("delay", c, 5), 5), U("delay", c, 10)),
 ("G: add 交换律", B("add", c, v), B("add", v, c)),
 ("G: mul 交换律", B("mul", h, r), B("mul", r, h)),
 ("G: ts_corr 交换律", B("ts_corr", c, v, 20), B("ts_corr", v, c, 20)),
 ("H: ts_rank 吸收 slog", U("ts_rank", U("slog", r), 20), U("ts_rank", r, 20)),
 ("H: ts_argmax 吸收 ssqrt", U("ts_argmax", U("ssqrt", v), 10), U("ts_argmax", v, 10)),
 ("I: ts_argmin(a,w) ≡ ts_argmax(neg(a),w)", U("ts_argmin", c, 5), U("ts_argmax", U("neg", c), 5)),
 ("J: ts_std(neg(a),w) ≡ ts_std(a,w)", U("ts_std", U("neg", r), 10), U("ts_std", r, 10)),
 ("neg∘neg → id", U("neg", U("neg", c)), c),
 ("abs∘neg → abs", U("abs", U("neg", r)), U("abs", r)),
 ("根部 ssqrt 剥离", U("ssqrt", U("ts_std", r, 5)), U("ts_std", r, 5)),
 ("嵌套组合", U("slog", U("ts_sum", U("abs", U("neg", r)), 20)),
              U("ts_mean", U("abs", r), 20)),
]
异 = [  # 必须【不】相同 —— 防误折叠
 ("窗口不同", U("ts_std", r, 5), U("ts_std", r, 10)),
 ("叶子不同", U("ts_std", c, 5), U("ts_std", h, 5)),
 ("算子不同", U("ts_mean", r, 5), U("ts_std", r, 5)),
 ("sub 不可交换", B("sub", c, h), B("sub", h, c)),
 ("div 不可交换", B("div", v, am), B("div", am, v)),
 ("abs 对可负量不可剥离", U("abs", r), r),
 ("abs 对可负截面叶子不可剥离", U("abs", L("limit_net")), L("limit_net")),
 ("ts_mean 不吸收 slog(非严格增吸收算子)", U("ts_mean", U("slog", r), 5), U("ts_mean", r, 5)),
 ("内层 neg 不能随意剥(ts_mean)", U("ts_mean", U("neg", r), 5), U("ts_mean", r, 5)),
]
坏 = 0
for 说, a, b in 同:
    ka, kb = 去向规范式(a), 去向规范式(b)
    ok = ka == kb
    坏 += not ok
    print(f"  {'✓' if ok else '✗ 应相同但不同'} {说}")
    if not ok: print(f"       {ka}\n       {kb}")
for 说, a, b in 异:
    ka, kb = 去向规范式(a), 去向规范式(b)
    ok = ka != kb
    坏 += not ok
    print(f"  {'✓' if ok else '✗ 误折叠!'} 反例:{说}")
    if not ok: print(f"       两者都归为 {ka}")
print(f"\n{'全部通过' if 坏==0 else f'{坏} 条失败'}  (同 {len(同)} 条 / 异 {len(异)} 条)")
sys.exit(1 if 坏 else 0)
