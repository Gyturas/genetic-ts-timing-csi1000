"""表达式树：基因的表示、随机生成、求值、交叉与变异的结构操作。

本模块是遗传编程（GP）的“基因层”：一个 alpha 因子被表示成一棵表达式树，
树上每个节点要么是算子（来自 ops.OPS），要么是叶子字段（field:<某个 TERMINAL>）。
GA 主循环在这些树上做初始化 / 求值 / 交叉 / 变异，逐代进化出更优的因子表达式。

设计要点：
  - Node 是不可变（frozen）对象，任何“修改”都通过构造新节点实现，避免副作用。
  - 所有随机性都走传入的 rng（random.Random 实例），保证整轮进化可复现。
  - 求值直接作用在“日期×资产”的 DataFrame 面板上，纯向量化（见 ops.py）。
"""
from __future__ import annotations

import random
from dataclasses import dataclass

import pandas as pd

# OPS：算子表（名字 -> OpSpec，含 arity/是否带窗口/实现函数）
# TERMINALS：可用作叶子的字段名（open/high/low/close/volume/...）
# WINDOWS：时序算子可选的回看窗口长度候选集
from csi1000.ga_alpha.ops import OPS, TERMINALS, WINDOWS


@dataclass(frozen=True)
class Node:
    """表达式树的一个节点（不可变）。

    既表示算子节点，也表示叶子节点，靠 op 前缀区分：
      - 叶子： op = "field:close"，children 为空，window 为 None
      - 算子： op = "ts_mean"，children 为子树元组，window 视算子是否带窗口而定
    """

    op: str                        # OPS 里的算子名，或 "field:<terminal>"
    children: tuple = ()           # 子节点元组；叶子为空 ()，一元算子长度 1，二元算子长度 2
    # window的值可以是int或者None，默认值是None
    window: int | None = None      # 回看窗口（仅带窗口的时序算子有值），非窗口算子/叶子为 None

    @property
    def is_leaf(self) -> bool:
        """是否为叶子节点：以 "field:" 开头即代表引用一个原始字段。"""
        return self.op.startswith("field:")

    def __str__(self) -> str:
        """把整棵树还原成人类可读的表达式字符串，例如 ts_mean(sub(close,open),20)。"""
        if self.is_leaf:
            # 叶子：去掉 "field:" 前缀，直接显示字段名（close / volume / ...）
            return self.op.removeprefix("field:")
        # 算子：先递归把每个子节点转成字符串
        args = [str(c) for c in self.children]
        if self.window is not None:
            # 带窗口算子把窗口值作为最后一个参数一并显示
            args.append(str(self.window))
        return f"{self.op}({','.join(args)})"


def size(node: Node) -> int:
    """树的节点总数（自身 1 个 + 所有子树节点数之和）。用于衡量表达式复杂度 / 惩罚膨胀。"""
    return 1 + sum(size(c) for c in node.children)


def depth(node: Node) -> int:
    """树的最大深度（叶子为 1，每往下一层 +1）。default=0 让叶子（无孩子）返回 1。"""
    return 1 + max((depth(c) for c in node.children), default=0)


def evaluate(node: Node, panels: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """自底向上求值整棵树，返回一个“日期×资产”的因子面板。

    panels: 字段名 -> 原始数据面板（如 {"close": df, "volume": df, ...}）。
    """
    if node.is_leaf:
        # 叶子：直接返回对应字段的原始面板
        return panels[node.op.removeprefix("field:")]
    spec = OPS[node.op]                              # 取出该算子的规格（arity/windowed/fn）
    args = [evaluate(c, panels) for c in node.children]  # 递归求值每个子树，得到子面板列表
    if spec.windowed:
        args.append(node.window)                    # 带窗口算子把 window 作为最后一个参数
    return spec.fn(*args)                            # 调用算子的向量化实现，得到本节点的面板


# ---------------------------------------------------------------- 随机生成

def random_leaf(rng: random.Random) -> Node:
    """随机生成一个叶子节点：从 TERMINALS 里等概率抽一个字段。"""
    return Node(f"field:{rng.choice(TERMINALS)}")


def random_tree(rng: random.Random, max_depth: int, full: bool = False) -> Node:
    """grow / full 两种初始化（ramped half-and-half 用）。

    - full=True （full 法）：不到深度上限就一定长算子，树被“撑满”到 max_depth。
    - full=False（grow 法）：每一层有 30% 概率提前收成叶子，长出形状参差不齐的树。
    ramped half-and-half 就是对不同 max_depth 各用一半 grow、一半 full，最大化初始种群多样性。
    """
    # 触底（深度只剩 1）或 grow 法掷中 30% 提前终止 —— 收成叶子
    if max_depth <= 1 or (not full and rng.random() < 0.3):
        return random_leaf(rng)
    name = rng.choice(list(OPS))                     # 等概率选一个算子
    spec = OPS[name]
    # 按算子的 arity 递归生成对应数量的子树，深度预算 -1 传下去
    children = tuple(random_tree(rng, max_depth - 1, full) for _ in range(spec.arity))
    window = rng.choice(WINDOWS) if spec.windowed else None  # 带窗口算子随机配一个窗口
    return Node(name, children, window)


# ---------------------------------------------------------------- 结构操作

def _all_paths(node: Node, prefix: tuple = ()) -> list[tuple]:
    """枚举树中每个节点的“路径坐标”，用子节点下标元组表示。

    根为 ()，根的第 0 个孩子为 (0,)，再往下第 1 个孩子为 (0,1)，以此类推。
    交叉 / 变异靠随机挑一条 path 来定位要操作的节点。
    """
    paths = [prefix]                                 # 先收录当前节点自身的路径
    for i, c in enumerate(node.children):
        paths.extend(_all_paths(c, prefix + (i,)))   # 递归收录每个子树，路径追加下标 i
    return paths


def _get(node: Node, path: tuple) -> Node:
    """沿路径逐层下钻，取出 path 指向的那个子节点。"""
    for i in path:
        node = node.children[i]
    return node


def _replace(node: Node, path: tuple, new: Node) -> Node:
    """返回一棵新树：把 path 处的子树替换成 new（不改动原树，纯函数式）。"""
    if not path:
        return new                                   # 路径走空 —— 到达目标位置，直接替换
    i, rest = path[0], path[1:]                       # 拆出当前该走的下标 i 和剩余路径
    children = list(node.children)                    # 复制孩子列表（元组不可变，转 list 才能改）
    children[i] = _replace(children[i], rest, new)    # 只在第 i 个孩子上递归替换，其余原样保留
    return Node(node.op, tuple(children), node.window)  # 用替换后的孩子重建本节点


def crossover(a: Node, b: Node, rng: random.Random, max_depth: int) -> Node:
    """把 b 的随机子树接到 a 的随机位置；超深则退回 a。

    这是子树交叉（subtree crossover）：从父代 a 挑一个嫁接点，用父代 b 的一段子树替换之，
    产生一个融合两者结构的子代。
    """
    pa = rng.choice(_all_paths(a))                    # a 上随机选一个嫁接位置
    pb = rng.choice(_all_paths(b))                    # b 上随机选一段子树
    child = _replace(a, pa, _get(b, pb))              # 把 b 的子树接到 a 的该位置
    return child if depth(child) <= max_depth else a  # 超过深度上限则放弃嫁接，退回原 a


def mutate(node: Node, rng: random.Random, max_depth: int) -> Node:
    """对一棵树做一处随机变异，返回新个体；超深则退回原树。

    kind 掷一次骰子，把变异分成四类（概率因附加结构条件而会向后倾斜）：
      kind<0.4                子树重生：整块子树换成新随机树（强变异）
      0.4≤kind<0.7 且非叶子    换算子：同 arity/窗口特性下换个运算符，保结构（中变异）
      目标带 window           窗口扰动：只改滚动窗口大小（弱变异）
      其余                    叶子换字段：换一个输入字段（弱变异）
    """
    kind = rng.random()                              # 一次性骰子，下面按区间切分变异类型
    paths = _all_paths(node)                          # 枚举所有节点位置
    path = rng.choice(paths)                          # 随机选一个变异点
    target = _get(node, path)                         # 取出该位置的目标节点

    if kind < 0.4:                                   # 子树重生
        sub = random_tree(rng, max_depth=3)          # 现场长一棵深度≤3 的新随机子树
        child = _replace(node, path, sub)            # 用它替换目标位置的整棵子树
    elif kind < 0.7 and not target.is_leaf:          # 同 arity 换算子（仅对算子节点有效）
        spec = OPS[target.op]
        # 找出所有“可平替”的算子：arity 与是否带窗口都相同、且不是自己 —— 保证换完结构合法
        same = [n for n, s in OPS.items() if s.arity == spec.arity and s.windowed == spec.windowed and n != target.op]
        if not same:
            return node                              # 没有可替换的候选，直接返回原树（不变异）
        # 只换算子名，孩子和窗口原样复用（如 add(x,y) -> sub(x,y)）
        new = Node(rng.choice(same), target.children, target.window)
        child = _replace(node, path, new)
    elif target.window is not None:                  # 窗口扰动（目标必须是带窗口的时序算子）
        # 算子与孩子不变，只把窗口随机换成另一个候选值（改“看多长历史”）
        new = Node(target.op, target.children, rng.choice(WINDOWS))
        child = _replace(node, path, new)
    else:                                            # 叶子换字段（兜底，主要针对叶子节点）
        child = _replace(node, path, random_leaf(rng))  # 换成一个新的随机字段

    return child if depth(child) <= max_depth else node  # 变异后超深则放弃，退回原树
