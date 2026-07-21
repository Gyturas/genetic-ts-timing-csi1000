"""表达式树：基因的表示、随机生成、求值、交叉与变异的结构操作。"""
from __future__ import annotations

import random
from dataclasses import dataclass

import pandas as pd

from ga_alpha.ops import OPS, TERMINALS, WINDOWS


@dataclass(frozen=True)
class Node:
    op: str                        # OPS 里的算子名，或 "field:<terminal>"
    children: tuple = ()
    window: int | None = None

    @property
    def is_leaf(self) -> bool:
        return self.op.startswith("field:")

    def __str__(self) -> str:
        if self.is_leaf:
            return self.op.removeprefix("field:")
        args = [str(c) for c in self.children]
        if self.window is not None:
            args.append(str(self.window))
        return f"{self.op}({','.join(args)})"


def size(node: Node) -> int:
    return 1 + sum(size(c) for c in node.children)


def depth(node: Node) -> int:
    return 1 + max((depth(c) for c in node.children), default=0)


def evaluate(node: Node, panels: dict[str, pd.DataFrame]) -> pd.DataFrame:
    if node.is_leaf:
        return panels[node.op.removeprefix("field:")]
    spec = OPS[node.op]
    args = [evaluate(c, panels) for c in node.children]
    if spec.windowed:
        args.append(node.window)
    return spec.fn(*args)


# ---------------------------------------------------------------- 随机生成

def random_leaf(rng: random.Random) -> Node:
    return Node(f"field:{rng.choice(TERMINALS)}")


def random_tree(rng: random.Random, max_depth: int, full: bool = False) -> Node:
    """grow / full 两种初始化（ramped half-and-half 用）。"""
    if max_depth <= 1 or (not full and rng.random() < 0.3):
        return random_leaf(rng)
    name = rng.choice(list(OPS))
    spec = OPS[name]
    children = tuple(random_tree(rng, max_depth - 1, full) for _ in range(spec.arity))
    window = rng.choice(WINDOWS) if spec.windowed else None
    return Node(name, children, window)


# ---------------------------------------------------------------- 结构操作

def _all_paths(node: Node, prefix: tuple = ()) -> list[tuple]:
    paths = [prefix]
    for i, c in enumerate(node.children):
        paths.extend(_all_paths(c, prefix + (i,)))
    return paths


def _get(node: Node, path: tuple) -> Node:
    for i in path:
        node = node.children[i]
    return node


def _replace(node: Node, path: tuple, new: Node) -> Node:
    if not path:
        return new
    i, rest = path[0], path[1:]
    children = list(node.children)
    children[i] = _replace(children[i], rest, new)
    return Node(node.op, tuple(children), node.window)


def crossover(a: Node, b: Node, rng: random.Random, max_depth: int) -> Node:
    """把 b 的随机子树接到 a 的随机位置；超深则退回 a。"""
    pa = rng.choice(_all_paths(a))
    pb = rng.choice(_all_paths(b))
    child = _replace(a, pa, _get(b, pb))
    return child if depth(child) <= max_depth else a


def mutate(node: Node, rng: random.Random, max_depth: int) -> Node:
    kind = rng.random()
    paths = _all_paths(node)
    path = rng.choice(paths)
    target = _get(node, path)

    if kind < 0.4:                                   # 子树重生
        sub = random_tree(rng, max_depth=3)
        child = _replace(node, path, sub)
    elif kind < 0.7 and not target.is_leaf:          # 同 arity 换算子
        spec = OPS[target.op]
        same = [n for n, s in OPS.items() if s.arity == spec.arity and s.windowed == spec.windowed and n != target.op]
        if not same:
            return node
        new = Node(rng.choice(same), target.children, target.window)
        child = _replace(node, path, new)
    elif target.window is not None:                  # 窗口扰动
        new = Node(target.op, target.children, rng.choice(WINDOWS))
        child = _replace(node, path, new)
    else:                                            # 叶子换字段
        child = _replace(node, path, random_leaf(rng))

    return child if depth(child) <= max_depth else node
