"""GA 主循环：锦标赛选择 + 子树交叉 + 变异 + 精英保留；HOF 负责去相关入库。"""
from __future__ import annotations

import random

import numpy as np
import pandas as pd

from csi1000.ga_alpha.expr import Node, crossover, mutate, random_tree
from csi1000.ga_alpha.fitness import Evaluator, Individual


class HallOfFame:
    """入选条件：valid 段 |IC| 达标、方向与 train 一致、与已入选因子相关性不超上限。"""

    def __init__(self, evaluator: Evaluator, cfg: dict):
        self.ev = evaluator
        self.size = cfg["hall_of_fame"]["size"]
        self.min_train_ic = cfg["hall_of_fame"].get("min_train_ic", 0.0)
        self.min_valid_ic = cfg["hall_of_fame"]["min_valid_ic"]
        self.max_corr = cfg["hall_of_fame"]["max_corr"]
        self.entries: list[dict] = []

    def _corr_ok(self, factor: pd.DataFrame, exclude: dict | None = None) -> bool:
        cand = factor[self.ev.valid_mask].stack()
        for e in self.entries:
            if e is exclude:
                continue
            joined = pd.concat([cand, e["_valid_values"]], axis=1, join="inner").dropna()
            if len(joined) > 10 and abs(joined.iloc[:, 0].corr(joined.iloc[:, 1], method="spearman")) > self.max_corr:
                return False
        return True

    def consider(self, ind: Individual) -> bool:
        if ind.factor is None or abs(ind.train_ic) < self.min_train_ic:
            return False
        if any(e["expr"] == ind.key for e in self.entries):
            return False
        # 满员时按 fitness 挤掉最弱的一个；打不过最弱者直接放弃
        weakest = None
        if len(self.entries) >= self.size:
            weakest = min(self.entries, key=lambda e: e["fitness"])
            if ind.fitness <= weakest["fitness"]:
                return False
        v_ic = self.ev.valid_ic(ind)
        if not np.isfinite(v_ic) or abs(v_ic) < self.min_valid_ic or np.sign(v_ic) != np.sign(ind.train_ic):
            return False
        if not self._corr_ok(ind.factor, exclude=weakest):
            return False
        if weakest is not None:
            self.entries.remove(weakest)
        self.entries.append({
            "expr": ind.key,
            "train_ic": ind.train_ic,
            "valid_ic": v_ic,
            "fitness": ind.fitness,
            "_tree": ind.tree,
            "_valid_values": ind.factor[self.ev.valid_mask].stack(),
            "_factor": ind.factor,
        })
        return True


def _tournament(pop: list[Individual], k: int, rng: random.Random) -> Individual:
    return max(rng.sample(pop, k), key=lambda i: i.fitness)


def run_ga(panels: dict[str, pd.DataFrame], cfg: dict, log=print) -> HallOfFame:
    g = cfg["ga"]
    rng = random.Random(g["seed"])
    ev = Evaluator(panels, cfg)
    hof = HallOfFame(ev, cfg)
    cache: dict[str, Individual] = {}

    def eval_pop(trees: list[Node]) -> list[Individual]:
        out = []
        for t in trees:
            key = str(t)
            if key not in cache:
                cache[key] = ev.evaluate(Individual(t))
            out.append(cache[key])
        return out

    # ramped half-and-half 初始化
    trees = []
    for i in range(g["population"]):
        d = g["init_depth_min"] + i % (g["init_depth_max"] - g["init_depth_min"] + 1)
        trees.append(random_tree(rng, d, full=(i % 2 == 0)))
    pop = eval_pop(trees)

    for gen in range(1, g["generations"] + 1):
        pop.sort(key=lambda i: i.fitness, reverse=True)
        for ind in pop:
            if ind.fitness > -np.inf:
                hof.consider(ind)

        best = pop[0]
        log(f"[gen {gen:>3}] best_fitness={best.fitness:+.4f} train_ic={best.train_ic:+.4f} "
            f"hof={len(hof.entries)}  {best.key[:100]}")

        next_trees = [ind.tree for ind in pop[:g["elitism"]]]
        while len(next_trees) < g["population"]:
            p1 = _tournament(pop, g["tournament_k"], rng)
            r = rng.random()
            if r < g["p_crossover"]:
                p2 = _tournament(pop, g["tournament_k"], rng)
                child = crossover(p1.tree, p2.tree, rng, g["max_depth"])
            elif r < g["p_crossover"] + g["p_mutation"]:
                child = mutate(p1.tree, rng, g["max_depth"])
            else:
                child = p1.tree
            next_trees.append(child)
        pop = eval_pop(next_trees)

    pop.sort(key=lambda i: i.fitness, reverse=True)
    for ind in pop:
        if ind.fitness > -np.inf:
            hof.consider(ind)
    return hof
