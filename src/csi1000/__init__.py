# -*- coding: utf-8 -*-
"""中证1000 时序择时(v4 定稿):GA 因子挖矿 + walk-forward 回测。"""
__version__ = "4.0.0"

# ---- 旧包名别名 ----------------------------------------------------------
# results/ops26/state.pkl 是重构前用 `ga_alpha.*` / `walkforward.*` 顶层包名
# pickle 的因子表达式树。反序列化按存档时的模块路径找类,故把老顶层名映射到
# 迁移后的 csi1000.* 子包。新代码一律用 csi1000.* 全名,这里只为读旧存档兜底。
import importlib as _il
import sys as _sys

for _old, _new in (("ga_alpha", "csi1000.ga_alpha"),
                   ("walkforward", "csi1000.walkforward")):
    if _old not in _sys.modules:
        _sys.modules[_old] = _il.import_module(_new)
for _sub in ("expr", "ops", "fitness", "evolve"):
    _sys.modules.setdefault(f"ga_alpha.{_sub}", _il.import_module(f"csi1000.ga_alpha.{_sub}"))
for _sub in ("config", "data", "elders", "engine", "ga_setup", "metrics"):
    _sys.modules.setdefault(f"walkforward.{_sub}", _il.import_module(f"csi1000.walkforward.{_sub}"))
