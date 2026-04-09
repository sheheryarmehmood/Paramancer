from __future__ import annotations

from .flat import FlatVar
from .pair import PairVar

# Temporary import compatibility only. The implementation lives in flat.py/pair.py.
Variable = FlatVar
AlgoVar = PairVar

__all__ = ["FlatVar", "PairVar", "Variable", "AlgoVar"]
