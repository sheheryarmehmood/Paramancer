from __future__ import annotations
import torch
from functools import wraps

from .types import (
    VariableType, VariableLike, ScalarLike, ApplyType, SpecType,
    IndexMapType, FlatParameter, TupleParameter, ParameterType, ParameterLike
)

class ParameterBundle:
    def __init__(
        self,
        data: ParameterType,
        indices: IndexMapType | None = None,
    ) -> None:
        self._data = data
        self._indices = indices or {}
        self._indices.setdefault("grad", None)
        self._indices.setdefault("lin_op", None)

    @property
    def data(self) -> ParameterType:
        return self._data

    def select(
        self,
        key: str,
    ) -> FlatParameter | TupleParameter:
        if key not in self._indices:
            return ()

        idx = self._indices[key]

        if isinstance(self._data, FlatParameter):
            if idx is None:
                return self._data
            raise ValueError(
                f"Key '{key}' cannot index a single torch.nn.Parameter."
            )

        if isinstance(idx, int):
            return self._data[idx]

        return tuple(self._data[i] for i in idx)

    @property
    def grad(self) -> FlatParameter | TupleParameter:
        return self.select("grad")

    @property
    def prox(self) -> FlatParameter | TupleParameter:
        return self.select("prox")
    
    @property
    def lin_op(self) -> FlatParameter | TupleParameter:
        return self.select("lin_op")

    @property
    def primal(self) -> FlatParameter | TupleParameter:
        return self.select("primal")
    
    @property
    def dual(self) -> FlatParameter | TupleParameter:
        return self.select("dual")
