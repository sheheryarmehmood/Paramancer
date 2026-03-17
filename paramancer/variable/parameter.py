from __future__ import annotations

from .types import (
    IndexMapType, ParameterType
)

class ParameterBundle:
    def __init__(
        self,
        data: ParameterType,
        indices: IndexMapType | None = None,
    ) -> None:
        self._data = data
        self._indices = indices or {}
        if not self._indices:
            self._indices.setdefault("grad", "all")
            self._indices.setdefault("lin_op", "all")

    @property
    def data(self) -> ParameterType:
        return self._data
    
    @data.setter
    def data(self, new_data: ParameterType):
        self._data = new_data
    
    def takes_params(self, key: str) -> bool:
        return key in self._indices and ( # No key -> no params.
            bool(self._indices[key]) or # Empty index -> no params.
            self._indices[key] == 0 # Index 0 is falsy but valid. 
        )

    def select(
        self,
        key: str,
    ) -> ParameterType:
        if not self.takes_params(key):
            return ()

        idx = self._indices[key]
        if idx == "all":        # Only way to access a `FlatParameter` `_data`.
            return self._data

        if isinstance(idx, int):
            return self._data[idx]

        return tuple(self._data[i] for i in idx)

    @property
    def grad(self) -> ParameterType:
        return self.select("grad")

    @property
    def prox(self) -> ParameterType:
        return self.select("prox")
    
    @property
    def lin_op(self) -> ParameterType:
        return self.select("lin_op")

    @property
    def primal(self) -> ParameterType:
        return self.select("primal")
    
    @property
    def dual(self) -> ParameterType:
        return self.select("dual")
