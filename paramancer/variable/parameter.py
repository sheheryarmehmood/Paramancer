from __future__ import annotations

from ._mixins import FlattenMixin, TensorOpsMixin
from .types import IndexMapType, RawParamType


class ParamBundle(TensorOpsMixin, FlattenMixin):
    def __init__(
        self,
        data: RawParamType,
        indices: IndexMapType | None = None,
    ) -> None:
        self._data = data
        self._indices = indices or {}
        if not self._indices:
            self._indices.setdefault("grad", "all")
            self._indices.setdefault("lin_op", "all")

    @property
    def data(self) -> RawParamType:
        return self._data

    @property
    def indices(self) -> IndexMapType:
        return self._indices

    @data.setter
    def data(self, new_data: RawParamType):
        self._data = new_data

    def takes_params(self, key: str) -> bool:
        return key in self._indices and (
            bool(self._indices[key]) or self._indices[key] == 0
        )

    def select(
        self,
        key: str,
    ) -> RawParamType:
        if not self.takes_params(key):
            return ()

        idx = self._indices[key]
        if idx == "all":
            return self._data

        if isinstance(idx, int):
            return self._data[idx]

        return tuple(self._data[i] for i in idx)

    @property
    def grad(self) -> RawParamType:
        return self.select("grad")

    @property
    def prox(self) -> RawParamType:
        return self.select("prox")

    @property
    def lin_op(self) -> RawParamType:
        return self.select("lin_op")

    @property
    def primal(self) -> RawParamType:
        return self.select("primal")

    @property
    def dual(self) -> RawParamType:
        return self.select("dual")

    def _map_tensors(self, fn):
        flat, spec = self.flatten()
        return self._unflatten_fn(tuple(fn(u) for u in flat), spec)

    def _iter_tensors(self):
        flat, _ = self.flatten()
        return iter(flat)

    def _new_like(self, data):
        return type(self)(data, self.indices.copy())


AlgoParam = ParamBundle
