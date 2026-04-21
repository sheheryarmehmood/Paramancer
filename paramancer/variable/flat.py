from __future__ import annotations

from ._mixins import (
    FlattenMixin,
    TensorOpsMixin,
    flat_binary,
    flat_unary,
    scale_flat,
    to_tuple,
)
from .types import FlatVarLike, FlatRawVarType, ScalarLike
from .util import as_flat_var, is_flat_raw_var, is_flat_var


class FlatVar(TensorOpsMixin, FlattenMixin):
    def __init__(self, data: FlatVarLike):
        if is_flat_var(data):
            self._data = data.data
            return
        if not is_flat_raw_var(data):
            raise TypeError(
                "Unsupported flat variable. Expected Tensor or "
                "tuple[Tensor, ...]."
            )
        self._data = data

    @property
    def data(self) -> FlatRawVarType:
        return self._data

    @classmethod
    def wrap(cls, data: FlatVarLike) -> FlatVar:
        return cls(data)

    def _map_tensors(self, fn):
        return flat_unary(self._data, fn)

    def _iter_tensors(self):
        return iter(to_tuple(self._data))

    def _new_like(self, data) -> FlatVar:
        return type(self)(data)

    def __add__(self, other: FlatVarLike) -> FlatVar:
        other_var = as_flat_var(other)
        return FlatVar(
            flat_binary(self._data, other_var.data, lambda x, y: x + y)
        )

    def __sub__(self, other: FlatVarLike) -> FlatVar:
        other_var = as_flat_var(other)
        return FlatVar(
            flat_binary(self._data, other_var.data, lambda x, y: x - y)
        )

    def __neg__(self) -> FlatVar:
        return FlatVar(flat_unary(self._data, lambda x: -x))

    def __mul__(self, scalar: ScalarLike) -> FlatVar:
        return FlatVar(scale_flat(self._data, scalar))

    __rmul__ = __mul__

    def __repr__(self):
        return f"FlatVar({self._data})"
