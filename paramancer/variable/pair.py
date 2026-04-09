from __future__ import annotations

from ._mixins import FlattenMixin, TensorOpsMixin
from .flat import FlatVar
from .types import FlatVarLike
from .util import as_flat_var, as_pair_var, is_pair_raw_var, is_pair_var


class PairVar(TensorOpsMixin, FlattenMixin):
    def __init__(self, *args):
        if len(args) == 1:
            data = args[0]
            if is_pair_var(data):
                self._first = data.first
                self._second = data.second
                return
            if not is_pair_raw_var(data):
                raise TypeError(
                    "Unsupported pair variable. Expected a pair of flat variables."
                )
            first, second = data
        elif len(args) == 2:
            first, second = args
        else:
            raise TypeError("PairVar expects one pair or two flat inputs.")

        self._first: FlatVar = self._to_flat_var(first)
        self._second: FlatVar = self._to_flat_var(second)

    @staticmethod
    def _to_flat_var(data: FlatVarLike) -> FlatVar:
        return as_flat_var(data)

    @property
    def data(self):
        return (self._first.data, self._second.data)

    @property
    def first(self) -> FlatVar:
        return self._first

    @property
    def second(self) -> FlatVar:
        return self._second

    def __getitem__(self, idx: int) -> FlatVar:
        if idx == 0:
            return self._first
        if idx == 1:
            return self._second
        raise IndexError(idx)

    def __iter__(self):
        yield self._first
        yield self._second

    def _map_tensors(self, fn):
        return (self._first._map_tensors(fn), self._second._map_tensors(fn))

    def _iter_tensors(self):
        yield from self._first._iter_tensors()
        yield from self._second._iter_tensors()

    def _new_like(self, data) -> PairVar:
        return type(self)(data)

    def __add__(self, other) -> PairVar:
        other_var = as_pair_var(other)
        return PairVar(
            self._first + other_var.first, self._second + other_var.second
        )

    def __sub__(self, other) -> PairVar:
        other_var = as_pair_var(other)
        return PairVar(
            self._first - other_var.first, self._second - other_var.second
        )

    def __repr__(self):
        return f"PairVar({self._first!r}, {self._second!r})"
