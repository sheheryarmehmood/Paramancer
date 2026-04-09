from __future__ import annotations

from typing import Callable

import torch

from .types import ScalarLike
from .util import is_flat_raw_var, is_pair_raw_var, is_tensor, unvlatten, vlatten


def to_tuple(data):
    return (data,) if is_tensor(data) else data


def from_tuple(data, was_tensor: bool):
    return data[0] if was_tensor else data


def flat_binary(lhs, rhs, fn):
    lhs_is_tensor = is_tensor(lhs)
    rhs_is_tensor = is_tensor(rhs)
    if lhs_is_tensor != rhs_is_tensor:
        raise TypeError("Flat variables must have matching structures.")

    lhs_items = to_tuple(lhs)
    rhs_items = to_tuple(rhs)
    if len(lhs_items) != len(rhs_items):
        raise ValueError(
            "Flat variables must have the same number of tensors."
        )

    out = tuple(fn(x, y) for x, y in zip(lhs_items, rhs_items))
    return from_tuple(out, lhs_is_tensor)


def flat_unary(data, fn):
    was_tensor = is_tensor(data)
    out = tuple(fn(x) for x in to_tuple(data))
    return from_tuple(out, was_tensor)


class TensorOpsMixin:
    def _map_tensors(self, fn: Callable[[torch.Tensor], torch.Tensor]):
        raise NotImplementedError

    def _iter_tensors(self):
        raise NotImplementedError

    def _new_like(self, data):
        raise NotImplementedError

    def clone(self):
        return self._new_like(self._map_tensors(lambda x: x.clone()))

    def zeros_like(self):
        return self._new_like(self._map_tensors(torch.zeros_like))

    def norm(self) -> torch.Tensor:
        norms = tuple(torch.norm(x) for x in self._iter_tensors())
        return torch.norm(torch.stack(norms))

    def requires_grad_(self, requires_grad: bool):
        return self._new_like(
            self._map_tensors(lambda x: x.requires_grad_(requires_grad))
        )

# TODO: I think the flattening logic can be simplified by just flattening the `FlatVar` type objects here. We can use it to flatten both `FlatVar` and `ParameterBundle` objects. We can also reuse this logic to implement flattening for `PairVar`. Right now, I don't like the idea of two different flattening/unflattening methods namely `platten`/`unplatten` and `vlatten`/`unvlatten`. Maybe we can just have one set of flattening/unflattening functions that work for all variable types. This will also make it easier to implement new variable types in the future without having to worry about implementing separate flattening logic for each type.
class FlattenMixin:
    _flatten_fn = staticmethod(vlatten)
    _unflatten_fn = staticmethod(unvlatten)

    def flatten(self):
        return self._flatten_fn(self.data)

    @classmethod
    def unflatten(cls, flat, spec):
        return cls(cls._unflatten_fn(flat, spec))


def scale_flat(data, scalar: ScalarLike):
    return flat_unary(data, lambda x: x * scalar)
