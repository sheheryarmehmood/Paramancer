from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch

from .types import (
    FlatRawVarType,
    FlatSpecType,
    FlatVarLike,
    FlattendType,
    PairRawVarType,
    PairSpec,
    PairVarLike,
    ParamBundleLike,
    RawParamType,
    VSpecType,
)

if TYPE_CHECKING:
    from .flat import FlatVar
    from .pair import PairVar
    from .parameter import ParamBundle


def is_tensor(x: Any) -> bool:
    return isinstance(x, torch.Tensor)


def is_flat_raw_var(x: Any) -> bool:
    return is_tensor(x) or (
        isinstance(x, tuple) and all(is_tensor(item) for item in x)
    )


def is_pair_raw_var(x: Any) -> bool:
    return (
        isinstance(x, tuple)
        and len(x) == 2
        and is_flat_raw_var(x[0])
        and is_flat_raw_var(x[1])
    )


def is_raw_var(x: Any) -> bool:
    return is_flat_raw_var(x) or is_pair_raw_var(x)


def is_flat_var(x: Any) -> bool:
    from .flat import FlatVar

    return isinstance(x, FlatVar)


def is_pair_var(x: Any) -> bool:
    from .pair import PairVar

    return isinstance(x, PairVar)


def is_param_bundle(x: Any) -> bool:
    from .parameter import ParamBundle

    return isinstance(x, ParamBundle)


def is_collection_of_parameters(x: Any) -> bool:
    return isinstance(x, torch.nn.ParameterList) or (
        isinstance(x, tuple) and all(is_tensor(p) for p in x)
    )


def is_valid_parameter(x: Any) -> bool:
    return is_tensor(x) or is_collection_of_parameters(x)


def as_flat_var(data: FlatVarLike) -> FlatVar:
    from .flat import FlatVar

    return data if is_flat_var(data) else FlatVar(data)


def as_pair_var(data: PairVarLike) -> PairVar:
    from .pair import PairVar

    return data if is_pair_var(data) else PairVar(data)


def flatten_flat_raw(
    data: FlatRawVarType | RawParamType
) -> tuple[FlattendType, FlatSpecType]:
    if is_tensor(data):
        return (data,), ("tensor",)
    if is_flat_raw_var(data) or is_collection_of_parameters(data):
        return tuple(data), ("tuple", len(data))
    raise TypeError(
        "Unsupported flat raw payload. Expected Tensor, tuple[Tensor, ...], "
        "or torch.nn.ParameterList."
    )


def flatten_pair_raw(data: PairRawVarType) -> tuple[FlattendType, PairSpec]:
    if not is_pair_raw_var(data):
        raise TypeError(
            "Unsupported pair raw variable. Expected pair of flat raw variables."
        )
    first_flat, _ = flatten_flat_raw(data[0])
    second_flat, _ = flatten_flat_raw(data[1])
    return (*first_flat, *second_flat), ("pair", len(first_flat), len(second_flat))


def flatten_raw(data) -> tuple[FlattendType, VSpecType]:
    if is_pair_raw_var(data):
        return flatten_pair_raw(data)
    return flatten_flat_raw(data)


def unflatten_flat_raw(
    flat: FlattendType, spec: FlatSpecType
) -> FlatRawVarType | RawParamType:
    if not isinstance(flat, tuple) or not all(is_tensor(item) for item in flat):
        raise TypeError("`flat` must be a tuple of torch.Tensor.")
    if spec[0] == "tensor":
        if len(flat) != 1:
            raise ValueError(
                f"Spec {spec} expects exactly 1 tensor, got {len(flat)}."
            )
        return flat[0]
    if spec[0] == "tuple":
        n = spec[1]
        if len(flat) != n:
            raise ValueError(
                f"Spec {spec} expects {n} tensors, got {len(flat)}."
            )
        return tuple(flat)
    raise TypeError(f"Unsupported flat spec {spec}.")


def unflatten_pair_raw(flat: FlattendType, spec: PairSpec) -> PairRawVarType:
    if spec[0] != "pair":
        raise TypeError(f"Unsupported pair spec {spec}.")
    n_first, n_second = spec[1], spec[2]
    if len(flat) != n_first + n_second:
        raise ValueError(
            f"Spec {spec} expects {n_first + n_second} tensors, got "
            f"{len(flat)}."
        )
    first = tuple(flat[:n_first])
    second = tuple(flat[n_first : n_first + n_second])
    first_raw = unflatten_flat_raw(
        first, ("tensor",) if n_first == 1 else ("tuple", n_first)
    )
    second_raw = unflatten_flat_raw(
        second, ("tensor",) if n_second == 1 else ("tuple", n_second)
    )
    return first_raw, second_raw


def unflatten_raw(flat: FlattendType, spec: VSpecType):
    if spec[0] == "pair":
        return unflatten_pair_raw(flat, spec)
    return unflatten_flat_raw(flat, spec)

def map_flat_raw(
    v: FlatRawVarType | RawParamType,
    fn,
) -> FlatRawVarType | RawParamType:
    flat, spec = flatten_flat_raw(v)
    return unflatten_flat_raw(tuple(fn(item) for item in flat), spec)


def map_pair_raw(
    v: PairRawVarType,
    flat_fn,
) -> PairRawVarType:
    if not is_pair_raw_var(v):
        raise TypeError(
            "Unsupported pair raw variable. Expected pair of flat raw "
            "variables."
        )
    return (
        map_flat_raw(v[0], flat_fn),
        map_flat_raw(v[1], flat_fn),
    )


def zeros_like_flat_raw(
    v: FlatRawVarType | RawParamType
) -> FlatRawVarType | RawParamType:
    return map_flat_raw(v, torch.zeros_like)


def zeros_like_pair_raw(v: PairRawVarType) -> PairRawVarType:
    return map_pair_raw(v, torch.zeros_like)


def map_raw(
    v: FlatRawVarType | RawParamType | PairRawVarType,
    flat_fn,
    pair_fn,
):
    if is_pair_raw_var(v):
        return pair_fn(v)
    if is_flat_raw_var(v) or is_valid_parameter(v):
        return flat_fn(v)
    raise TypeError(
        "Unsupported raw payload. Expected flat raw tensors, a parameter "
        "payload, or a pair of flat raw variables."
    )


def zeros_like_raw(v: FlatRawVarType | RawParamType | PairRawVarType):
    return map_raw(v, zeros_like_flat_raw, zeros_like_pair_raw)

def map_wrapper(v, method_name: str, raw_fn, *args, **kwargs):
    if is_flat_var(v) or is_pair_var(v) or is_param_bundle(v):
        return getattr(v, method_name)(*args, **kwargs)
    return raw_fn(v, *args, **kwargs)


def zeros_like(v: PairVarLike | ParamBundleLike | FlatRawVarType):
    return map_wrapper(v, "zeros_like", zeros_like_raw)


def clone_flat_raw(
    v: FlatRawVarType | RawParamType
) -> FlatRawVarType | RawParamType:
    return map_flat_raw(v, torch.Tensor.clone)


def clone_pair_raw(v: PairRawVarType) -> PairRawVarType:
    return map_pair_raw(v, torch.Tensor.clone)


def clone_raw(v: FlatRawVarType | RawParamType | PairRawVarType):
    return map_raw(v, clone_flat_raw, clone_pair_raw)


def clone(v: PairVarLike | ParamBundleLike | FlatRawVarType):
    return map_wrapper(v, "clone", clone_raw)


def requires_grad_flat_raw_(
    v: FlatRawVarType | RawParamType, requires_grad: bool
) -> FlatRawVarType | RawParamType:
    return map_flat_raw(
        v, lambda item: item.requires_grad_(requires_grad)
    )


def requires_grad_pair_raw_(
    v: PairRawVarType, requires_grad: bool
) -> PairRawVarType:
    return map_pair_raw(
        v, lambda item: item.requires_grad_(requires_grad)
    )


def requires_grad_raw_(
    v: FlatRawVarType | RawParamType | PairRawVarType, requires_grad: bool
):
    return map_raw(
        v,
        lambda raw: requires_grad_flat_raw_(raw, requires_grad),
        lambda raw: requires_grad_pair_raw_(raw, requires_grad),
    )


def requires_grad_(
    v: PairVarLike | ParamBundleLike | FlatRawVarType, requires_grad: bool
):
    return map_wrapper(
        v,
        "requires_grad_",
        requires_grad_raw_,
        requires_grad,
    )
