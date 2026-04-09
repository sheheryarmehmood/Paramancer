from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal
import torch

from .types import (
    FlatVarLike,
    PairVarLike,
    FlatParameter,
    FlatRawVarType,
    FlattendType,
    PSpecType,
    PairRawVarType,
    ParameterLike,
    ParameterType,
    PairVarLike,
    TensorSpec,
    TupleSpec,
    PairSpec,
    VSpecType,
)

if TYPE_CHECKING:
    from .flat import FlatVar
    from .pair import PairVar

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

def is_parameter(x: Any) -> bool:
    return isinstance(x, FlatParameter)

def is_collection_of_parameters(x: Any) -> bool:
    return isinstance(x, torch.nn.ParameterList) or (
        isinstance(x, tuple) and all(is_parameter(p) for p in x)
    )

def is_valid_parameter(x: Any) -> bool:
    return is_parameter(x) or is_collection_of_parameters(x)

def as_flat_var(data: FlatVarLike) -> FlatVar:
    from .flat import FlatVar

    return data if is_flat_var(data) else FlatVar(data)

def as_pair_var(data: PairVarLike) -> PairVar:
    from .pair import PairVar

    return data if is_pair_var(data) else PairVar(data)

def flatten_flat_raw(data: FlatRawVarType) -> tuple[FlattendType, TensorSpec | TupleSpec]:
    if is_tensor(data):
        return (data,), ("tensor",)
    if is_flat_raw_var(data):
        return data, ("tuple", len(data))
    raise TypeError(
        "Unsupported flat raw variable. Expected Tensor or tuple[Tensor, ...]."
    )


def flatten_pair_raw(data: PairRawVarType) -> tuple[FlattendType, PairSpec]:
    if not is_pair_raw_var(data):
        raise TypeError(
            "Unsupported pair raw variable. Expected pair of flat raw variables."
        )
    first_flat, first_spec = flatten_flat_raw(data[0])
    second_flat, second_spec = flatten_flat_raw(data[1])
    return (*first_flat, *second_flat), ("pair", len(first_flat), len(second_flat))


def vlatten(data) -> tuple[FlattendType, VSpecType]:
    if is_flat_raw_var(data):
        return flatten_flat_raw(data)
    if is_pair_raw_var(data):
        return flatten_pair_raw(data)
    raise TypeError(
        "Unsupported raw variable. Expected Tensor, tuple[Tensor, ...], or "
        "pair of flat raw variables."
    )


def unflatten_flat_raw(flat: FlattendType, spec: TensorSpec | TupleSpec) -> FlatRawVarType:
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
            raise ValueError(f"Spec {spec} expects {n} tensors, got {len(flat)}.")
        return tuple(flat)
    raise TypeError(f"Unsupported flat spec {spec}.")


def unflatten_pair_raw(flat: FlattendType, spec: PairSpec) -> PairRawVarType:
    if spec[0] != "pair":
        raise TypeError(f"Unsupported pair spec {spec}.")
    n_first, n_second = spec[1], spec[2]
    if len(flat) != n_first + n_second:
        raise ValueError(
            f"Spec {spec} expects {n_first + n_second} tensors, got {len(flat)}."
        )
    first = tuple(flat[:n_first])
    second = tuple(flat[n_first : n_first + n_second])
    first_raw = unflatten_flat_raw(first, ("tensor",) if n_first == 1 else ("tuple", n_first))
    second_raw = unflatten_flat_raw(second, ("tensor",) if n_second == 1 else ("tuple", n_second))
    return first_raw, second_raw


def unvlatten(flat: FlattendType, spec: VSpecType):
    if spec[0] in {"tensor", "tuple"}:
        return unflatten_flat_raw(flat, spec)
    if spec[0] == "pair":
        return unflatten_pair_raw(flat, spec)
    raise TypeError(f"Unsupported variable spec {spec}.")


def platten(par: ParameterType) -> tuple[FlattendType, PSpecType]:
    if is_parameter(par):
        return (par,), ("tensor",)
    if is_collection_of_parameters(par):
        return tuple(par), ("tuple", len(par))
    raise TypeError(
        "Unsupported ParameterType. Expected torch.Tensor, tuple[torch.Tensor, ...] "
        "or torch.nn.ParameterList."
    )


def unplatten(flat: FlattendType, spec: PSpecType) -> ParameterType:
    if not is_collection_of_parameters(flat):
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
            raise ValueError(f"Spec {spec} expects {n} tensors, got {len(flat)}.")
        return tuple(flat)
    raise TypeError(f"Unsupported parameter spec {spec}.")


def zeros_like(
    v: PairVarLike | ParameterLike | FlatRawVarType,
    typ: Literal["variable"] | Literal["parameter"] = "variable",
):
    from .flat import FlatVar
    from .pair import PairVar
    from .parameter import ParameterBundle

    if typ == "variable":
        input_is_wrapper = is_flat_var(v) or is_pair_var(v)
        wrapper = (
            v
            if input_is_wrapper
            else (as_pair_var(v) if is_pair_raw_var(v) else as_flat_var(v))
        )
    elif typ == "parameter":
        input_is_wrapper = isinstance(v, ParameterBundle)
        wrapper = v if input_is_wrapper else ParameterBundle(v)
    else:
        raise ValueError("`typ` must be either 'variable' or 'parameter'.")

    zero = wrapper.zeros_like()
    return zero if input_is_wrapper else zero.data
