from __future__ import annotations
from typing import Any, Literal
import torch

from .types import (
    VariableType, ParameterType, VariableLike, ParameterLike, FlatParameter,
    FlattendType, VSpecType, PSpecType
)


def is_tensor(x: Any) -> bool:
    return isinstance(x, torch.Tensor)

def is_tuple_of_tensors(x: Any) -> bool:
    return isinstance(x, tuple) and all(is_tensor(t) for t in x)

def is_nested_variable(x: Any) -> bool:
    # exactly: ( (tensors...), (tensors...) )
    return (
        isinstance(x, tuple)
        and len(x) == 2
        and is_tuple_of_tensors(x[0])
        and is_tuple_of_tensors(x[1])
    )

def is_parameter(x: Any) -> bool:
    return isinstance(x, FlatParameter)

def is_collection_of_parameters(x: Any) -> bool:
    return (
        isinstance(x, torch.nn.ParameterList) or
        isinstance(x, tuple) and all(is_parameter(p) for p in x)
    )

def is_valid_variable(x: Any) -> bool:
    return is_tensor(x) or is_tuple_of_tensors(x) or is_nested_variable(x)

def is_valid_parameter(x: Any) -> bool:
    return is_parameter(x) or is_collection_of_parameters(x)

# TODO: Perhaps, I should come up with unified `flatten` and `unflatten` 
# methods which apply to both `VariableType` and `ParameterType`. One way to
# do that is to implement one base class which handles all the arithmetic and
# does not differentiate between if it is a lower or an upper variable. It 
# also has methods to `flatten` and `unflatten` its data. Then I inherit both
# `Variable` and `ParameterBundle` from it.
def vlatten(data: VariableType) -> tuple[FlattendType, VSpecType]:
    """
    Flattens `VariableType` into a flat tuple of tensors plus a spec to 
    reconstruct it.

    Spec encodings:
    - `("flat",)`
    - `("tuple", n)`
    - `("nested", n_left, n_right)`
    """
    if is_tensor(data):
        return (data,), ("flat",)
    
    if is_tuple_of_tensors(data):
        return data, ("tuple", len(data))
    
    if is_nested_variable(data):
        left, right = data
        flat = (*left, *right)
        return flat, ("nested", len(left), len(right))
    
    raise TypeError(
        "Unsupported `VariableType`. Expected `Tensor`, or `tuple[Tensor,...]`"
        " or `tuple[tuple[Tensor,...], tuple[Tensor,...]]`."
    )

def unvlatten(flat: FlattendType, spec: VSpecType) -> VariableType:
    """
    Reconstructs `VariableType` from a flat tuple of tensors and a spec 
    produced by `vlatten`.
    """
    if not is_tuple_of_tensors(flat):
        raise TypeError("`flat` must be a tuple of torch.Tensor.")
    
    if not isinstance(spec, tuple) or len(spec) < 1:
        raise TypeError("`spec` must be a non-empty tuple.")
    
    tag = spec[0]
    if tag == "flat":
        if len(flat) != 1:
            raise ValueError(
                f"Spec {spec} expects exactly 1 tensor, got {len(flat)}."
            )
        return flat[0]
    
    if tag == "tuple":
        if len(spec) != 2 or not isinstance(spec[1], int):
            raise TypeError("Spec ('tuple', n) must have an int n.")
        n = spec[1]
        if len(flat) != n:
            raise ValueError(
                f"Spec {spec} expects {n} tensors, got {len(flat)}."
            )
        return tuple(flat)
    
    if tag == "nested":
        if (
            len(spec) != 3 or not isinstance(spec[1], int) or 
            not isinstance(spec[2], int)
        ):
            raise TypeError(
                "Spec ('nested', n_left, n_right) must have two ints."
            )
        nL, nR = spec[1], spec[2]
        if len(flat) != nL + nR:
            raise ValueError(
                f"Spec {spec} expects {nL+nR} tensors, got {len(flat)}."
            )
        left = tuple(flat[:nL])
        right = tuple(flat[nL:nL + nR])
        return (left, right)


def platten(par: ParameterType) -> tuple[FlattendType, PSpecType]:
    if is_parameter(par):
        return (par,), ("flat",)
    if is_collection_of_parameters(par):
        # vvvv set spec to 'tuple' for both tuple and ParameterList.
        return par, ("tuple", len(par))
    raise TypeError(
        "Unsupported ParameterType. Expected `torch.Tensor`, "
        "`tuple[torch.Tensor,...]` or `torch.nn.ParameterList`."
    )

def unplatten(flat: FlattendType, spec: PSpecType) -> ParameterType:
    """
    Reconstructs ParameterType from a flat tuple of Parameters and a spec 
    produced by `platten`.
    """
    if not is_collection_of_parameters(flat):
        raise TypeError("`flat` must be a tuple of torch.Tensor.")
    
    if not isinstance(spec, tuple) or len(spec) < 1:
        raise TypeError("`spec` must be a non-empty tuple.")
    
    tag = spec[0]
    if tag == "flat":
        if len(flat) != 1:
            raise ValueError(
                f"Spec {spec} expects exactly 1 tensor, got {len(flat)}."
            )
        return flat[0]
    
    if tag == "tuple":
        if not isinstance(spec[1], int):
            raise TypeError("Spec ('tuple', n) must have an int n.")
        n = spec[1]
        if len(flat) != n:
            raise ValueError(
                f"Spec {spec} expects {n} tensors, got {len(flat)}."
            )
        return tuple(flat)

# Maybe move it to the new proposed base class or wrap it with the 
# `ensure_raw_input` method.
def zeros_like(
    v: VariableLike | ParameterLike,
    typ: Literal["variable"] | Literal["parameter"] = "variable"
):
    from .variable import Variable
    from .parameter import ParameterBundle
    
    if typ == "variable":
        flatten, unflatten = vlatten, unvlatten
        Type = Variable
    elif typ == "parameter":
        flatten, unflatten = platten, unplatten
        Type = ParameterBundle
    else:
        raise ValueError(
            "`typ` must be either 'variable' or 'parameter'."
        )
    v_is_typ = isinstance(v, Type)
    v_typ = v if v_is_typ else Type(v)
    v_zero_typ = v_typ.zeros_like()
    return v_zero_typ if v_is_typ else v_zero_typ.data
