from __future__ import annotations

import torch

from ..variable.flat import FlatVar
from ..variable.types import (
    FlatRawVarType,
    ParamListType,
    PLinOpType,
    is_parameter_type,
)
from ..variable.util import as_flat_var, is_flat_var


def adjoint(lin_op: PLinOpType, zero_el: FlatRawVarType) -> PLinOpType:
    zero_was_var = is_flat_var(zero_el)
    zero_data = zero_el.data if zero_was_var else zero_el

    multi_var_op = isinstance(zero_data, tuple)
    if multi_var_op:
        zero_base = tuple(z.detach().clone() for z in zero_data)
    else:
        zero_base = zero_data.detach().clone()

    def lin_op_adj(y: FlatRawVarType, *args, **kwargs):
        outer_grad_enabled = torch.is_grad_enabled()
        if zero_was_var and not is_flat_var(y):
            raise TypeError("Expected `y` to be FlatVar when zero_el is FlatVar.")
        if not zero_was_var and is_flat_var(y):
            raise TypeError(
                "Expected raw flat variable input when zero_el is raw flat data."
            )

        u_flat: tuple[torch.Tensor, ...] = ()
        if len(args) > 0 and is_parameter_type(args[0]):
            u_flat = (
                tuple(args[0]) if isinstance(args[0], (ParamListType, tuple)) else (args[0],)
            )

        y_data = y.data if is_flat_var(y) else y
        ys = (y_data,) if isinstance(y_data, torch.Tensor) else y_data
        create_graph = outer_grad_enabled and (
            any(inp.requires_grad for inp in ys)
            or any(u_f.requires_grad for u_f in u_flat)
        )

        with torch.enable_grad():
            def func(*zl):
                raw = tuple(zl) if multi_var_op else zl[0]
                return lin_op(raw, *args, **kwargs)

            out = torch.autograd.functional.vjp(
                func, zero_base, y_data, create_graph=create_graph
            )[1]

        return as_flat_var(out) if zero_was_var else out

    return lin_op_adj
