from __future__ import annotations
import torch

from ..variable import Variable
from ..variable.types import (
    TupleVariable, BaseVariableType,
    ParameterList, is_parameter_type,
    PLinOpType
)


def adjoint(
    lin_op: PLinOpType, zero_el: BaseVariableType
) -> PLinOpType:
    """Returns the adjoint (transpose) of a given linear map.

    Given a linear map `lin_op` and a zero element from its input space,
    this function constructs a callable that computes the adjoint map
    using PyTorch's vector-Jacobian product (VJP).

    The zero element is used to determine the structure (type, shape, device)
    of the adjoint's domain and codomain.

    Args:
        lin_op (Callable): Linear map to compute the adjoint of. Must be 
            differentiable with respect to its first argument(s).
        zero_el (torch.Tensor or tuple of torch.Tensor): Zero element of the 
            input space of `lin_op`. For tuple/list inputs, each element must 
            have the same structure as the inputs to `lin_op`.

    Returns:
        Callable: Function `lin_op_adj(y, *params)` that computes the adjoint 
        map applied to `y`. Preserves gradient tracking if `y` or `params`
        require gradients.

    Notes:
        - The returned adjoint is defined with respect to the standard 
          Euclidean inner product on the involved tensor spaces.
        - Handles both single-tensor and tuple/list-tensor input spaces.
        - The type of `zero_el` determines the input/output type:
          if `zero_el` is `Variable`, then `lin_op` and `lin_op_adj` should
          receive `Variable` and `lin_op_adj` returns `Variable`. Otherwise
          they use `BaseVariableType`.
        - The adjoint is differentiable only when:
          - `lin_op_adj` is called with grad enabled (i.e., not inside
            `with torch.no_grad(): ...`), and
          - `y` (or any element of `args[0]` when it is a `ParameterType`)
            has `requires_grad=True`.

    Example:
        >>> import torch
        >>> A = torch.tensor([[1., 2.],
        ...                   [3., 4.]])
        >>> lin_op = lambda x: A @ x  # Linear map: R^2 -> R^2
        >>> zero_el = torch.zeros(2)  # Zero element in input space
        >>> lin_op_adj = adjoint(lin_op, zero_el)
        >>> y = torch.tensor([1., 1.])
        >>> lin_op_adj(y)  # Equivalent to A.T @ y
        tensor([4., 6.])
    """
    zero_was_var = isinstance(zero_el, Variable)
    zero_data = zero_el.data if zero_was_var else zero_el

    multi_var_op = isinstance(zero_data, (tuple, list))
    if multi_var_op:
        typ = type(zero_data)
        zero_el = typ([z.detach().clone() for z in zero_data])
    else:
        zero_el = zero_data.detach().clone()

    def lin_op_adj(y: BaseVariableType, *args, **kwargs):
        outer_grad_enabled = torch.is_grad_enabled()
        if zero_was_var and not isinstance(y, Variable):
            raise TypeError("Expected `y` to be Variable when zero_el is Variable.")
        if not zero_was_var and isinstance(y, Variable):
            raise TypeError("Expected `y` to be BaseVariableType when zero_el is not Variable.")

        u_flat: tuple[torch.Tensor, ...] = ()
        if len(args) > 0 and is_parameter_type(args[0]):
            u_flat = (
                tuple(args[0])
                if isinstance(args[0], (ParameterList, tuple))
                else (args[0],)
            )
        y_data = y.data if isinstance(y, Variable) else y
        ys = (y_data,) if isinstance(y_data, torch.Tensor) else y_data
        
        create_graph = outer_grad_enabled and (
            any(inp.requires_grad for inp in ys) or
            any(u_f.requires_grad for u_f in u_flat)
        )
        
        with torch.enable_grad():
            def func(*zl):
                return lin_op(zl if multi_var_op else zl[0], *args, **kwargs)
            out = torch.autograd.functional.vjp(
                func, zero_el, y_data, create_graph=create_graph
            )[1]
        
        return Variable(out) if zero_was_var else out
    return lin_op_adj
