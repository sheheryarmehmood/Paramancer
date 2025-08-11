import torch
from typing import Callable, Tuple, Union



def adjoint(
    lin_op: Callable, zero_el: Union[torch.Tensor, Tuple[torch.Tensor]]
) -> Callable:
    """Returns the adjoint (transpose) of a given linear map.

    Given a linear map `lin_op` and a zero element from its input space,
    this function constructs a callable that computes the adjoint map
    using PyTorch's vector–Jacobian product (VJP).

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
    multi_var_op = isinstance(zero_el, (tuple, list))
    if multi_var_op:
        typ = type(zero_el)
        zero_el = typ([z.clone().detach() for z in zero_el])
    else:
        zero_el = zero_el.clone().detach()
    def lin_op_adj(inps, *params):
        inputs = inps + params if isinstance(inps, tuple) else (inps, *params)
        create_graph = True in [inp.requires_grad for inp in inputs]
        return torch.autograd.functional.vjp(
            lambda *zl: lin_op(*zl, *params), zero_el, inps,
            create_graph=create_graph
        )[1]
    return lin_op_adj