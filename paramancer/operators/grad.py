from __future__ import annotations
import torch
from collections.abc import Callable
from typing import Any

from ..variable import Variable, flatten, unflatten
from ..variable.types import (
    BaseVariableType, TupleParameter, ParameterType,
    PSmoothObjType, PGradMapType
)


# Decide whether to build a higher-order graph
# (If any input requires grad, we want the gradient itself to be
# differentiable.)
# def _req_grad(obj: Any) -> bool:
#     return isinstance(obj, torch.Tensor) and obj.requires_grad

def _set_req_grad(
    vars: tuple[torch.Tensor, ...], rgs: bool | tuple[bool, ...]
):
    if isinstance(rgs, bool):
        rgs = (rgs,) * len(vars)
    for var, rg in zip(vars, rgs):
        var.requires_grad_(rg)

def gradient(smooth: PSmoothObjType) -> PGradMapType:
    """
    Return a callable that differentiates `smooth` with respect to its first 
    argument `x`.

    Expected objective signature:
        smooth(x, *args, **kwargs) -> torch.Tensor

    The returned function has signature:
        grad(x, *args, **kwargs) -> grad_x

    Notes:
    - `x` may be a `Variable`, a Tensor, or a tuple of Tensors
      (BaseVariableLike).
    - Gradients are computed w.r.t. `x` only.
    - When `len(args) > 0`, `args[0]` can optionally be of `ParameterType`.
    - The returned gradient is differentiable only when:
      - `grad` is called with grad enabled (i.e., not inside
        `with torch.no_grad(): ...`), and
      - `x` (or any element of `args[0]` when it is a `ParameterType`) has
        `requires_grad=True`.
    - If `x` is a `Variable`, `smooth` and `grad` receives a `Variable` and
      `grad` returns a `Variable`. If `x` is `BaseVariableType`, `smooth` and 
      `grad` receives a `BaseVariableType` and `grad` returns a 
      `BaseVariableType`.
    - Differentiable inputs are expected to be leaf tensors; non-leaf inputs
      may error.
    """

    def grad_s(x, *args, **kwargs):
        outer_grad_enabled = torch.is_grad_enabled()
        x_was_var = isinstance(x, Variable)
        x_data = x.data if x_was_var else x

        # Flatten x into a tuple of tensors we can hand to autograd.grad
        x_flat, x_spec = flatten(x_data)

        # Optionally flatten params if they are a ParameterType
        # (for create_graph detection only)
        u_flat: tuple[torch.Tensor, ...] = ()
        if len(args) > 0 and isinstance(args[0], ParameterType):
            u_flat = (
                tuple(args[0]) if isinstance(args[0], TupleParameter) 
                else (args[0],)
            )
        
        # Store previous requries_grad of `x` and set them all to `True`.
        rgs = tuple(x_f.requires_grad for x_f in x_flat)
        _set_req_grad(x_flat, True)

        create_graph = outer_grad_enabled and (
            any(rg for rg in rgs) or any(u_f.requires_grad for u_f in u_flat)
        )

        # Compute objective value and grads w.r.t. x_flat
        with torch.enable_grad():
            x_unflat = unflatten(x_flat, x_spec)
            x_in = Variable(x_unflat) if x_was_var else x_unflat
            out = smooth(x_in, *args, **kwargs).sum()

        gd = torch.autograd.grad(
            out, x_flat, create_graph=create_graph, allow_unused=True
        )
        gd = tuple(
            torch.zeros_like(x_f) if g is None else g
            for x_f, g in zip(x_flat, gd)
        )
        
        # restore the old requries_grad.
        _set_req_grad(x_flat, rgs)

        # Unflatten gradient back to the same structure as x
        out_grad = unflatten(gd, x_spec)
        return Variable(out_grad) if x_was_var else out_grad

    return grad_s

def _gradient(
    smooth: Callable[..., torch.Tensor], *dargs: int
) -> Callable[..., BaseVariableType]:
    """
    Create a callable that computes the partial gradient of a smooth function.
    
    Given a scalar-valued callable `smooth`, this returns a new function that
    computes the gradient of `smooth` with respect to the arguments specified
    by `dargs`. Furthermore, the new function is also differentiable and can
    be used to compute the vector hessian products using backward() or any
    other autograd module of torch. Important Note: All differentiable inputs 
    are expected to be leaf tensors. Passing non-leaf tensors is unsupported 
    and may raise an error.
    
    Args:
        smooth (Callable): Smooth, scalar-valued function. Supports batched
            outputs (summed internally before differentiation).
        *dargs (int): Indices of arguments to differentiate w.r.t. If omitted,
            defaults to `(0,)`.

    Returns:
        Callable: The gradient computing function. Takes the same arguments
            as the smooth function.
    
    Example:
        >>> import torch
        >>> from paramancer.operators import gradient
        >>> w = lambda x, y, z: (x**2).sum() + (y*z).sum()
        >>> grad_xz = gradient(w, 0, 2)
        >>> x = torch.randn(3, requires_grad=True)
        >>> y = torch.randn(5, requires_grad=True)
        >>> z = torch.randn(5)
        >>> gx, gz = grad_xz(x, y, z)
        >>> assert torch.allclose(gx, 2*x)
        >>> assert torch.allclose(gz, y)
        >>> (gx.sum() + gz.sum()).backward()
        >>> assert z.grad is None
        >>> assert torch.allclose(x.grad, 2*torch.ones(3))
        >>> assert torch.allclose(y.grad, torch.ones(5))
    """
    def grad_s(*args):
        inps = [args[i] for i in dargs] if dargs else [args[0]]
        rgs = [inp.requires_grad for inp in inps]
        create_graph = any(
            getattr(arg, "requires_grad", False) for arg in args
        )
        
        # Temporarily enable requires_grad for inputs if needed
        for inp, rg in zip(inps, rgs):
            if not rg:
                inp.requires_grad_(True)
        with torch.enable_grad():
            out = smooth(*args).sum()
        gd = torch.autograd.grad(
            out, inps, create_graph=create_graph, allow_unused=True
        )
        if len(gd) == 1:
            gd = gd[0]
        
        # Restore original requires_grad state
        for inp, rg in zip(inps, rgs):
            if not rg:
                inp.requires_grad_(False)
        return gd
    return grad_s
