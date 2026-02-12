from __future__ import annotations
import torch
from collections.abc import Callable

from ..variable import Variable, flatten, unflatten
from ..variable.types import BaseVariableType, PObjType, PGradMapType, ParameterType

def gradient(
    smooth: PObjType
) -> PGradMapType:
    """
    Return a callable that differentiates `smooth` with respect to its first
    argument (`BaseVariableLike`), including parametric objectives where the
    first argument must not be a parameter.

    If a parameter argument is of type `ParameterType`, or if the first
    argument has `requires_grad=True`, the returned gradient function is itself
    differentiable, enabling second-order derivatives with respect to those
    arguments.
    
    This can be useful in differentiating non-parametric objectives as well as
    lower level objectives which take additional parameters including the 
    """

    def grad_s(x, *args):
        x_was_var = isinstance(x, Variable)
        x_data = x.data if x_was_var else x
        x_flat, x_spec = flatten(x_data)

        prm_flat = ()
        if isinstance(args[0], ParameterType):
            prm_flat = tuple(
                args[0]) if isinstance(args[0], torch.nn.ParameterList
            ) else (args[0],)

        x_len = len(x_flat)

        def smooth_flat(*flat_args):
            x_args = flat_args[:x_len]
            x_unflat = unflatten(x_args, x_spec)
            x_in = Variable(x_unflat) if x_was_var else x_unflat
            return smooth(x_in, *args)

        grad_fn = _gradient(smooth_flat, *range(x_len))
        gd = grad_fn(*x_flat, *prm_flat, *args)
        if not isinstance(gd, tuple):
            gd = (gd,)
        out = unflatten(gd, x_spec)
        return Variable(out) if x_was_var else out
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
        gd = torch.autograd.grad(out, inps, create_graph=create_graph)
        if len(gd) == 1:
            gd = gd[0]
        
        # Restore original requires_grad state
        for inp, rg in zip(inps, rgs):
            if not rg:
                inp.requires_grad_(False)
        return gd
    return grad_s
