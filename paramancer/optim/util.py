from dataclasses import dataclass
import torch
from typing import Optional, Union
from paramancer.operators.norms import l2


def tmap(fn, x, y=None):
    """Apply a function elementwise to tensor(s) or tuple(s) of tensors."""
    if not isinstance(x, (torch.Tensor, tuple)):
        raise TypeError(f"Unsupported type {type(x)}")
    if isinstance(x, torch.Tensor):
        return fn(x, y) if y is not None else fn(x)
    if y is None:
        return tuple(fn(xi) for xi in x)
    return tuple(fn(xi, yi) for xi, yi in zip(x, y))


def tclone(x):
    return tmap(lambda t: t.clone(), x)


def tadd(x, y):
    return tmap(lambda a, b: a + b, x, y)


def tsub(x, y):
    return tmap(lambda a, b: a - b, x, y)


def tscale(x, alpha):
    return tmap(lambda t: alpha * t, x)


def tnorm2(x):
    """Compute L2 norm squared for tensor or tuple of tensors."""
    if isinstance(x, torch.Tensor):
        return torch.norm(x)
    elif isinstance(x, tuple):
        return torch.sqrt(sum(torch.sum(t**2) for t in x))
    else:
        raise TypeError(f"Unsupported type {type(x)}")


def default_metric(residual):
    return torch.inf if residual is None else l2(residual)


@dataclass
class OptimizationResult:
    solution: Union[torch.Tensor, tuple[torch.Tensor]]
    iterations: int
    metric: Optional[float]
    converged: bool
