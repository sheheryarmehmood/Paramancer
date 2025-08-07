import torch
import torch.linalg as la
from typing import Union, Tuple
from ._util import reduction_dims
from ._docstrings import norm_doc

"""
Definition of various (group) norms defined for tensors
"""


# %% Euclidean Inner Product

def inner_product(
    p: torch.Tensor, q: torch.Tensor, batch: Union[int, Tuple[int]] = -1
) -> torch.Tensor:
    return (p*q).sum(dim=reduction_dims(p.ndim, batch))


# %% Squared Euclidean Norm

def l2_sq(
    p: torch.Tensor, batch: Union[int, Tuple[int]] = -1
) -> torch.Tensor:
    return inner_product(p, p, batch=batch)


# %% Simple Norms

def l2(
    p: torch.Tensor, batch: Union[int, Tuple[int]] = -1
) -> torch.Tensor:
    return la.vector_norm(p, ord=2, dim=reduction_dims(p.ndim, batch))

def l1(
    p: torch.Tensor, batch: Union[int, Tuple[int]] = -1
) -> torch.Tensor:
    return p.abs().sum(dim=reduction_dims(p.ndim, batch))

def inf(
    p: torch.Tensor, batch: Union[int, Tuple[int]] = -1
) -> torch.Tensor:
    return p.abs().amax(dim=reduction_dims(p.ndim, batch))


# %% Composite Norms

def l2_in(
    p: torch.Tensor, dim: Union[int, Tuple[int]], keepdim: bool = False
) -> torch.Tensor:
    return la.vector_norm(p, ord=2, dim=dim, keepdim=keepdim)
def l2_l1(
    p: torch.Tensor,
    dim: Union[int, Tuple[int]],
    batch: Union[int, Tuple[int]] = -1
) -> torch.Tensor:
    q = l2_in(p, dim)
    dim = reduction_dims(q.ndim, batch)
    return q.sum(dim=dim)

def l2_inf(
    p: torch.Tensor,
    dim: Union[int, Tuple[int]],
    batch: Union[int, Tuple[int]] = -1
) -> torch.Tensor:
    q = l2_in(p, dim)
    dim = reduction_dims(q.ndim, batch)
    return q.amax(dim=dim)

def nuc_in(p: torch.Tensor, keepdim: bool = False) -> torch.Tensor:
    return la.matrix_norm(p, ord='nuc', keepdim=keepdim)
def nuc_l1(
    p: torch.Tensor, batch: Union[int, Tuple[int]] = -1
) -> torch.Tensor:
    q = nuc_in(p)
    return q.sum(dim=reduction_dims(q.ndim, batch))

def spec_in(p: torch.Tensor, keepdim: bool = False) -> torch.Tensor:
    return la.matrix_norm(p, ord=2, keepdim=keepdim)
def spec_inf(
    p: torch.Tensor, batch: Union[int, Tuple[int]] = -1
) -> torch.Tensor:
    q = spec_in(p)
    return q.amax(dim=reduction_dims(q.ndim, batch))


inner_product.__doc__ = norm_doc("Euclidean dot product")
l2_sq.__doc__ = norm_doc("squared Euclidean", squared=True)
l2.__doc__ = norm_doc("Euclidean")
l1.__doc__ = norm_doc("l1")
inf.__doc__ = norm_doc("infinity")
l2_in.__doc__ = norm_doc("Euclidean", inner=True, group_l2=True)
l2_l1.__doc__ = norm_doc("composite Euclidean-l1", group_l2=True)
l2_inf.__doc__ = norm_doc("composite Euclidean-infinity", group_l2=True)
nuc_in.__doc__ = norm_doc("nuclear", inner=True)
nuc_l1.__doc__ = norm_doc("composite nuclear-l1")
nuc_in.__doc__ = norm_doc("spectral", inner=True)
spec_inf.__doc__ = norm_doc("composite spectral-infinity")

