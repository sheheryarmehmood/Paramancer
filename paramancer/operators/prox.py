import torch
import torch.linalg as la
from .norms import l2, l2_in
from ._docstrings import prox_doc
from ._util import dual

from typing import Union, Tuple


# %% Simple Vector Norms

def inf_ball(p: torch.Tensor, rad: torch.Tensor) -> torch.Tensor:
    return torch.maximum(torch.minimum(p, rad), -rad)
def l1_norm(p: torch.Tensor, scal: torch.Tensor) -> torch.Tensor:
    return dual(inf_ball, (p, scal))


def l2_ball(
    p: torch.Tensor, rad: torch.Tensor, eps: float=1e-8
) -> torch.Tensor:
    return p / max(1., l2(p) / rad + eps)
def l2_norm(
    p: torch.Tensor, scal: torch.Tensor, eps: float=1e-8
) -> torch.Tensor:
    return dual(l2_ball, (p, scal))


# %% Simple Matrix Norms

def spec_ball(p: torch.Tensor, rad: torch.Tensor) -> torch.Tensor:
    U, s, Vh = la.svd(p, full_matrices=False)
    ps = torch.minimum(s, rad)
    return U @ ps.diag() @ Vh
def nuc_norm(p: torch.Tensor, scal: torch.Tensor) -> torch.Tensor:
    return dual(spec_ball, (p, scal))


# %% Composite Vector Norms

def l2_inf_ball(
    p: torch.Tensor,
    rad: torch.Tensor,
    dim: Union[int, Tuple[int]],
    keepdim: bool=False,
    eps: float=1e-8
) -> torch.Tensor:
    return p / torch.maximum(
        torch.ones_like(rad), l2_in(p, dim, keepdim=keepdim) / rad + eps
    )
def l2_l1_norm(
    p: torch.Tensor,
    scal: torch.Tensor,
    dim: Union[int, Tuple[int]],
    keepdim: bool=False,
    eps: float=1e-8
) -> torch.Tensor:
    return dual(l2_inf_ball, (p, scal, dim, keepdim, eps))


# %% Composite Matrix Norms

def spec_inf_ball(p: torch.Tensor, rad: torch.Tensor) -> torch.Tensor:
    U, s, Vh = la.svd(p, full_matrices=False)
    ps = torch.minimum(s, rad)
    return U @ ps.diag_embed() @ Vh
def nuc_l1_norm(p: torch.Tensor, scal: torch.Tensor) -> torch.Tensor:
    return dual(spec_inf_ball, (p, scal))


# %% Docstring Bindings

inf_ball.__doc__ = prox_doc("infinity", projection=True)
l1_norm.__doc__ = prox_doc("l1")

l2_ball.__doc__ = prox_doc("Euclidean", projection=True)
l2_norm.__doc__ = prox_doc("Euclidean")

spec_ball.__doc__ = prox_doc("spectral", projection=True)
nuc_norm.__doc__ = prox_doc("nuclear")

l2_inf_ball.__doc__ = prox_doc(
    "composite Euclidean-infinity", projection=True
)
l2_l1_norm.__doc__ = prox_doc("composite Euclidean-l1")

spec_inf_ball.__doc__ = prox_doc(
    "composite spectral-infinity", projection=True,
)
nuc_l1_norm.__doc__ = prox_doc("composite nuclear-l1")


