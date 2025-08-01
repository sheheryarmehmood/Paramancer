import torch
import torch.linalg as la
from .norms import l2, l2_in

from typing import Union, Tuple, Callable, Any

def dual(
    primal: Callable[..., torch.Tensor],
    inps: Tuple[Any, ...]
):
    """Given the prox of a function f, computes the prox of f*, i.e., the \
        Fenchel conjugate of f, by using Moreau Identity.

    Args:
        primal (callable): A function to compute the prox of f.
        inps (tuple): Input arguments of the prox of f* or f. Prox is \
            computed w.r.t the first argument only. Rest are the parameters.

    Returns:
        prox of the dual of f at given input values.
    """
    return inps[0] - primal(*inps)

# %% Simple Norms

def li_ball(
    x: torch.Tensor,
    rad: torch.Tensor
) -> torch.Tensor:
    """Computes the projection of a point onto a ball given by infinity norm.
    
    Args:
        x (torch.Tensor): Given point.
        rad (torch.Tensor): Radius of the ball. Either rad.numel() must be 1 \
            or rad.shape should be the same as x.shape.

    Returns:
        torch.Tensor: projection onto the infinity ball. Has same shape as x.
    """
    return torch.maximum(torch.minimum(x, rad), -rad)

def l1_norm(
    x: torch.Tensor,
    scal: torch.Tensor
) -> torch.Tensor:
    """Computes the prox of l1 norm at a point, i.e.,
                    prox         (x)
                        scal*|.|
                                1

    Args:
        x (torch.Tensor): Given point.
        scal (torch.Tensor): Scale of l1 norm. Either scal.numel() must be \
            1 or scal.shape should be the same as x.shape.

    Returns:
        torch.Tensor: prox of l1 norm at a point. Has same shape as x.
    """
    return dual(li_ball, (x, scal))


def l2_ball(
    x: torch.Tensor,
    rad: torch.Tensor,
    eps: float=1e-8
) -> torch.Tensor:
    """Computes the projection of a point onto a ball given by Euclidean norm.

    Args:
        x (torch.Tensor): Given point.
        rad (torch.Tensor): Radius of the ball. rad.numel() must be 1.
        eps (float): small value to ensure norms.l2(l2_ball(x)) <= rad.

    Returns:
        torch.Tensor: projection onto the Euclidean ball. Has same shape as x.
    """
    return x / max(1., l2(x) / rad + eps)
def l2_norm(
    x: torch.Tensor,
    scal: torch.Tensor
) -> torch.Tensor:
    """Computes the prox of Euclidean norm at a point, i.e.,
                    prox         (x)
                        scal*|.|
                                2

    Args:
        x (torch.Tensor): Given point.
        scal (torch.Tensor): Scale of Euclidean norm. scal.numel() must be 1.

    Returns:
        torch.Tensor: prox of Euclidean norm at a point. Has same shape as x.
    """
    return dual(l2_ball, (x, scal))


def lo_ball(
    X: torch.Tensor,
    rad: torch.Tensor
) -> torch.Tensor:
    """Computes the projection of a point onto a ball given by spectral or \
        operator norm.

    Args:
        X (torch.Tensor): Given point. X.ndim must be 2.
        rad (torch.Tensor): Radius of the ball. Either rad.numel() must be \
            1 or rad.shape should be the same as (rank(X),).

    Returns:
        torch.Tensor: projection onto the spectral norm ball. Has same shape \
            as X.
    """
    U, s, Vh = la.svd(X, full_matrices=False)
    ps = torch.minimum(s, rad)
    return U @ ps.diag() @ Vh
def ln_norm(
    X: torch.Tensor,
    scal: torch.Tensor
) -> torch.Tensor:
    """Computes the prox of nuclear norm at a point, i.e.,
                    prox         (X)
                        scal*|.|
                                n

    Args:
        X (torch.Tensor): Given point. X.ndim must be 2.
        scal (torch.Tensor): Scale of nuclear norm. Either scal.numel() must \
            be 1 or scal.shape should be the same as (rank(X),).

    Returns:
        torch.Tensor: prox of nuclear norm at a point. Has same shape as X.
    """
    return dual(lo_ball, (X, scal))


# %% Group Norms

def l2i_ball(
    X: torch.Tensor,
    rad: torch.Tensor,
    dim: Union[int, Tuple[int]],
    keepdim: bool=False,
    eps: float=1e-8
) -> torch.Tensor:
    """Computes the projection of a point onto a ball given by the group \
        l2-li norm. Here i in li stands for infinity.

    Args:
        X (torch.Tensor): Given point.
        rad (torch.Tensor): Radius of the ball. rad.numel() must be 1.
        dim (int | tuple[int]): dimensions along which l2-norm is computed.
        keepdim (bool): whether to keep the reduced dimension or not.
        eps (float): small value to ensure norms.l2(l2i_ball(x)) <= rad.

    Returns:
        torch.Tensor: projection onto the l2-li ball. Has same shape as X.
    """
    return X / torch.maximum(
        torch.ones_like(rad), l2_in(X, dim, keepdim=keepdim) / rad + eps
    )
def l21_norm(
    X: torch.Tensor,
    scal: torch.Tensor,
    dim: Union[int, Tuple[int]],
    keepdim: bool=False
):
    """Computes the prox of the group l2-l1 norm at a point, i.e.,
                    prox           (X)
                        scal*|.|
                                2,1

    Args:
        X (torch.Tensor): Given point,.
        scal (torch.Tensor): Scale of l2-l1 norm. scal.numel() must be 1,.
        dim (int | tuple[int]): dimensions along which l2-norm is computed.
        keepdim (bool): whether to keep the reduced dimension or not.

    Returns:
        _type_: prox of l2-l1 norm at a point. Has same shape as X.
    """
    return dual(l2i_ball, (X, scal, dim, keepdim))


def loi_ball(
    X: torch.Tensor,
    r: torch.Tensor
) -> torch.Tensor:
    """Computes the projection of a point onto a ball given by the group \
        lo-li norm. Here o and i in lo-li stand for operator and infinity \
        respectively. Operator norm is computed along the last two dimensions.

    Args:
        X (torch.Tensor): Given point.
        r (torch.Tensor): Radius of the ball. r.numel() must be 1.

    Returns:
        torch.Tensor: projection onto the lo-li ball. Has same shape as X.
    """
    U, s, Vh = la.svd(X, full_matrices=False)
    ps = torch.minimum(s, r)
    return U @ ps.diag_embed() @ Vh
def ln1_norm(
    X: torch.Tensor,
    scal: torch.Tensor
) -> torch.Tensor:
    """Computes the prox of the group ln-l1 norm at a point, i.e.,
                    prox           (X)
                        scal*|.|
                                n,1
    Here n in ln stand for nuclear. Operator norm is computed along the last \
        two dimensions.

    Args:
        X (torch.Tensor): Given point.
        scal (torch.Tensor): Scale of ln-l1 norm. scal.numel() must be 1.

    Returns:
        torch.Tensor: prox of ln-l1 norm at a point. Has same shape as X.
    """
    return dual(loi_ball, (X, scal))
