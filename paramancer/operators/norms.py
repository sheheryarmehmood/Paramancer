import torch
import torch.linalg as la
from typing import Union, Tuple

"""
Definition of various (group) norms defined for tensors
"""

def _reduction_dims(
    ndim: int,
    batch: Union[int, Tuple[int]]
) -> Union[int, Tuple[int]]:
    """Given a tensor of dimension ndim, suppose that we want to perform \
        an operation along the dimension given by dim. Then the dimensions \
        given by batch will be preserved. This method does the conversion \
        from batch to dim (of course, we can also make the conversion in \
        the reverse direction as well using the same method.) \

    Args:
        ndim (int)
        batch (int | tuple[int])
    
    Raises:
        ValueError: when value of batch is not correct.

    Returns:
        int | tuple[int]
    """
    if not isinstance(batch, (list, tuple)):
        batch = [batch]
    for bat in batch:
        if not (-1 <= bat < ndim):
            raise ValueError("Value of batch is not correct.")
    return tuple(set(range(ndim)) - set(batch))

# %% Simple Norms

def ip(
    p: torch.Tensor,
    q: torch.Tensor,
    batch: Union[int, Tuple[int]] = -1
) -> torch.Tensor:
    """Computes the Euclidean dot product of tensors p and q.

    Args:
        p (torch.Tensor)
        q (torch.Tensor): p.shape must be the same as q.shape
        batch (int | tuple[int], optional): dimension along which inner \
            product should not be computed. The default is -1 which means \
            the operation is performed along all dimensions. Defaults to -1.

    Returns:
        torch.Tensor
    """
    dim = _reduction_dims(p.ndim, batch)
    return (p*q).sum(dim=dim)
def l2_sq(
    p: torch.Tensor,
    batch: Union[int, Tuple[int]] = -1
) -> torch.Tensor:
    """Computes the squared Euclidean norm of tensor p.

    Args:
        p (torch.Tensor)
        batch (int | tuple[int], optional): dimension along which the squared \
            norm should not be computed. The default is -1 which means the \
            operation is performed along all dimensions. Defaults to -1.

    Returns:
        torch.Tensor
    """
    return ip(p, p, batch=batch)

def l2(
    p: torch.Tensor,
    batch: Union[int, Tuple[int]] = -1
) -> torch.Tensor:
    """Computes the Euclidean norm of tensor p.

    Args:
        p (torch.Tensor)
        batch (int | tuple[int], optional): dimension along which the norm \
            should not be computed. The default is -1 which means the \
            operation is performed along all dimensions. Defaults to -1.

    Returns:
        torch.Tensor
    """
    dim = _reduction_dims(p.ndim, batch)
    return la.vector_norm(p, ord=2, dim=dim)

def l1(
    p: torch.Tensor,
    batch: Union[int, Tuple[int]] = -1
) -> torch.Tensor:
    """Computes the l1 norm of tensor p.

    Args:
        p (torch.Tensor)
        batch (int | tuple[int], optional): dimension along which the norm \
            should not be computed. The default is -1 which means the \
            operation is performed along all dimensions. Defaults to -1.

    Returns:
        torch.Tensor
    """
    dim = _reduction_dims(p.ndim, batch)
    return p.abs().sum(dim=dim)

def li(
    p: torch.Tensor,
    batch: Union[int, Tuple[int]] = -1
) -> torch.Tensor:
    """Computes the infinity norm of tensor p.

    Args:
        p (torch.Tensor)
        batch (int | tuple[int], optional): dimension along which the norm \
            should not be computed. The default is -1 which means the \
            operation is performed along all dimensions. Defaults to -1.

    Returns:
        torch.Tensor
    """
    dim = _reduction_dims(p.ndim, batch)
    return p.abs().amax(dim=dim)

# %% Group Norms

def l2_in(
    p: torch.Tensor,
    dim: Union[int, Tuple[int]],
    keepdim: bool = False
) -> torch.Tensor:
    """Computes the Euclidean norm of tensor p. Useful when computing the \
        group norms where the inner norm is Euclidean. 'in' in l2_in stands \
        for inner.

    Args:
        p (torch.Tensor)
        dim (int | tuple[int]): dimension along which the norm should be \
            computed.
        keepdim (bool, optional): After the reduction, the p.ndim matches \
            the number of dimensions in the output tensor, if keepdim is \
            true. The default is False. Defaults to False.

    Returns:
        torch.Tensor
    """
    return la.vector_norm(p, ord=2, dim=dim, keepdim=keepdim)
def l21(
    p: torch.Tensor,
    dim: Union[int, Tuple[int]],
    batch: Union[int, Tuple[int]] = -1
) -> torch.Tensor:
    """Computes the group l2-l1 norm of tensor p. l2 is the inner norm.

    Args:
        p (torch.Tensor)
        dim (int | tuple[int]): dimension along which the inner norm should \
            be computed.
        batch (int | tuple[int], optional): dimension along which the norm \
            should not be computed. The default is -1 which means the \
            operation is performed along all dimensions. Defaults to -1.

    Returns:
        torch.Tensor
    """
    q = l2_in(p, dim)
    dim = _reduction_dims(q.ndim, batch)
    return q.sum(dim=dim)

def l2i(
    p: torch.Tensor,
    dim: Union[int, Tuple[int]],
    batch: Union[int, Tuple[int]] = -1
) -> torch.Tensor:
    """Computes the group l2-li norm of tensor p. l2 is the inner norm and \
        'i' stands for infinity.

    Args:
        p (torch.Tensor)
        dim (int | tuple[int]): dimension along which the inner norm should \
            be computed.
        batch (int | tuple[int], optional): dimension along which the norm \
            should not be computed. The default is -1 which means the \
            operation is performed along all dimensions. Defaults to -1.

    Returns:
        torch.Tensor
    """
    q = l2_in(p, dim)
    dim = _reduction_dims(q.ndim, batch)
    return q.amax(dim=dim)

def ln_in(
    p: torch.Tensor,
    keepdim: bool = False
) -> torch.Tensor:
    """Computes the Nuclear norm of tensor p. Useful when computing the group \
        norms where the inner norm is Nuclear. 'n' and 'in' in ln_in stand \
        respectively for nuclear and inner. Always computed along the last \
        two dimensions.

    Args:
        p (torch.Tensor)
        keepdim (bool, optional): After the reduction, the p.ndim matches the \
            number of dimensions in the output tensor, if keepdim is true. \
            The default is False. Defaults to False.

    Returns:
        torch.Tensor
    """
    return la.matrix_norm(p, ord='nuc', keepdim=keepdim)
def ln1(
    p: torch.Tensor,
    batch: Union[int, Tuple[int]] = -1
) -> torch.Tensor:
    """Computes the group ln-l1 norm of tensor p. ln is the inner norm and \
        'n' stands for nuclear. ln is computed along the last two dimensions.

    Args:
        p (torch.Tensor)
        batch (int | tuple[int], optional): dimension along which the norm \
            should not be computed. The default is -1 which means the \
            operation is performed along all dimensions. Defaults to -1.

    Returns:
        torch.Tensor
    """
    q = ln_in(p)
    dim = _reduction_dims(q.ndim, batch)
    return q.sum(dim=dim)

def lo_in(
    p: torch.Tensor,
    keepdim: bool = False
) -> torch.Tensor:
    """Computes the Operator norm of tensor p. Useful when computing the \
        group norms where the inner norm is Operator. 'o' and 'in' in lo_in \
        stand respectively for operator and inner. Always computed along the \
        last two dimensions.

    Args:
        p (torch.Tensor)
        keepdim (bool, optional): After the reduction, the p.ndim matches the \
            number of dimensions in the output tensor, if keepdim is true. \
            The default is False. Defaults to False.

    Returns:
        torch.Tensor
    """
    return la.matrix_norm(p, ord=2, keepdim=keepdim)
def loi(
    p: torch.Tensor,
    batch: Union[int, Tuple[int]] = -1
) -> torch.Tensor:
    """Computes the group lo-li norm of tensor p. lo is the inner norm and \
        'o' and 'i' stand respectively for operator and infinity. lo is \
        computed along the last two dimensions.

    Args:
        p (torch.Tensor)
        batch (int | tuple[int], optional): dimension along which the norm \
            should not be computed. The default is -1 which means the \
            operation is performed along all dimensions. Defaults to -1.

    Returns:
        torch.Tensor
    """
    q = lo_in(p)
    dim = _reduction_dims(q.ndim, batch)
    return q.amax(dim=dim)
