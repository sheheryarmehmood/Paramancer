from torch import Tensor
from typing import Any, Callable, Tuple, Union

def dual(
    primal: Callable[..., Tensor],
    inps: Tuple[Any, ...]
):
    """Computes the prox of the Fenchel conjugate of a function.
    
    Given the prox of a function f, it uses Moreau Identity to compute the
    prox of f*, the Fenchel conjugate of f.
        

    Args:
        primal (callable): A function to compute the prox of f.
        inps (tuple): Input arguments of the prox of f* or f. Prox is computed
            w.r.t the first argument only. Rest are the parameters.

    Returns:
        prox of the dual of f at given input values.
    """
    return inps[0] - primal(*inps)


def reduction_dims(
    ndim: int, batch: Union[int, Tuple[int]]
) -> Union[int, Tuple[int]]:
    """Given a tensor of dimension ndim, suppose that we want to perform an
    operation along the dimension given by dim. Then the dimensions given by
    batch will be preserved. This method does the conversion from batch to dim.
    We can also make the conversion in the reverse direction as well using the
    same method.

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