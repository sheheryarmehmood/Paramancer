import torch
from typing import Callable
from paramancer.optim.step import AffineStep
from paramancer.optim import Optimizer
from paramancer.optim.variable import Variable

class NeumannSeries(Optimizer):
    def __init__(
        self,
        operator: Callable,
        vector: torch.Tensor,
        tol: float=1e-5,
        iters: int=100,
        metric: None | str | Callable=None,
        store_history: bool=False,
        verbose: bool=False
    ):
        tracking = metric == "default"
        step = AffineStep(operator, vector, residual_tracking=tracking)
        super().__init__(step, tol, iters, metric, store_history, verbose)
    
    def __call__(
        self, iters: None | int=None
    ) -> torch.Tensor:
        return self.run(torch.zeros_like(self.step.vector.data), iters)


def neumann_series(
    operator: Callable,
    vector: torch.Tensor,
    tol: float=1e-5,
    iters: int=100,
    metric: None | str | Callable=None,
    store_history: bool=False,
    verbose: bool=False
):
    neumann = NeumannSeries(
        operator, vector, tol, iters, metric, store_history, verbose
    )
    return neumann()