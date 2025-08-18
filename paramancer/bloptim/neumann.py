import torch
from typing import Callable
from paramancer.optim.step import OptimizerStep
from paramancer.optim import Optimizer

class AffineStep(OptimizerStep):
    def __init__(
        self,
        operator: Callable,
        vector: torch.Tensor,
        residual_tracking: bool=False
    ):
        super().__init__(tracking=residual_tracking)
        self.operator = operator
        self.vector = vector
    
    def step(self, x_curr: torch.Tensor) -> torch.Tensor:
        x_new = self.operator(x_curr) + self.vector
        if self._residual_tracking:
            self._residual = x_new - x_curr
        return x_new

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
        return self.run(0 * self.step.vector, iters)


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