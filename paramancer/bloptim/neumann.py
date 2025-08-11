import torch
from typing import Callable
from paramancer.optim.step import OptimizerStep
from paramancer.optim import Optimizer

class AffineStep(OptimizerStep):
    def __init__(self, operator: Callable, vector: torch.Tensor):
        self.operator = operator
        self.vector = vector
    
    def step(self, x_curr: torch.Tensor) -> torch.Tensor:
        return self.operator(x_curr) + self.vector

class NeumannSeries(Optimizer):
    def __init__(
        self,
        operator: Callable,
        vector: torch.Tensor,
        tol: float=1e-5,
        iters: int=100,
        metric: None | Callable=None,
        store_history: bool=False,
        verbose: bool=False
    ):
        step = AffineStep(operator, vector)
        super().__init__(step, tol, iters, metric, store_history, verbose)

