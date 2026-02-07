from __future__ import annotations

import torch

from paramancer.optim.step import AffineStep
from paramancer.optim import Optimizer
from paramancer.optim.types import VariableLike, MetricSpec, LinOpType

class NeumannSeries(Optimizer):
    def __init__(
        self,
        lin_op: LinOpType,
        vector: VariableLike,
        tol: float = 1e-5,
        iters: int = 100,
        metric: MetricSpec = None,
        store_history: bool=False,
        verbose: bool = False
    ):
        tracking = metric == "default"
        step = AffineStep(lin_op, vector, tracking=tracking)
        super().__init__(step, tol, iters, metric, store_history, verbose)
    
    def __call__(
        self, init: VariableLike | None = None, iters: int | None = None
    ) -> torch.Tensor:
        if init is None:
            init = torch.zeros_like(self.step.vector.data)
        return self.run(init, iters)


def neumann_series(
    lin_op: LinOpType,
    vector: VariableLike,
    init: VariableLike | None = None,
    tol: float = 1e-5,
    iters: int = 100,
    metric: MetricSpec = None,
    store_history: bool = False,
    verbose: bool = False
):
    neumann = NeumannSeries(
        lin_op, vector, tol, iters, metric, store_history, verbose
    )
    return neumann(init)