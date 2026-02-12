from __future__ import annotations

from ..optim.optimizer import NeumannSeries
from ..variable.types import VariableLike, MetricSpec, LinOpType


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
