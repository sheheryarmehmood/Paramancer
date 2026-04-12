import torch

from ..optim.step import OptimizerStep

from ..optim.optimizer import NeumannSeries
from ..variable.util import zeros_like
from ..variable.flat import FlatVar
from ..variable.parameter import AlgoParam
from ..variable.types import (
    LinOpType, MetricSpec, ParamMarkovStep, AlgoVar, AlgoVarLike
)


def neumann_series(
    lin_op: LinOpType,
    vector: AlgoVarLike,
    init: AlgoVarLike | None = None,
    tol: float = 1e-5,
    iters: int = 100,
    metric: MetricSpec | None = None,
    store_history: bool = False,
    verbose: bool = False
):
    neumann = NeumannSeries(
        lin_op, vector, tol, iters, metric, store_history, verbose
    )
    return neumann(init=init)


class ImplicitDifferentiation:
    def __init__(
        self,
        param_step: ParamMarkovStep,
        tol: float = 1e-5,
        iters: int = 100,
        metric: MetricSpec | None = None,
        verbose: bool = False
    ):
        self.step = param_step
        self._adjoint_state = None
        self.tol = tol
        self.iters = iters
        self.metric = metric
        self.verbose = verbose
    
    def vjp(
        self,
        xmin: FlatVar,
        u_given: AlgoParam,
        xmin_grad: FlatVar,
        init: AlgoVar | None = None
    ) -> AlgoParam:
        zmin = xmin if self.step.is_markovian() else (xmin, xmin)
        zmin_grad = (
            xmin_grad if self.step.is_markovian() 
            else (xmin_grad, xmin_grad.zeros_like())
        )
        self._adjoint_state = neumann_series(
            lambda v: self.step.vjp_var(zmin, u_given, v), zmin_grad,
            init=init, tol=self.tol, iters=self.iters,
            metric=self.metric, verbose=self.verbose
        )
        return self.step.vjp_par(zmin, u_given, self._adjoint_state)
    
    def jvp(
        self,
        xmin: FlatVar,
        u_given: AlgoParam,
        u_tangent: AlgoParam,
        init: AlgoVar | None = None
    ) -> AlgoVar:
        zmin = xmin if self.step.is_markovian() else (xmin, xmin)
        jvp_par_tan = self.step.jvp_par(zmin, u_given, u_tangent)
        self._zmin_tangent = neumann_series(
            lambda v: self.step.jvp_var(zmin, u_given, v), jvp_par_tan,
            init=init, tol=self.tol, iters=self.iters,
            metric=self.metric, verbose=self.verbose
        )
        return self._zmin_tangent

    @property
    def adjoint_state(self):
        return self._adjoint_state
    
    @property
    def zmin_tangent(self):
        return self._zmin_tangent
    
    


