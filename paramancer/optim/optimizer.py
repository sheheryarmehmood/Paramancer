from __future__ import annotations
from tqdm import tqdm
import torch

from .variable import Variable
from .step import (
    OptimizerStep,
    AffineStep, GDStep, PolyakStep, NesterovStep,
    ProxGradStep, FISTAStep, PDHGStep
)
from .util import OptimizationResult, to_float_scalar
from .types import (
    GradMapType, ProxMapType, LinOpType,
    MomentumSchedType, MetricSpec,
    ScalarLike, FlatVariable, TupleVariable, VariableLike
)

class Optimizer:
    def __init__(
        self,
        step: OptimizerStep,
        tol: float = 1e-5,
        iters: int = 100,
        metric: MetricSpec | None = None,
        store_history: bool = False,
        verbose: bool = False
    ):
        self.step = step
        self.tol = tol
        self.iters = iters
        if metric == "default":
            self.metric = None
        else:
            self.metric = metric
        self.verbose = verbose
        self.store_history = store_history
        self.history = [] if store_history else None
        self.result = None
    
    def __call__(
        self, x_init: VariableLike, iters: int | None = None
    ) -> VariableLike:
        return self.run(x_init, iters)
    
    @Variable.ensure_var_input
    def run(self, x_init: Variable, iters: None | int = None) -> Variable:
        
        x_curr = x_init.clone()
        if self.store_history:
            self.history.append(x_curr.clone())
        
        if iters is not None:
            self.iters = iters
        
        converged = False
        pbar = range(self.iters)
        if self.verbose:
            pbar = tqdm(pbar)
        metric_val = None
        for k in pbar:
            x_curr = self.step(x_curr)
            
            if self.store_history:
                self.history.append(x_curr.clone())
            
            if self.metric is None and not self.step.residual_tracking:
                continue
            
            if self.step.residual_tracking:
                # vvvvv torch.Tensor Operation vvvvv
                metric_val = to_float_scalar(self.step.residual.norm())
            else:
                metric_val = to_float_scalar(self.metric(x_curr.data))

            if self.verbose:
                pbar.set_description(f"{metric_val:.6e}")
            
            if metric_val < self.tol:
                converged = True
                break
        
        self.result = OptimizationResult(
            solution=x_curr,
            iterations=k + 1,
            metric=metric_val,
            converged=converged
        )

        return x_curr


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
    ) -> VariableLike:
        if init is None:
            init = torch.zeros_like(self.step.vector.data)
        return self.run(init, iters)


class GradientDescent(Optimizer):
    def __init__(
        self,
        stepsize: ScalarLike,
        grad_map: GradMapType,
        tol: float = 1e-5,
        iters: int = 100,
        metric: MetricSpec | None = None,
        store_history: bool = False,
        verbose: bool = False
    ):
        tracking = metric == "default"
        step = GDStep(stepsize, grad_map, tracking=tracking)
        super().__init__(step, tol, iters, metric, store_history, verbose)


class HeavyBall(Optimizer):
    def __init__(
        self,
        stepsize: ScalarLike,
        momentum: ScalarLike,
        grad_map: GradMapType,
        tol: float = 1e-5,
        iters: int = 100,
        metric: MetricSpec | None = None,
        store_history: bool = False,
        verbose: bool = False
    ):
        tracking = metric == "default"
        step = PolyakStep(
            stepsize, momentum, grad_map, tracking=tracking
        )
        super().__init__(step, tol, iters, metric, store_history, verbose)

class AcceleratedGradient(Optimizer):
    def __init__(
        self,
        stepsize,
        grad_map: GradMapType,
        momentum_scheduler: MomentumSchedType | None = None,
        tol: float = 1e-5,
        iters: int = 100,
        metric: MetricSpec | None = None,
        store_history: bool = False,
        verbose: bool = False
    ):
        tracking = metric == "default"
        step = NesterovStep(
            stepsize, grad_map, momentum_scheduler=momentum_scheduler,
            tracking=tracking
        )
        super().__init__(step, tol, iters, metric, store_history, verbose)
    
    def restart(self):
        self.step.restart()


class ProximalGradient(Optimizer):
    def __init__(
        self,
        stepsize: ScalarLike,
        grad_map: GradMapType,
        prox_map: ProxMapType,
        tol: float = 1e-5,
        iters: int = 100,
        metric: MetricSpec | None = None,
        store_history: bool = False,
        verbose: bool = False
    ):
        tracking = metric == "default"
        step = ProxGradStep(
            stepsize, grad_map, prox_map, tracking=tracking
        )
        super().__init__(step, tol, iters, metric, store_history, verbose)


class FISTA(Optimizer):
    def __init__(
        self,
        stepsize: ScalarLike,
        grad_map: GradMapType,
        prox_map: ProxMapType,
        momentum_scheduler: MomentumSchedType | None = None,
        tol: float = 1e-5,
        iters: int = 100,
        metric: MetricSpec | None = None,
        store_history: bool = False,
        verbose: bool = False
    ):
        tracking = metric == "default"
        step = FISTAStep(
            stepsize, grad_map, prox_map, tracking=tracking,
            momentum_scheduler=momentum_scheduler
        )
        super().__init__(step, tol, iters, metric, store_history, verbose)
    
    def restart(self):
        self.step.restart()


class PDHG(Optimizer):
    def __init__(
        self,
        stepsize_primal: ScalarLike,
        stepsize_dual: ScalarLike,
        prox_map_primal: ProxMapType,
        prox_map_dual: ProxMapType,
        lin_op: LinOpType,
        lin_op_adj: LinOpType | None = None,
        zero_el: FlatVariable | TupleVariable | None = None,
        tol: float = 1e-5,
        iters: int = 100,
        metric: MetricSpec | None = None,
        store_history: bool = False,
        verbose: bool = False
    ):
        """Eactly one out of `zero_el` and `lin_op_adj` should be `None`.
        """
        step = PDHGStep(
            stepsize_primal, stepsize_dual, prox_map_primal, prox_map_dual,
            lin_op, lin_op_adj, zero_el, tracking=(metric == "default")
        )
        super().__init__(step, tol, iters, metric, store_history, verbose)
    
    def __call__(
        self,
        x_init_primal: VariableLike,
        x_init_dual: VariableLike,
        iters: int | None = None
    ) -> VariableLike:
        x_init = Variable.from_pdhg(x_init_primal, x_init_dual)
        return self.run(x_init, iters)

