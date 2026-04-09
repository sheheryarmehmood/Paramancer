from __future__ import annotations

from typing import Any

from .step import (
    AffineStep,
    FISTAStep,
    GDStep,
    NesterovStep,
    OptimizerStep,
    PDHGStep,
    PolyakStep,
    ProxGradStep,
)
from .util import OptimizationResult, to_float_scalar
from ..variable.flat import FlatVar
from ..variable.pair import PairVar
from ..variable.types import (
    AlgoVarLike,
    FlatLinOpType,
    FlatRawVarType,
    FlatVarLike,
    LinOpType,
    MetricSpec,
    MomentumSchedType,
    PGradMapType,
    ProxMapType,
    PSmoothObjType,
    ScalarLike,
)
from ..variable.util import is_pair_raw_var, zeros_like
from ..variable.util import as_pair_var, is_flat_var, is_pair_var


class Optimizer:
    def __init__(
        self,
        step: OptimizerStep,
        tol: float = 1e-5,
        iters: int = 100,
        metric: MetricSpec | None = None,
        store_history: bool = False,
        verbose: bool = False,
    ):
        self.step = step
        self.tol = tol
        self.iters = iters
        self.metric = None if metric == "default" else metric
        self.verbose = verbose
        self.store_history = store_history
        self.history = [] if store_history else None
        self.result = None

    def __call__(
        self, x_init: AlgoVarLike, *args: Any, iters: int | None = None, **kwargs: Any
    ) -> AlgoVarLike:
        return self.run(x_init, *args, iters=iters, **kwargs)

    def run(
        self, x_init: AlgoVarLike, *args: Any, iters: int | None = None, **kwargs: Any
    ) -> AlgoVarLike:
        input_is_wrapper = is_flat_var(x_init) or is_pair_var(x_init)
        x_curr = (
            x_init.clone()
            if input_is_wrapper
            else (as_pair_var(x_init) if is_pair_raw_var(x_init) else FlatVar(x_init))
        )

        if self.store_history:
            self.history.append(x_curr.clone())

        if iters is not None:
            self.iters = iters

        converged = False
        pbar = range(self.iters)
        if self.verbose:
            from tqdm import tqdm

            pbar = tqdm(pbar)
        metric_val = None

        for k in pbar:
            x_curr = self.step(x_curr, *args, **kwargs)

            if self.store_history:
                self.history.append(x_curr.clone())

            if self.metric is None and not self.step.residual_tracking:
                continue

            if self.step.residual_tracking:
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
            converged=converged,
        )

        return x_curr if input_is_wrapper else x_curr.data


class NeumannSeries(Optimizer):
    def __init__(
        self,
        lin_op: LinOpType,
        vector: AlgoVarLike,
        tol: float = 1e-5,
        iters: int = 100,
        metric: MetricSpec = None,
        store_history: bool = False,
        verbose: bool = False,
    ):
        tracking = metric == "default"
        step = AffineStep(lin_op, vector, tracking=tracking)
        super().__init__(step, tol, iters, metric, store_history, verbose)

    def __call__(
        self,
        *args: Any,
        init: AlgoVarLike | None = None,
        iters: int | None = None,
        **kwargs: Any,
    ) -> AlgoVarLike:
        if init is None:
            init = zeros_like(self.step.vector.data)
        return self.run(init, *args, iters=iters, **kwargs)


class GradientDescent(Optimizer):
    def __init__(
        self,
        stepsize: ScalarLike,
        smooth_obj: PSmoothObjType | None = None,
        grad_map: PGradMapType | None = None,
        tol: float = 1e-5,
        iters: int = 100,
        metric: MetricSpec | None = None,
        store_history: bool = False,
        verbose: bool = False,
    ):
        tracking = metric == "default"
        step = GDStep(
            stepsize, smooth_obj=smooth_obj, grad_map=grad_map, tracking=tracking
        )
        super().__init__(step, tol, iters, metric, store_history, verbose)


class HeavyBall(Optimizer):
    def __init__(
        self,
        stepsize: ScalarLike,
        momentum: ScalarLike,
        smooth_obj: PSmoothObjType | None = None,
        grad_map: PGradMapType | None = None,
        tol: float = 1e-5,
        iters: int = 100,
        metric: MetricSpec | None = None,
        store_history: bool = False,
        verbose: bool = False,
    ):
        tracking = metric == "default"
        step = PolyakStep(
            stepsize, momentum, smooth_obj=smooth_obj, grad_map=grad_map, tracking=tracking
        )
        super().__init__(step, tol, iters, metric, store_history, verbose)


class AcceleratedGradient(Optimizer):
    def __init__(
        self,
        stepsize: ScalarLike,
        smooth_obj: PSmoothObjType | None = None,
        grad_map: PGradMapType | None = None,
        momentum_scheduler: MomentumSchedType | None = None,
        tol: float = 1e-5,
        iters: int = 100,
        metric: MetricSpec | None = None,
        store_history: bool = False,
        verbose: bool = False,
    ):
        tracking = metric == "default"
        step = NesterovStep(
            stepsize,
            smooth_obj=smooth_obj,
            grad_map=grad_map,
            momentum_scheduler=momentum_scheduler,
            tracking=tracking,
        )
        super().__init__(step, tol, iters, metric, store_history, verbose)

    def restart(self):
        self.step.restart()


class ProximalGradient(Optimizer):
    def __init__(
        self,
        stepsize: ScalarLike,
        prox_map: ProxMapType,
        smooth_obj: PSmoothObjType | None = None,
        grad_map: PGradMapType | None = None,
        tol: float = 1e-5,
        iters: int = 100,
        metric: MetricSpec | None = None,
        store_history: bool = False,
        verbose: bool = False,
    ):
        tracking = metric == "default"
        step = ProxGradStep(
            stepsize, prox_map, smooth_obj=smooth_obj, grad_map=grad_map, tracking=tracking
        )
        super().__init__(step, tol, iters, metric, store_history, verbose)


class FISTA(Optimizer):
    def __init__(
        self,
        stepsize: ScalarLike,
        prox_map: ProxMapType,
        smooth_obj: PSmoothObjType | None = None,
        grad_map: PGradMapType | None = None,
        momentum_scheduler: MomentumSchedType | None = None,
        tol: float = 1e-5,
        iters: int = 100,
        metric: MetricSpec | None = None,
        store_history: bool = False,
        verbose: bool = False,
    ):
        tracking = metric == "default"
        step = FISTAStep(
            stepsize,
            prox_map,
            smooth_obj=smooth_obj,
            grad_map=grad_map,
            tracking=tracking,
            momentum_scheduler=momentum_scheduler,
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
        lin_op: FlatLinOpType,
        lin_op_adj: FlatLinOpType | None = None,
        zero_el: FlatRawVarType | None = None,
        tol: float = 1e-5,
        iters: int = 100,
        metric: MetricSpec | None = None,
        store_history: bool = False,
        verbose: bool = False,
    ):
        step = PDHGStep(
            stepsize_primal,
            stepsize_dual,
            prox_map_primal,
            prox_map_dual,
            lin_op,
            lin_op_adj,
            zero_el,
            tracking=(metric == "default"),
        )
        super().__init__(step, tol, iters, metric, store_history, verbose)

    def __call__(
        self,
        x_init_primal: FlatVarLike,
        x_init_dual: FlatVarLike,
        *args: Any,
        iters: int | None = None,
        **kwargs: Any,
    ) -> AlgoVarLike:
        x_init = (
            PairVar(x_init_primal, x_init_dual)
            if is_flat_var(x_init_primal) or is_flat_var(x_init_dual)
            else (x_init_primal, x_init_dual)
        )
        return self.run(x_init, *args, iters=iters, **kwargs)
