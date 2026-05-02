from __future__ import annotations

from typing import Any

import torch
from torch import nn

from ... import optim
from ...optim.step import FISTAStep, GDStep, NesterovStep, OptimizerStep, PolyakStep, ProxGradStep
from ...variable import FlatVar, PairVar, ParamBundle
from ...variable.types import (
    AlgoRawVarType,
    AlgoVar,
    AlgoVarLike,
    IndexMapType,
    MetricSpec,
    MomentumSchedType,
    ParamGradMapType,
    ParamProxMapType,
    ParamSmoothObjType,
    RawParamType,
    ScalarLike,
    StepsizeSchedType,
)
from ...variable.util import as_flat_var, as_pair_var, is_pair_raw_var, is_pair_var


def _as_parameter_payload(u: RawParamType) -> RawParamType:
    if isinstance(u, nn.ParameterList):
        return nn.ParameterList(
            item if isinstance(item, nn.Parameter) else nn.Parameter(item)
            for item in u
        )
    if isinstance(u, tuple):
        return nn.ParameterList(
            item if isinstance(item, nn.Parameter) else nn.Parameter(item)
            for item in u
        )
    return u if isinstance(u, nn.Parameter) else nn.Parameter(u)


def _validate_parametric_grad_inputs(
    smooth_obj_prm: ParamSmoothObjType | None,
    grad_map_prm: ParamGradMapType | None,
) -> None:
    if (smooth_obj_prm is None) == (grad_map_prm is None):
        raise ValueError(
            "Either `grad_map_prm` should be supplied or `smooth_obj_prm`, but not "
            "both. One of them must be set to `None`."
        )


class Optimizer(nn.Module):
    def __init__(
        self,
        step: OptimizerStep,
        u: RawParamType,
        u_indices: IndexMapType | None = None,
        iters: int = 100,
        truncation: int = 0,
        omega: float = 2,
        tol: float = 1e-5,
        metric: MetricSpec | None = None,
        store_history: bool = False,
        verbose: bool = False,
    ):
        super().__init__()
        if truncation < 0:
            raise ValueError("`truncation` must be non-negative.")
        if truncation > iters:
            raise ValueError("`truncation` must not exceed `iters`.")

        self.step = step
        self.u = _as_parameter_payload(u)
        self.u_indices = u_indices
        self.iters = iters
        self.truncation = truncation
        self.omega = omega
        self.tol = tol
        self.metric = metric
        self.store_history = store_history
        self.verbose = verbose
        self.optimizer = optim.Optimizer(
            step,
            tol=tol,
            iters=iters,
            metric=metric,
            store_history=store_history,
            verbose=verbose,
        )
        self._z_init: AlgoVar | None = None
        self._fwd_sol: AlgoVar | None = None

    def _param_bundle(self) -> ParamBundle:
        return ParamBundle(self.u, self.u_indices)

    def _coerce_state(self, z: AlgoVarLike | AlgoRawVarType) -> FlatVar | PairVar:
        return as_pair_var(z) if is_pair_var(z) or is_pair_raw_var(z) else as_flat_var(z)

    def _idle_iters(self) -> int:
        idle_iters = (1 + self.omega) * self.truncation
        if isinstance(idle_iters, float) and not idle_iters.is_integer():
            raise ValueError(
                "`(1 + omega) * truncation` must be an integer-valued iteration count."
            )
        return int(idle_iters)

    @property
    def fwd_init(self) -> FlatVar | PairVar:
        if self._z_init is None:
            raise RuntimeError(
                "Forward initialization is unset. Set `fwd_init` or pass `z_init=...` "
                "to the module call."
            )
        return self._z_init

    @fwd_init.setter
    def fwd_init(self, z: AlgoVarLike | AlgoRawVarType):
        self._z_init = self._coerce_state(z)

    @property
    def fwd_sol(self) -> AlgoVar:
        if self._fwd_sol is None:
            raise RuntimeError("Forward solution accessed before `forward` ran.")
        return self._fwd_sol

    def forward(
        self,
        *args: Any,
        z_init: AlgoVarLike | AlgoRawVarType | None = None,
        **kwargs: Any,
    ) -> FlatVar | PairVar:
        if z_init is not None:
            self.fwd_init = z_init

        _ = self._param_bundle()
        z = self.fwd_init
        idle_iters = self._idle_iters()
        unrolled_iters = self.iters - self.truncation

        with torch.no_grad():
            z_idle = self.optimizer(z, *args, iters=idle_iters, **kwargs)
        z_min = self.optimizer(z_idle, *args, iters=unrolled_iters, **kwargs)
        self._fwd_sol = z_min
        return z_min

    def _make_grad_map(
        self,
        smooth_obj_prm: ParamSmoothObjType | None,
        grad_map_prm: ParamGradMapType | None,
    ):
        _validate_parametric_grad_inputs(smooth_obj_prm, grad_map_prm)
        smooth_grad = (
            None if smooth_obj_prm is None else torch.func.grad(smooth_obj_prm, argnums=0)
        )

        def grad_map(x, *args: Any, **kwargs: Any):
            u_bundle = self._param_bundle()
            if grad_map_prm is not None:
                if u_bundle.takes_params("grad"):
                    return grad_map_prm(x, u_bundle.grad, *args, **kwargs)
                return grad_map_prm(x, *args, **kwargs)
            if u_bundle.takes_params("grad"):
                return smooth_grad(x, u_bundle.grad, *args, **kwargs)
            return smooth_grad(x, *args, **kwargs)

        return grad_map

    def _make_prox_map(self, prox_map_prm: ParamProxMapType):
        def prox_map(x):
            u_bundle = self._param_bundle()
            if u_bundle.takes_params("prox"):
                return prox_map_prm(x, u_bundle.prox)
            return prox_map_prm(x)

        return prox_map


class GradientDescent(Optimizer):
    def __init__(
        self,
        stepsize: ScalarLike,
        u: RawParamType,
        smooth_obj_prm: ParamSmoothObjType | None = None,
        grad_map_prm: ParamGradMapType | None = None,
        u_indices: IndexMapType | None = None,
        tol: float = 1e-5,
        iters: int = 100,
        truncation: int = 0,
        omega: float = 2,
        metric: MetricSpec | None = None,
        store_history: bool = False,
        verbose: bool = False,
        stepsize_scheduler: StepsizeSchedType | None = None,
        linesearch: bool = True,
        tracking: bool = False,
    ):
        step = GDStep(
            stepsize,
            grad_map=self._make_grad_map(smooth_obj_prm, grad_map_prm),
            stepsize_scheduler=stepsize_scheduler,
            linesearch=linesearch,
            tracking=tracking,
        )
        super().__init__(
            step,
            u,
            u_indices=u_indices,
            iters=iters,
            truncation=truncation,
            omega=omega,
            tol=tol,
            metric=metric,
            store_history=store_history,
            verbose=verbose,
        )


class HeavyBall(Optimizer):
    def __init__(
        self,
        stepsize: ScalarLike,
        momentum: ScalarLike,
        u: RawParamType,
        smooth_obj_prm: ParamSmoothObjType | None = None,
        grad_map_prm: ParamGradMapType | None = None,
        u_indices: IndexMapType | None = None,
        tol: float = 1e-5,
        iters: int = 100,
        truncation: int = 0,
        omega: float = 2,
        metric: MetricSpec | None = None,
        store_history: bool = False,
        verbose: bool = False,
        tracking: bool = False,
    ):
        step = PolyakStep(
            stepsize,
            momentum,
            grad_map=self._make_grad_map(smooth_obj_prm, grad_map_prm),
            tracking=tracking,
        )
        super().__init__(
            step,
            u,
            u_indices=u_indices,
            iters=iters,
            truncation=truncation,
            omega=omega,
            tol=tol,
            metric=metric,
            store_history=store_history,
            verbose=verbose,
        )


class Nesterov(Optimizer):
    def __init__(
        self,
        stepsize: ScalarLike,
        u: RawParamType,
        smooth_obj_prm: ParamSmoothObjType | None = None,
        grad_map_prm: ParamGradMapType | None = None,
        u_indices: IndexMapType | None = None,
        tol: float = 1e-5,
        iters: int = 100,
        truncation: int = 0,
        omega: float = 2,
        metric: MetricSpec | None = None,
        store_history: bool = False,
        verbose: bool = False,
        momentum_scheduler: MomentumSchedType | None = None,
        tracking: bool = False,
    ):
        step = NesterovStep(
            stepsize,
            grad_map=self._make_grad_map(smooth_obj_prm, grad_map_prm),
            momentum_scheduler=momentum_scheduler,
            tracking=tracking,
        )
        super().__init__(
            step,
            u,
            u_indices=u_indices,
            iters=iters,
            truncation=truncation,
            omega=omega,
            tol=tol,
            metric=metric,
            store_history=store_history,
            verbose=verbose,
        )


class ProximalGradient(Optimizer):
    def __init__(
        self,
        stepsize: ScalarLike,
        prox_map_prm: ParamProxMapType,
        u: RawParamType,
        smooth_obj_prm: ParamSmoothObjType | None = None,
        grad_map_prm: ParamGradMapType | None = None,
        u_indices: IndexMapType | None = None,
        tol: float = 1e-5,
        iters: int = 100,
        truncation: int = 0,
        omega: float = 2,
        metric: MetricSpec | None = None,
        store_history: bool = False,
        verbose: bool = False,
        tracking: bool = False,
    ):
        step = ProxGradStep(
            stepsize,
            prox_map=self._make_prox_map(prox_map_prm),
            grad_map=self._make_grad_map(smooth_obj_prm, grad_map_prm),
            tracking=tracking,
        )
        super().__init__(
            step,
            u,
            u_indices=u_indices,
            iters=iters,
            truncation=truncation,
            omega=omega,
            tol=tol,
            metric=metric,
            store_history=store_history,
            verbose=verbose,
        )


class FISTA(Optimizer):
    def __init__(
        self,
        stepsize: ScalarLike,
        prox_map_prm: ParamProxMapType,
        u: RawParamType,
        smooth_obj_prm: ParamSmoothObjType | None = None,
        grad_map_prm: ParamGradMapType | None = None,
        u_indices: IndexMapType | None = None,
        tol: float = 1e-5,
        iters: int = 100,
        truncation: int = 0,
        omega: float = 2,
        metric: MetricSpec | None = None,
        store_history: bool = False,
        verbose: bool = False,
        momentum_scheduler: MomentumSchedType | None = None,
        tracking: bool = False,
    ):
        step = FISTAStep(
            stepsize,
            prox_map=self._make_prox_map(prox_map_prm),
            grad_map=self._make_grad_map(smooth_obj_prm, grad_map_prm),
            momentum_scheduler=momentum_scheduler,
            tracking=tracking,
        )
        super().__init__(
            step,
            u,
            u_indices=u_indices,
            iters=iters,
            truncation=truncation,
            omega=omega,
            tol=tol,
            metric=metric,
            store_history=store_history,
            verbose=verbose,
        )
