from __future__ import annotations

from typing import Any

import torch
from torch import nn

from .diff import OptimizerID
from .step import (
    FISTAParamMarkovStep,
    GDParamMarkovStep,
    NesterovParamMarkovStep,
    PolyakParamMarkovStep,
    ProxGradParamMarkovStep,
)
from ...variable import FlatVar, PairVar, ParamBundle
from ...variable.types import (
    AlgoRawVarType,
    AlgoVar,
    AlgoVarLike,
    IndexMapType,
    MetricSpec,
    ParamGradMapType,
    ParamMarkovStep,
    ParamProxMapType,
    ParamSmoothObjType,
    RawParamType,
    ScalarLike,
    StepsizeSchedType,
    MomentumSchedType,
)
from ...variable.util import as_flat_var, is_pair_raw_var, is_pair_var


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


class Optimizer(nn.Module):
    def __init__(
        self,
        pm_step: ParamMarkovStep,
        u: RawParamType,
        u_indices: IndexMapType | None = None,
        tol: float = 1e-5,
        iters: int = 100,
        metric: MetricSpec | None = None,
        verbose: bool = False,
    ):
        super().__init__()
        self.pm_step = pm_step
        self.u = _as_parameter_payload(u)
        self.u_indices = u_indices
        self.tol = tol
        self.iters = iters
        self.metric = metric
        self.verbose = verbose
        self._z_init: FlatVar | None = None
        self._z_adj_init: FlatVar | None = None
        self._fwd_sol: AlgoVar | None = None

    def _param_bundle(self) -> ParamBundle:
        return ParamBundle(self.u, self.u_indices)

    def _coerce_state(self, z: AlgoVarLike | AlgoRawVarType) -> FlatVar:
        # FIXIT: Later, when PDHG is also implemented, we may want to allow pair states here. For now, we require a flat state for simplicity.
        if is_pair_var(z) or is_pair_raw_var(z):
            raise TypeError(
                "This implicit optimizer expects a flat state. Pair states "
                "are reserved for pair-valued methods such as PDHG."
            )
        return as_flat_var(z)

    @property
    def fwd_init(self) -> FlatVar:
        if self._z_init is None:
            raise RuntimeError(
                "Forward initialization is unset. Set `fwd_init` or pass "
                "`z_init=...` to the module call."
            )
        return self._z_init

    @fwd_init.setter
    def fwd_init(self, z: AlgoVarLike | AlgoRawVarType):
        self._z_init = self._coerce_state(z)

    @property
    def bwd_init(self) -> FlatVar | None:
        return self._z_adj_init

    @bwd_init.setter
    def bwd_init(self, z: AlgoVarLike | AlgoRawVarType | None):
        self._z_adj_init = None if z is None else self._coerce_state(z)

    @property
    def fwd_sol(self) -> AlgoVar:
        if self._fwd_sol is None:
            raise RuntimeError("Forward solution accessed before `forward` ran.")
        return self._fwd_sol

    @property
    def bwd_sol(self) -> AlgoVar:
        if not hasattr(self.pm_step, "_adjoint_state"):
            raise RuntimeError(
                "Backward solution accessed before any backward pass ran."
            )
        return self.pm_step._adjoint_state

    def forward(
        self,
        *args: Any,
        z_init: AlgoVarLike | AlgoRawVarType | None = None,
        z_adj_init: AlgoVarLike | AlgoRawVarType | None = None,
        **kwargs: Any,
    ) -> FlatVar | PairVar:
        if z_init is not None:
            self.fwd_init = z_init
        if z_adj_init is not None:
            self.bwd_init = z_adj_init

        u_bundle = self._param_bundle()
        u_flat, u_spec = u_bundle.flatten()
        z_init_flat, z_spec = self.fwd_init.flatten()
        z_adj_init_flat = ()
        if self.bwd_init is not None:
            z_adj_init_flat, _ = self.bwd_init.flatten()

        self.pm_step._optimizer_call_args = args
        self.pm_step._optimizer_call_kwargs = kwargs
        z_min_flat = OptimizerID.apply(
            self.pm_step,
            u_spec,
            z_spec,
            u_bundle.indices,
            self.tol,
            self.iters,
            self.metric,
            self.verbose,
            *u_flat,
            *z_init_flat,
            *z_adj_init_flat,
        )
        if isinstance(z_min_flat, torch.Tensor):
            z_min_flat = (z_min_flat,)

        Var = PairVar if z_spec[0] == "pair" else FlatVar
        self._fwd_sol = Var.unflatten(z_min_flat, z_spec)
        return self._fwd_sol


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
        metric: MetricSpec | None = None,
        verbose: bool = False,
        stepsize_scheduler: StepsizeSchedType | None = None,
        linesearch: bool = True,
        tracking: bool = False,
    ):
        pm_step = GDParamMarkovStep(
            stepsize,
            grad_map_prm=grad_map_prm,
            smooth_obj_prm=smooth_obj_prm,
            stepsize_scheduler=stepsize_scheduler,
            linesearch=linesearch,
            tracking=tracking,
        )
        super().__init__(pm_step, u, u_indices, tol, iters, metric, verbose)


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
        metric: MetricSpec | None = None,
        verbose: bool = False,
        tracking: bool = False,
    ):
        pm_step = PolyakParamMarkovStep(
            stepsize,
            momentum,
            grad_map_prm=grad_map_prm,
            smooth_obj_prm=smooth_obj_prm,
            tracking=tracking,
        )
        super().__init__(pm_step, u, u_indices, tol, iters, metric, verbose)


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
        metric: MetricSpec | None = None,
        verbose: bool = False,
        momentum_scheduler: MomentumSchedType | None = None,
        tracking: bool = False,
    ):
        pm_step = NesterovParamMarkovStep(
            stepsize,
            grad_map_prm=grad_map_prm,
            smooth_obj_prm=smooth_obj_prm,
            momentum_scheduler=momentum_scheduler,
            tracking=tracking,
        )
        super().__init__(pm_step, u, u_indices, tol, iters, metric, verbose)


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
        metric: MetricSpec | None = None,
        verbose: bool = False,
        tracking: bool = False,
    ):
        pm_step = ProxGradParamMarkovStep(
            stepsize,
            prox_map_prm,
            grad_map_prm=grad_map_prm,
            smooth_obj_prm=smooth_obj_prm,
            tracking=tracking,
        )
        super().__init__(pm_step, u, u_indices, tol, iters, metric, verbose)


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
        metric: MetricSpec | None = None,
        verbose: bool = False,
        momentum_scheduler: MomentumSchedType | None = None,
        tracking: bool = False,
    ):
        pm_step = FISTAParamMarkovStep(
            stepsize,
            prox_map_prm,
            grad_map_prm=grad_map_prm,
            smooth_obj_prm=smooth_obj_prm,
            momentum_scheduler=momentum_scheduler,
            tracking=tracking,
        )
        super().__init__(pm_step, u, u_indices, tol, iters, metric, verbose)
