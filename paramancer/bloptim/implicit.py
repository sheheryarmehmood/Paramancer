from __future__ import annotations

from collections.abc import Callable

import torch

from ..optim.step import OptimizerStep

from ..optim.optimizer import NeumannSeries, Optimizer
from ..variable.util import zeros_like
from ..variable.flat import FlatVar
from ..variable.pair import PairVar
from ..variable.parameter import AlgoParam
from ..variable.types import (
    FlattendType, LinOpType, MetricSpec, ParamMarkovStep, AlgoVar, AlgoVarLike,
    VSpecType, PSpecType, IndexMapType
)
from ..variable.util import unflatten_raw

def unflatten_var(z_flat: FlattendType, x_spec: VSpecType) -> AlgoVar:
    z_raw = unflatten_raw(z_flat, x_spec)
    return PairVar(z_raw) if x_spec[0] == "pair" else FlatVar(z_raw)

def unflatten_param(
    u_flat: FlattendType, u_spec: PSpecType, indices: IndexMapType
) -> AlgoParam:
    u_raw = unflatten_raw(u_flat, u_spec)
    return AlgoParam(u_raw, indices)

def extend_if_not_markovian(
    x: AlgoVar | None,
    fn: Callable[[FlatVar], FlatVar],
    is_markovian: bool
) -> AlgoVar | None:
    return x if is_markovian or x is None else PairVar(x, fn(x))

def project_if_not_markovian(
    x: AlgoVar,
    is_markovian: bool
) -> AlgoVar:
    return x if is_markovian else x.first


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


class VJP:
    def __init__(
        self,
        param_step: ParamMarkovStep,
        tol: float = 1e-5,
        iters: int = 100,
        metric: MetricSpec | None = None,
        verbose: bool = False
    ):
        self.step = param_step
        self._lin_sol = None
        self.tol = tol
        self.iters = iters
        self.metric = metric
        self.verbose = verbose
    
    def __call__(
        self,
        z_root: AlgoVar,
        u_given: AlgoParam,
        z_grad: AlgoVar,
        init: AlgoVar = None,
        *args,
        **kwargs
    ) -> AlgoParam:
        self._lin_sol = neumann_series(
            lambda v: self.step.vjp_var(z_root, u_given, v), z_grad,
            init=init, tol=self.tol, iters=self.iters,
            metric=self.metric, verbose=self.verbose
        )
        return self.step.vjp_par(z_root, u_given, self._lin_sol)
    
    @property
    def lin_sol(self):
        return self._lin_sol

class JVP:
    def __init__(
        self,
        param_step: ParamMarkovStep,
        tol: float = 1e-5,
        iters: int = 100,
        metric: MetricSpec | None = None,
        verbose: bool = False
    ):
        self.step = param_step
        self._lin_sol = None
        self.tol = tol
        self.iters = iters
        self.metric = metric
        self.verbose = verbose
    
    def __call__(
        self,
        z_root: AlgoVar,
        u_given: AlgoParam,
        u_tan: AlgoParam,
        init: AlgoVar | None = None,
        *args,
        **kwargs
    ) -> AlgoVar:
        jvp_par_tan = self.step.jvp_par(z_root, u_given, u_tan)
        self._lin_sol = neumann_series(
            lambda v: self.step.jvp_var(z_root, u_given, v), jvp_par_tan,
            init=init, tol=self.tol, iters=self.iters,
            metric=self.metric, verbose=self.verbose
        )
        return self._lin_sol
    
    @property
    def lin_sol(self):
        return self._lin_sol

def _flat_len(spec: PSpecType | VSpecType) -> int:
    if spec[0] == "tensor":
        return 1
    if spec[0] == "tuple":
        return spec[1]
    if spec[0] == "pair":
        return spec[1] + spec[2]
    raise TypeError(f"Unsupported spec {spec}.")


def _split_flat_args(
    flat_args: tuple[torch.Tensor, ...],
    u_spec: PSpecType,
    z_spec: VSpecType,
) -> tuple[FlattendType, FlattendType, FlattendType | None]:
    num_u = _flat_len(u_spec)
    num_z = _flat_len(z_spec)
    base_count = num_u + num_z
    total_count = len(flat_args)

    if total_count == base_count:
        has_z_adj_init = False
    elif total_count == base_count + num_z:
        has_z_adj_init = True
    else:
        raise ValueError(
            "Expected flattened inputs to contain parameter tensors, initial "
            "state tensors, and optionally adjoint-initialization tensors."
        )

    u_given_fl = flat_args[:num_u]
    z_init_fl = flat_args[num_u:base_count]
    z_adj_init_fl = flat_args[base_count:] if has_z_adj_init else None
    return u_given_fl, z_init_fl, z_adj_init_fl


class OptimizerID(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        pm_step: ParamMarkovStep,
        u_spec: PSpecType,
        z_spec: VSpecType,
        u_indices: IndexMapType,
        tol: float = 1e-5,
        iters: int = 100,
        metric: MetricSpec | None = None,
        verbose: bool = False,
        *flat_args: torch.Tensor,
    ):
        u_given_fl, z_init_fl, z_adj_init_fl = _split_flat_args(
            flat_args, u_spec, z_spec
        )
        is_markovian = pm_step.is_markovian()
        z_init = extend_if_not_markovian(
            unflatten_var(z_init_fl, z_spec), lambda x: x, is_markovian
        )
        u_given = unflatten_param(u_given_fl, u_spec, u_indices)
        pm_step.u_given = u_given
        z_min = Optimizer(
            pm_step, tol=tol, iters=iters, metric=metric, verbose=verbose
        )(z_init)
        z_min_fl, _ = project_if_not_markovian(z_min, is_markovian).flatten()

        tensors_to_save = [*z_min_fl, *u_given_fl]
        if z_adj_init_fl is not None:
            tensors_to_save.extend(z_adj_init_fl)
        ctx.save_for_backward(*tensors_to_save)
        ctx.pm_step = pm_step
        ctx.u_spec = u_spec
        ctx.z_spec = z_spec
        ctx.u_indices = u_indices
        ctx.tol = tol
        ctx.iters = iters
        ctx.metric = metric
        ctx.verbose = verbose
        ctx.has_z_adj_init = z_adj_init_fl is not None

        return z_min_fl[0] if len(z_min_fl) == 1 else z_min_fl

    @staticmethod
    def backward(ctx, *z_adj_fl):
        if len(z_adj_fl) == 1:
            z_adj_fl = (z_adj_fl[0],)

        pm_step = ctx.pm_step
        is_markovian = pm_step.is_markovian()
        num_z = _flat_len(ctx.z_spec)
        num_u = _flat_len(ctx.u_spec)
        saved_tensors = ctx.saved_tensors
        z_root_fl = saved_tensors[:num_z]
        u_given_fl = saved_tensors[num_z : num_z + num_u]
        z_adj_init_fl = (
            saved_tensors[num_z + num_u :] if ctx.has_z_adj_init else None
        )
        z_root = extend_if_not_markovian(
            unflatten_var(z_root_fl, ctx.z_spec), lambda x: x, is_markovian
        )
        u_given = unflatten_param(u_given_fl, ctx.u_spec, ctx.u_indices)
        z_adj = extend_if_not_markovian(
            unflatten_var(z_adj_fl, ctx.z_spec),
            lambda x: x.zeros_like(),
            is_markovian,
        )

        init = None
        if z_adj_init_fl is not None:
            z_adj_raw = unflatten_var(z_adj_fl, ctx.z_spec)

            def fn(x: FlatVar) -> FlatVar:
                if is_markovian:
                    return x
                z_adj_init = extend_if_not_markovian(
                    x, lambda y: y.zeros_like(), is_markovian
                )
                return z_adj_raw - x + pm_step.vjp_var(z_root, u_given, z_adj_init)

            init = extend_if_not_markovian(
                unflatten_var(z_adj_init_fl, ctx.z_spec), fn, is_markovian
            )

        u_grad_fl, _ = VJP(
            pm_step,
            tol=ctx.tol,
            iters=ctx.iters,
            metric=ctx.metric,
            verbose=ctx.verbose,
        )(z_root, u_given, z_adj, init=init).flatten()

        metadata_grads = (None,) * 8
        z_init_grads = (None,) * num_z
        z_adj_init_grads = (None,) * num_z if ctx.has_z_adj_init else ()
        return (
            *metadata_grads,
            *u_grad_fl,
            *z_init_grads,
            *z_adj_init_grads,
        )



