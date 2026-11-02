from collections.abc import Callable
from dataclasses import dataclass
from typing import Concatenate, ParamSpec, TypeAlias

import torch

from ...optim.optimizer import NeumannSeries, Optimizer
from ...optim.step import OptimizerStep
from ...variable import (
    AlgoVar,
    FlattendType,
    IndexMapType,
    LinOpType,
    MetricSpec,
    ParamBundle,
    ParamGradMapType,
    ParamLinOpType,
    ParamProxMapType,
    PSpecType,
    VSpecType,
)
from ...variable.flat import FlatVar
from ...variable.pair import PairVar

P = ParamSpec("P")
FlatStep: TypeAlias = Callable[Concatenate[FlatVar, P], FlatVar]
PairStep: TypeAlias = Callable[Concatenate[PairVar, P], PairVar]
MarkovStep: TypeAlias = FlatStep | PairStep

ParamFlatStep: TypeAlias = Callable[
    Concatenate[FlatVar, ParamBundle, P], FlatVar
]
ParamPairStep: TypeAlias = Callable[
    Concatenate[PairVar, ParamBundle, P], PairVar
]
ParamMarkovStep: TypeAlias = ParamFlatStep | ParamPairStep

FlattenedParamStep: TypeAlias = Callable[
    [FlattendType, FlattendType], FlattendType
]


def markovify(step: OptimizerStep) -> MarkovStep:
    if step.is_markovian():
        return step
    def markovian_step(z_curr: PairVar, *args, **kwds) -> PairVar:
        x_next = step._transition(
            z_curr.first,
            z_curr.second,
            *args,
            **kwds,
        )
        return PairVar(x_next, z_curr.first)
    return markovian_step


def parameterize(
    markovian_step: MarkovStep,
    optim_step: OptimizerStep,
    grad_map: ParamGradMapType | None,
    prox_map: ParamProxMapType | None,
    prox_map_dual: ParamProxMapType | None,
    lin_op: ParamLinOpType | None,
    lin_op_adj: ParamLinOpType | None,
) -> ParamMarkovStep:    
    def parameterized_step(
        z_curr: AlgoVar, outer_vars: ParamBundle, *args, **kwds
    ) -> AlgoVar:
        def bound_grad_map(x, *args, **kwds):
            return grad_map(x, outer_vars.grad, *args, **kwds)
        def bound_prox_map(x):
            return prox_map(x, outer_vars.prox)
        def bound_prox_map_primal(x):
            return prox_map(x, outer_vars.primal)
        def bound_prox_map_dual(x):
            return prox_map_dual(x, outer_vars.dual)
        def bound_lin_op(x):
            return lin_op(x, outer_vars.lin_op)
        def bound_lin_op_adj(x):
            return lin_op_adj(x, outer_vars.lin_op)
        
        orig_grad_map = None
        orig_prox_map = None
        orig_prox_map_primal = None
        orig_prox_map_dual = None
        orig_lin_op = None
        orig_lin_op_adj = None
        
        try:
            if grad_map is not None:
                orig_grad_map = optim_step.grad_map
                optim_step.grad_map = bound_grad_map
            if lin_op is not None:
                orig_prox_map_primal = optim_step.prox_map_primal
                orig_prox_map_dual = optim_step.prox_map_dual
                orig_lin_op = optim_step.lin_op
                orig_lin_op_adj = optim_step.lin_op_adj
                optim_step.prox_map_primal = bound_prox_map_primal
                optim_step.prox_map_dual = bound_prox_map_dual
                optim_step.replace_lin_op(bound_lin_op, bound_lin_op_adj)
            elif prox_map is not None:
                orig_prox_map = optim_step.prox_map
                optim_step.prox_map = bound_prox_map
            return markovian_step(z_curr, *args, **kwds)
        finally:
            if orig_grad_map is not None:
                optim_step.grad_map = orig_grad_map
            if orig_lin_op is not None:
                optim_step.prox_map_primal = orig_prox_map_primal
                optim_step.prox_map_dual = orig_prox_map_dual
                optim_step.replace_lin_op(orig_lin_op, orig_lin_op_adj)
            if orig_prox_map is not None:
                optim_step.prox_map = orig_prox_map
    
    return parameterized_step


def prepare_flat_param_step(
    parameterized_markovian_step: ParamMarkovStep,
    z_spec: VSpecType,
    z_wrapper_type: type[FlatVar] | type[PairVar],
    u_spec: PSpecType,
    u_indices: IndexMapType,
    *args,
    **kwds,
) -> FlattenedParamStep:
    def flattened_parameterized_markovian_step(
        z_flat: FlattendType,
        u_flat: FlattendType,
    ) -> FlattendType:
        z = z_wrapper_type.unflatten(z_flat, z_spec)
        u = ParamBundle.unflatten(u_flat, u_spec, u_indices)
        return parameterized_markovian_step(z, u, *args, **kwds).flatten()[0]
    return flattened_parameterized_markovian_step


def neumann_series(
    lin_op: LinOpType,
    vector: AlgoVar,
    init: AlgoVar | None = None,
    tol: float = 1e-5,
    iters: int = 100,
    metric: MetricSpec | None = None,
    store_history: bool = False,
    verbose: bool = False,
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
        verbose: bool = False,
    ):
        self._param_step = param_step
        self.tol = tol
        self.iters = iters
        self.metric = metric
        self.verbose = verbose
        self._adjoint: AlgoVar | None = None

    def __call__(
        self,
        z_root: AlgoVar,
        u_given: ParamBundle,
        z_grad: AlgoVar,
        init: AlgoVar | None = None,
        *args,
        **kwds,
    ) -> ParamBundle:
        specs = self._create_pullback(z_root, u_given, *args, **kwds)
        vjp_var = lambda z_grad: self._call_pullback(z_grad, *specs)[0]
        vjp_par = lambda z_grad: self._call_pullback(z_grad, *specs)[1]

        self._adjoint = neumann_series(
            vjp_var,
            z_grad,
            init=init,
            tol=self.tol,
            iters=self.iters,
            metric=self.metric,
            verbose=self.verbose,
        )
        return vjp_par(self._adjoint)

    def _call_pullback(
        self,
        z_grad: AlgoVar,
        pullback: Callable[
            [FlattendType], tuple[FlattendType, FlattendType]
        ],
        z_spec: VSpecType,
        z_wrapper_type: type[FlatVar] | type[PairVar],
        u_spec: PSpecType,
        u_indices: IndexMapType,
    ) -> tuple[AlgoVar, ParamBundle]:
        zg_flat, _ = z_grad.flatten()
        zrg_flat, ug_flat = pullback(zg_flat)
        zr_grad = z_wrapper_type.unflatten(zrg_flat, z_spec)
        u_grad = ParamBundle.unflatten(ug_flat, u_spec, u_indices)
        return zr_grad, u_grad

    def _create_pullback(
        self,
        z_root: AlgoVar,
        u_given: ParamBundle,
        *args,
        **kwds,
    ) -> tuple[
        Callable[[FlattendType], tuple[FlattendType, FlattendType]],
        VSpecType,
        type[FlatVar] | type[PairVar],
        PSpecType,
        IndexMapType,
    ]:
        z_flat, z_spec = z_root.flatten()
        z_wrapper_type = type(z_root)
        u_flat, u_spec = u_given.flatten()
        u_indices = u_given.indices
        flat_param_step = prepare_flat_param_step(
            self._param_step,
            z_spec,
            z_wrapper_type,
            u_spec,
            u_indices,
            *args,
            **kwds,
        )
        pullback = torch.func.vjp(flat_param_step, z_flat, u_flat)[1]
        return pullback, z_spec, z_wrapper_type, u_spec, u_indices

    @property
    def adjoint_state(self) -> AlgoVar:
        if self._adjoint is None:
            raise RuntimeError(
                "Adjoint state accessed before implicit differentiation ran."
            )
        return self._adjoint


def _flat_len(spec: PSpecType | VSpecType) -> int:
    if spec[0] == "tensor":
        return 1
    if spec[0] == "tuple":
        return spec[1]
    if spec[0] == "pair":
        return _flat_len(spec[1]) + _flat_len(spec[2])
    raise TypeError(f"Unsupported spec {spec}.")


def _split_flat_args(
    flat_args: tuple[torch.Tensor, ...],
    u_spec: PSpecType,
    z_spec: VSpecType,
    is_markovian: bool,
) -> tuple[FlattendType, FlattendType, FlattendType | None]:
    num_u = _flat_len(u_spec)
    num_z = _flat_len(z_spec)
    num_z_adj = num_z if is_markovian else 2 * num_z
    base_count = num_u + num_z
    total_count = len(flat_args)

    if total_count == base_count:
        has_z_adj_init = False
    elif total_count == base_count + num_z_adj:
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

def lift_fixed_point(x_root: AlgoVar, is_markovian: bool) -> AlgoVar:
    return x_root if is_markovian else PairVar(x_root, x_root)

def lift_output_gradient(x_grad: AlgoVar, is_markovian: bool) -> AlgoVar:
    return x_grad if is_markovian else PairVar(x_grad, x_grad.zeros_like())


@dataclass
class AdjointStateCache:
    value: AlgoVar | None = None


class OptimizerID(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        step: OptimizerStep,
        grad_map: ParamGradMapType | None,
        prox_map: ParamProxMapType | None,
        prox_map_dual: ParamProxMapType | None,
        lin_op: ParamLinOpType | None,
        lin_op_adj: ParamLinOpType | None,
        u_spec: PSpecType,
        u_indices: IndexMapType,
        z_spec: VSpecType,
        z_wrapper_type: type[FlatVar] | type[PairVar],
        call_args: tuple,
        call_kwargs: dict,
        tol: float,
        iters: int,
        metric: MetricSpec | None,
        verbose: bool,
        adjoint_cache: AdjointStateCache | None,
        *flat_args: torch.Tensor,
    ) -> tuple[torch.Tensor, ...] | torch.Tensor:
        u_given_fl, z_init_fl, z_adj_init_fl = _split_flat_args(
            flat_args, u_spec, z_spec, step.is_markovian()
        )
        z_init = z_wrapper_type.unflatten(z_init_fl, z_spec)
        optimizer = Optimizer(
            step, tol=tol, iters=iters, metric=metric, verbose=verbose
        )
        z_min = optimizer(z_init, *call_args, **call_kwargs)
        z_min_fl, _ = z_min.flatten()

        tensors_to_save = [*z_min_fl, *u_given_fl]
        if z_adj_init_fl is not None:
            tensors_to_save.extend(z_adj_init_fl)
        ctx.save_for_backward(*tensors_to_save)
        ctx.step = step
        ctx.grad_map = grad_map
        ctx.prox_map = prox_map
        ctx.prox_map_dual = prox_map_dual
        ctx.lin_op = lin_op
        ctx.lin_op_adj = lin_op_adj
        ctx.u_spec = u_spec
        ctx.u_indices = u_indices
        ctx.z_spec = z_spec
        ctx.z_wrapper_type = z_wrapper_type
        ctx.call_args = call_args
        ctx.call_kwargs = call_kwargs
        ctx.tol = tol
        ctx.iters = iters
        ctx.metric = metric
        ctx.verbose = verbose
        ctx.has_z_adj_init = z_adj_init_fl is not None
        ctx.adjoint_cache = adjoint_cache

        return z_min_fl[0] if len(z_min_fl) == 1 else z_min_fl

    @staticmethod
    def backward(ctx, *z_grad_fl):
        z_spec = ctx.z_spec
        num_u = _flat_len(ctx.u_spec)
        num_z = _flat_len(z_spec)
        is_markovian = ctx.step.is_markovian()
        z_adj_spec = z_spec if is_markovian else ("pair", z_spec, z_spec)
        z_adj_wrapper = ctx.z_wrapper_type if is_markovian else PairVar
        num_z_adj = num_z if is_markovian else 2 * num_z
        
        u_given = ParamBundle.unflatten(
            ctx.saved_tensors[num_z : num_z + num_u],
            ctx.u_spec,
            ctx.u_indices,
        )
        z_root = ctx.z_wrapper_type.unflatten(
            ctx.saved_tensors[:num_z],
            z_spec,
        )
        z_grad = ctx.z_wrapper_type.unflatten(z_grad_fl, z_spec)
        z_adj_init = (
            z_adj_wrapper.unflatten(
                ctx.saved_tensors[num_z + num_u :],
                z_adj_spec,
            )
            if ctx.has_z_adj_init
            else None
        )
        
        z_root_lifted = lift_fixed_point(z_root, is_markovian)
        z_grad_lifted = lift_output_gradient(z_grad, is_markovian)
        
        grad_map = ctx.grad_map
        prox_map = ctx.prox_map
        prox_map_dual = ctx.prox_map_dual
        lin_op = ctx.lin_op
        lin_op_adj = ctx.lin_op_adj
        m_step = markovify(ctx.step)
        pm_step = parameterize(
            m_step,
            ctx.step,
            grad_map,
            prox_map,
            prox_map_dual,
            lin_op,
            lin_op_adj
        )
        
        vjp = VJP(
            pm_step,
            tol=ctx.tol,
            iters=ctx.iters,
            metric=ctx.metric,
            verbose=ctx.verbose,
        )
        u_grad_fl, _ = vjp(
            z_root_lifted,
            u_given,
            z_grad_lifted,
            z_adj_init,
            *ctx.call_args,
            **ctx.call_kwargs,
        ).flatten()
        if ctx.adjoint_cache is not None:
            ctx.adjoint_cache.value = vjp.adjoint_state

        metadata_grads = (None,) * 17
        z_init_grads = (None,) * num_z
        z_adj_init_grads = (None,) * num_z_adj if ctx.has_z_adj_init else ()
        return (
            *metadata_grads,
            *u_grad_fl,
            *z_init_grads,
            *z_adj_init_grads,
        )



class ImplicitBackend:
    def __init__(
        self,
        optimizer: Optimizer,
        init: AlgoVar,
        outer_vars: ParamBundle,
        grad_map: ParamGradMapType | None = None,
        prox_map: ParamProxMapType | None = None,
        lin_op: ParamLinOpType | None = None,
        lin_op_adj: ParamLinOpType | None = None,
    ):
        self._optimizer = optimizer
        self._outer_vars = outer_vars
        self._grad_map = grad_map
        self._prox_map = prox_map
        self._lin_op = lin_op
        self._lin_op_adj = lin_op_adj
    
    def __call__(
        self,
        init: AlgoVar,
        *args,
        **kwds,
    ):
        if self._grad_map is None:
            self._optimizer.step.grad_map = self._grad_map
        if self._prox_map is None:
            self._optimizer.step.prox_map = self._prox_map
        if self._lin_op is None:
            self._optimizer.step.replace_lin_op(
                self._lin_op,
                self._lin_op_adj,
            )
