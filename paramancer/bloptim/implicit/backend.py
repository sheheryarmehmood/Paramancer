from ...optim.optimizer import Optimizer
from ...optim.step import OptimizerStep
from ...variable import (
    AlgoVar,
    FlattendType,
    IndexMapType,
    ParamBundle,
    ParamGradMapType,
    ParamLinOpType,
    ParamProxMapType,
    PSpecType,
    VSpecType,
)
from ...variable.flat import FlatVar
from ...variable.pair import PairVar


def markovify(step: OptimizerStep):
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
    markovian_step,
    optim_step: OptimizerStep,
    grad_map: ParamGradMapType | None,
    prox_map: ParamProxMapType | None,
    prox_map_dual: ParamProxMapType | None,
    lin_op: ParamLinOpType | None,
    lin_op_adj: ParamLinOpType | None,
):    
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
    parameterized_markovian_step,
    z_spec: VSpecType,
    z_wrapper_type: type[FlatVar] | type[PairVar],
    u_spec: PSpecType,
    u_indices: IndexMapType,
    *args,
    **kwds,
):
    def flattened_parameterized_markovian_step(
        z_flat: FlattendType,
        u_flat: FlattendType,
    ):
        z = z_wrapper_type.unflatten(z_flat, z_spec)
        u = ParamBundle.unflatten(u_flat, u_spec, u_indices)
        return parameterized_markovian_step(z, u, *args, **kwds).flatten()[0]
    return flattened_parameterized_markovian_step


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
