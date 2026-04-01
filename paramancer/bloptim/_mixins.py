from __future__ import annotations
import torch

from ..optim.util import ensure_var_input
from ..operators.grad import gradient
from ..variable import Variable, ParameterBundle
from ..variable.util import vlatten, unvlatten, platten, unplatten
from ..variable.types import (
    IndexMapType, VariableLike,
    ParameterLike, ParameterType,
    FlattendType, VSpecType, PSpecType, P,
    ParamSmoothObjType, ParamGradMapType, ParamProxMapType
)


class ParamMarkovStepMixin:
    def __init__(
        self,
        u_in: ParameterLike | None = None,
        indices: IndexMapType | None = None
    ):
        self._is_input_parambundle = isinstance(u_in, ParameterBundle)
        if self._is_input_parambundle:
            self._u_given = u_in
        else:
            # vvvv If `u_in` is `None`, initialize a dummy `ParameterBundle`.
            self._u_given = ParameterBundle(u_in, indices=indices)
    
    @ensure_var_input
    def step(
        self,
        x_curr: Variable,
        u_in: ParameterLike | None = None,
        *args: P.args,
        **kwargs: P.kwargs
    ) -> Variable:
        if not self.is_markovian():
            self.x_prev = x_curr.previous
            x_curr = x_curr.current
        self.u_given = u_in
        x_new = super().step(x_curr, *args, **kwargs)
        if not self.is_markovian():
            x_new = Variable.from_momentum(x_new, x_curr)
        return x_new
    
    def __call__(
        self,
        x_curr: VariableLike,
        u_given: ParameterLike | None = None,
        *args: P.args, 
        **kwargs: P.kwargs
    ) -> VariableLike:
        return self.step(x_curr, u_given, *args, **kwargs)
    
    @property
    def u_given(self) -> ParameterLike:
        if self._u_given is None:
            raise RuntimeError(
                "Parameter accessed without setting u_given"
            )
        return (
            self._u_given if self._is_input_parambundle
            else self._u_given.data
        )
    
    @u_given.setter
    def u_given(self, u_in: ParameterLike | None):
        """Ignores the incoming `u_in` when it is `None`"""
        if isinstance(u_in, ParameterBundle):
            u_in = u_in.data
        if u_in is None or u_in is self._u_given.data:
            return      # Don't update if same or no parameter was given.
        else:
            self._u_given.data = u_in

def flatten_inputs(
    step,
    x_spec: VSpecType,
    u_spec: PSpecType,
    *args: P.args,
    **kwargs: P.kwargs
):
    num_x = sum(x_spec[1:]) if len(x_spec) > 1 else 1
    def step_flat(*vars_and_pars: torch.Tensor) -> FlattendType:
        x = unvlatten(vars_and_pars[:num_x], x_spec)
        u = unplatten(vars_and_pars[num_x:], u_spec)
        out = step(x, u, *args, **kwargs)
        return vlatten(out.data if isinstance(out, Variable) else out)[0]
    return step_flat


class JVPMixin:
    def jvp(
        self,
        x_in: VariableLike,
        u_in: ParameterLike,
        x_tan: VariableLike,
        u_tan: ParameterType,
        *args: P.args,
        **kwargs: P.kwargs
    ) -> VariableLike:
        x_is_var = isinstance(x_in, Variable)
        u_is_par = isinstance(u_in, ParameterBundle)
        x_flat, x_spec = vlatten(x_in.data if x_is_var else x_in)
        u_flat, u_spec = platten(u_in.data if u_is_par else u_in)
        x_tan_flat, _ = vlatten(x_tan.data if x_is_var else x_tan)
        u_tan_flat, _ = platten(u_tan.data if u_is_par else u_in)
        step = flatten_inputs(self.step, x_spec, u_spec, *args, **kwargs)
        _, out_tan_flat = torch.func.jvp(
            step, (*x_flat, *u_flat), (*x_tan_flat, *u_tan_flat)
        )
        out_tan = unvlatten(out_tan_flat, x_spec)
        return Variable(out_tan) if x_is_var else out_tan
    
    def jvp_var(
        self,
        x_in: VariableLike,
        u_in: ParameterLike,
        x_tan: VariableLike,
        *args: P.args,
        **kwargs: P.kwargs
    ) -> VariableLike:
        u_is_par = isinstance(u_in, ParameterBundle)
        u_par = u_in if u_is_par else ParameterBundle(u_in)
        u_par_tan = u_par.zeros_like()
        u_tan = u_par_tan if u_is_par else u_par_tan.data
        return self.jvp(x_in, u_in, x_tan, u_tan, *args, **kwargs)
    
    def jvp_par(
        self,
        x_in: VariableLike,
        u_in: ParameterLike,
        u_tan: ParameterType,
        *args: P.args,
        **kwargs: P.kwargs
    ) -> VariableLike:
        x_is_var = isinstance(x_in, Variable)
        x_var = x_in if x_is_var else Variable(x_in)
        x_var_tan = x_var.zeros_like()
        x_tan = x_var_tan if x_is_var else x_var_tan.data
        return self.jvp(x_in, u_in, x_tan, u_tan, *args, **kwargs)


class VJPMixin:
    def vjp(
        self,
        x_in: VariableLike,
        u_in: ParameterLike,
        grad_out: VariableLike,
        *args: P.args,
        **kwargs: P.kwargs
    ) -> tuple[VariableLike, ParameterLike]:
        x_is_var = isinstance(x_in, Variable)
        u_is_par = isinstance(u_in, ParameterBundle)
        x_flat, x_spec = vlatten(x_in.data if x_is_var else x_in)
        u_flat, u_spec = platten(u_in.data if u_is_par else u_in)
        grad_out_flat, _ = vlatten(grad_out.data if x_is_var else grad_out)
        step = flatten_inputs(self.step, x_spec, u_spec, *args, **kwargs)
        (_, vjp) = torch.func.vjp(step, *x_flat, *u_flat)
        grad_in_flat = vjp(grad_out_flat)
        grad_x = unvlatten(grad_in_flat[:len(x_flat)], x_spec)
        grad_u = unvlatten(grad_in_flat[len(x_flat):], u_spec)
        return (
            Variable(grad_x) if x_is_var else grad_x,
            ParameterBundle(grad_u, u_in.indices) if u_is_par else grad_u
        )
    
    def vjp_var(
        self,
        x_in: VariableLike,
        u_in: ParameterLike,
        grad_out: VariableLike,
        *args: P.args,
        **kwargs: P.kwargs
    ) -> VariableLike:
        return self.vjp(x_in, u_in, grad_out, *args, **kwargs)[0]
    
    def vjp_par(
        self,
        x_in: VariableLike,
        u_in: ParameterLike,
        grad_out: VariableLike,
        *args: P.args,
        **kwargs: P.kwargs
    ) -> ParameterLike:
        return self.vjp(x_in, u_in, grad_out, *args, **kwargs)[1]


class ParamGradMixin:
    def __init__(
        self,
        smooth_obj_prm: ParamSmoothObjType | None = None,
        grad_map_prm: ParamGradMapType | None = None
    ):
        self._init_grad_map_prm(smooth_obj_prm, grad_map_prm)
    
    def _init_grad_map_prm(self, smooth_map_prm, grad_map_prm):
        if (smooth_map_prm is None) == (grad_map_prm is None):
            raise ValueError(
                "Either `grad_map_prm` should be supplied or `smooth_map_prm`,"
                " but not both. One of them must be set to `None`."
            )
        if grad_map_prm is None:
            self.grad_map_prm = gradient(smooth_map_prm)
        else:
            self.grad_map_prm = grad_map_prm
    
    def _grad_map(
        self,
        x: VariableLike,
        *args: P.args,
        **kwargs: P.kwargs
    ) -> VariableLike:
        if not self._u_given.takes_params("grad"):
            return self.grad_map_prm(x, *args, **kwargs)
        return self.grad_map_prm(x, self._u_given.grad, *args, **kwargs)


class ParamProxMixin:
    def __init__(self, prox_map_prm: ParamProxMapType):
        self.prox_map_prm = prox_map_prm
    
    def _prox_map(self, x: VariableLike) -> VariableLike:
        if not self._u_given.takes_params("prox"):
            return self.prox_map_prm(x)
        return self.prox_map_prm(x, self._u_given.prox)
