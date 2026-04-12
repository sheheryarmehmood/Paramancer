from __future__ import annotations

import torch

from ..operators.grad import gradient
from ..variable import ParamBundle
from ..variable.flat import FlatVar
from ..variable.pair import PairVar
from ..variable.types import (
    AlgoVarLike,
    FlatVarLike,
    FlattendType,
    IndexMapType,
    P,
    ParameterLike,
    PSpecType,
    PairVarLike,
    ParamGradMapType,
    ParamProxMapType,
    ParamSmoothObjType,
    VSpecType,
)
from ..variable.util import (
    as_flat_var,
    as_pair_var,
    flatten_raw,
    flatten_flat_raw,
    is_flat_var,
    is_param_bundle,
    is_pair_raw_var,
    is_pair_var,
    unflatten_raw,
    unflatten_flat_raw,
)


class ParamMarkovStepMixin:
    def __init__(
        self,
        u_in: ParameterLike | None = None,
        indices: IndexMapType | None = None,
    ):
        self._is_input_parambundle = is_param_bundle(u_in)
        self._u_given = u_in if self._is_input_parambundle else ParamBundle(
            u_in, indices=indices
        )

    def step(
        self,
        x_curr: AlgoVarLike,
        u_in: ParameterLike | None = None,
        *args: P.args,
        **kwargs: P.kwargs,
    ):
        input_is_wrapper = is_flat_var(x_curr) or is_pair_var(x_curr)
        if self.is_markovian():
            x_flat = as_flat_var(x_curr)
            pair_mode = False
        else:
            if is_pair_var(x_curr) or is_pair_raw_var(x_curr):
                x_pair = as_pair_var(x_curr)
                self.x_prev = x_pair.second
                x_flat = x_pair.first
                pair_mode = True
            else:
                x_flat = as_flat_var(x_curr)
                pair_mode = False

        self._input_is_wrapper = input_is_wrapper
        self.u_given = u_in
        x_new = super().step(x_flat, *args, **kwargs)

        if not self.is_markovian() and pair_mode:
            x_new = PairVar(x_new, x_flat)
        return x_new if input_is_wrapper else x_new.data

    def __call__(
        self,
        x_curr: AlgoVarLike,
        u_given: ParameterLike | None = None,
        *args: P.args,
        **kwargs: P.kwargs,
    ):
        return self.step(x_curr, u_given, *args, **kwargs)

    @property
    def u_given(self) -> ParameterLike:
        if self._u_given is None:
            raise RuntimeError("Parameter accessed without setting u_given")
        return self._u_given if self._is_input_parambundle else self._u_given.data

    @u_given.setter
    def u_given(self, u_in: ParameterLike | None):
        if is_param_bundle(u_in):
            u_in = u_in.data
        if u_in is None or u_in is self._u_given.data:
            return
        self._u_given.data = u_in


def flatten_inputs(step, x_spec: VSpecType, u_spec: PSpecType, *args: P.args, **kwargs: P.kwargs):
    num_x = sum(x_spec[1:]) if len(x_spec) > 1 else 1

    def step_flat(*vars_and_pars: torch.Tensor) -> FlattendType:
        x = unflatten_raw(vars_and_pars[:num_x], x_spec)
        u = unflatten_flat_raw(vars_and_pars[num_x:], u_spec)
        out = step(x, u, *args, **kwargs)
        out_raw = out.data if is_flat_var(out) or is_pair_var(out) else out
        return flatten_raw(out_raw)[0]

    return step_flat


class JVPMixin:
    def jvp(
        self,
        x_in: FlatVarLike,
        u_in: ParameterLike,
        x_tan: FlatVarLike,
        u_tan: ParameterLike,
        *args: P.args,
        **kwargs: P.kwargs,
    ) -> FlatVarLike:
        x_is_var = is_flat_var(x_in)
        u_is_par = is_param_bundle(u_in)
        u_tan_data = u_tan.data if is_param_bundle(u_tan) else u_tan
        x_flat, x_spec = flatten_flat_raw(x_in.data if x_is_var else x_in)
        u_flat, u_spec = flatten_flat_raw(u_in.data if u_is_par else u_in)
        x_tan_flat, _ = flatten_flat_raw(x_tan.data if x_is_var else x_tan)
        u_tan_flat, _ = flatten_flat_raw(u_tan_data)
        step = flatten_inputs(self.step, x_spec, u_spec, *args, **kwargs)
        _, out_tan_flat = torch.func.jvp(
            step, (*x_flat, *u_flat), (*x_tan_flat, *u_tan_flat)
        )
        out_tan = unflatten_flat_raw(out_tan_flat, x_spec)
        return FlatVar(out_tan) if x_is_var else out_tan

    def jvp_var(
        self,
        x_in: FlatVarLike,
        u_in: ParameterLike,
        x_tan: FlatVarLike,
        *args: P.args,
        **kwargs: P.kwargs,
    ) -> FlatVarLike:
        u_is_par = is_param_bundle(u_in)
        u_par = u_in if u_is_par else ParamBundle(u_in)
        u_par_tan = u_par.zeros_like()
        u_tan = u_par_tan if u_is_par else u_par_tan.data
        return self.jvp(x_in, u_in, x_tan, u_tan, *args, **kwargs)

    def jvp_par(
        self,
        x_in: FlatVarLike,
        u_in: ParameterLike,
        u_tan: ParameterLike,
        *args: P.args,
        **kwargs: P.kwargs,
    ) -> FlatVarLike:
        x_is_var = is_flat_var(x_in)
        x_var = as_flat_var(x_in)
        x_var_tan = x_var.zeros_like()
        x_tan = x_var_tan if x_is_var else x_var_tan.data
        return self.jvp(x_in, u_in, x_tan, u_tan, *args, **kwargs)


class VJPMixin:
    def vjp(
        self,
        x_in: FlatVarLike,
        u_in: ParameterLike,
        grad_out: FlatVarLike,
        *args: P.args,
        **kwargs: P.kwargs,
    ) -> tuple[FlatVarLike, ParameterLike]:
        x_is_var = is_flat_var(x_in)
        u_is_par = is_param_bundle(u_in)
        x_flat, x_spec = flatten_flat_raw(x_in.data if x_is_var else x_in)
        u_flat, u_spec = flatten_flat_raw(u_in.data if u_is_par else u_in)
        grad_out_flat, _ = flatten_flat_raw(grad_out.data if x_is_var else grad_out)
        step = flatten_inputs(self.step, x_spec, u_spec, *args, **kwargs)
        _, vjp = torch.func.vjp(step, *x_flat, *u_flat)
        grad_in_flat = vjp(grad_out_flat)
        grad_x = unflatten_flat_raw(grad_in_flat[: len(x_flat)], x_spec)
        grad_u = unflatten_flat_raw(grad_in_flat[len(x_flat) :], u_spec)
        return (
            FlatVar(grad_x) if x_is_var else grad_x,
            ParamBundle(grad_u, u_in.indices) if u_is_par else grad_u,
        )

    def vjp_var(
        self,
        x_in: FlatVarLike,
        u_in: ParameterLike,
        grad_out: FlatVarLike,
        *args: P.args,
        **kwargs: P.kwargs,
    ) -> FlatVarLike:
        return self.vjp(x_in, u_in, grad_out, *args, **kwargs)[0]

    def vjp_par(
        self,
        x_in: FlatVarLike,
        u_in: ParameterLike,
        grad_out: FlatVarLike,
        *args: P.args,
        **kwargs: P.kwargs,
    ) -> ParameterLike:
        return self.vjp(x_in, u_in, grad_out, *args, **kwargs)[1]


class ParamGradMixin:
    def __init__(
        self,
        smooth_obj_prm: ParamSmoothObjType | None = None,
        grad_map_prm: ParamGradMapType | None = None,
    ):
        self._init_grad_map_prm(smooth_obj_prm, grad_map_prm)

    def _init_grad_map_prm(self, smooth_map_prm, grad_map_prm):
        if (smooth_map_prm is None) == (grad_map_prm is None):
            raise ValueError(
                "Either `grad_map_prm` should be supplied or `smooth_map_prm`,"
                " but not both. One of them must be set to `None`."
            )
        self.grad_map_prm = gradient(smooth_map_prm) if grad_map_prm is None else grad_map_prm

    def _grad_map(self, x: FlatVarLike, *args: P.args, **kwargs: P.kwargs) -> FlatVarLike:
        if not self._u_given.takes_params("grad"):
            return self.grad_map_prm(x, *args, **kwargs)
        return self.grad_map_prm(x, self._u_given.grad, *args, **kwargs)


class ParamProxMixin:
    def __init__(self, prox_map_prm: ParamProxMapType):
        self.prox_map_prm = prox_map_prm

    def _prox_map(self, x: FlatVarLike) -> FlatVarLike:
        if not self._u_given.takes_params("prox"):
            return self.prox_map_prm(x)
        return self.prox_map_prm(x, self._u_given.prox)
