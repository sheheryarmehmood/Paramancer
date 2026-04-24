from __future__ import annotations

import torch

from ...operators.grad import gradient
from ...variable import ParamBundle
from ...variable.flat import FlatVar
from ...variable.pair import PairVar
from ...variable.types import (
    IndexMapType,
    AlgoVar,
    FlatRawVarType,
    FlattendType,
    P,
    ParamGradMapType,
    ParamProxMapType,
    ParamSmoothObjType,
    PSpecType,
    VSpecType,
)
from ...variable.util import (
    flatten_raw,
    is_flat_var,
    is_pair_var,
    unflatten_raw,
)

class ParamMarkovStepMixin:
    def __init__(
        self,
        u_in: ParamBundle | None = None,
    ):
        self._u_given = u_in

    def markov_step(
        self,
        z_curr: AlgoVar,
        u_in: ParamBundle | None = None,
        *args: P.args,
        **kwargs: P.kwargs,
    ) -> AlgoVar:
        self.u_given = u_in
        if self.is_markovian():
            return self.step(z_curr, *args, **kwargs)
        x_curr, self.x_prev = z_curr
        x_new = self.step(x_curr, *args, **kwargs)
        return PairVar(x_new, x_curr)

    @property
    def u_given(self) -> ParamBundle:
        if self._u_given is None:
            raise RuntimeError("Parameter accessed without setting u_given")
        return self._u_given

    @u_given.setter
    def u_given(self, u_in: ParamBundle | None):
        if u_in is None or u_in is self._u_given:
            return
        self._u_given = u_in


def flatten_inputs(
    step,
    x_spec: VSpecType,
    u_spec: PSpecType,
    indices: IndexMapType,
    *args: P.args,
    **kwargs: P.kwargs,
):
    num_x = sum(x_spec[1:]) if len(x_spec) > 1 else 1

    def step_flat(*vars_and_pars: torch.Tensor) -> FlattendType:
        x_raw = unflatten_raw(vars_and_pars[:num_x], x_spec)
        u = ParamBundle(unflatten_raw(vars_and_pars[num_x:], u_spec), indices)
        x = PairVar(x_raw) if x_spec[0] == "pair" else FlatVar(x_raw)
        out = step(x, u, *args, **kwargs)
        out_raw = out.data if is_flat_var(out) or is_pair_var(out) else out
        return flatten_raw(out_raw)[0]

    return step_flat


class JVPMixin:
    def jvp(
        self,
        z_in: AlgoVar,
        u_in: ParamBundle,
        z_tan: AlgoVar,
        u_tan: ParamBundle,
        *args: P.args,
        **kwargs: P.kwargs,
    ) -> AlgoVar:
        z_flat, z_spec = z_in.flatten()
        u_flat, u_spec = u_in.flatten()
        z_tan_flat, _ = z_tan.flatten()
        u_tan_flat, _ = u_tan.flatten()
        step = flatten_inputs(
            self.markov_step, z_spec, u_spec, u_in.indices, *args, **kwargs
        )
        _, out_tan_flat = torch.func.jvp(
            step, (*z_flat, *u_flat), (*z_tan_flat, *u_tan_flat)
        )
        Var = PairVar if z_spec[0] == "pair" else FlatVar
        return Var.unflatten(out_tan_flat, z_spec)

    def jvp_var(
        self,
        z_in: AlgoVar,
        u_in: ParamBundle,
        z_tan: AlgoVar,
        *args: P.args,
        **kwargs: P.kwargs,
    ) -> AlgoVar:
        return self.jvp(
            z_in, u_in, z_tan, u_in.zeros_like(), *args, **kwargs
        )

    def jvp_par(
        self,
        z_in: AlgoVar,
        u_in: ParamBundle,
        u_tan: ParamBundle,
        *args: P.args,
        **kwargs: P.kwargs,
    ) -> AlgoVar:
        return self.jvp(
            z_in, u_in, z_in.zeros_like(), u_tan, *args, **kwargs
        )


class VJPMixin:
    def vjp(
        self,
        z_in: AlgoVar,
        u_in: ParamBundle,
        grad_out: AlgoVar,
        *args: P.args,
        **kwargs: P.kwargs,
    ) -> tuple[AlgoVar, ParamBundle]:
        z_flat, x_spec = z_in.flatten()
        u_flat, u_spec = u_in.flatten()
        grad_out_flat, _ = grad_out.flatten()
        step = flatten_inputs(
            self.markov_step, x_spec, u_spec, u_in.indices, *args, **kwargs
        )
        _, vjp = torch.func.vjp(step, *z_flat, *u_flat)
        grad_in_flat = vjp(grad_out_flat)
        Var = PairVar if x_spec[0] == "pair" else FlatVar
        grad_z = Var.unflatten(grad_in_flat[: len(z_flat)], x_spec)
        grad_u = ParamBundle(
            unflatten_raw(grad_in_flat[len(z_flat) :], u_spec), u_in.indices
        )
        return grad_z, grad_u

    def vjp_var(
        self,
        z_in: AlgoVar,
        u_in: ParamBundle,
        grad_out: AlgoVar,
        *args: P.args,
        **kwargs: P.kwargs,
    ) -> AlgoVar:
        return self.vjp(z_in, u_in, grad_out, *args, **kwargs)[0]

    def vjp_par(
        self,
        z_in: AlgoVar,
        u_in: ParamBundle,
        grad_out: AlgoVar,
        *args: P.args,
        **kwargs: P.kwargs,
    ) -> ParamBundle:
        self._adjoint_state = grad_out
        return self.vjp(z_in, u_in, grad_out, *args, **kwargs)[1]
    
    @property
    def adjoint_state(self) -> AlgoVar:
        if not hasattr(self, "_adjoint_state"):
            raise RuntimeError("Adjoint state accessed before any vjp_par call.")
        return self._adjoint_state


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
        self.grad_map_prm = (
            gradient(smooth_map_prm)
            if grad_map_prm is None
            else grad_map_prm
        )

    def _grad_map(
        self,
        x: FlatRawVarType,
        *args: P.args,
        **kwargs: P.kwargs,
    ) -> FlatRawVarType:
        if not self._u_given.takes_params("grad"):
            return self.grad_map_prm(x, *args, **kwargs)
        return self.grad_map_prm(x, self._u_given.grad, *args, **kwargs)


class ParamProxMixin:
    def __init__(self, prox_map_prm: ParamProxMapType):
        self.prox_map_prm = prox_map_prm

    def _prox_map(self, x: FlatRawVarType) -> FlatRawVarType:
        if not self._u_given.takes_params("prox"):
            return self.prox_map_prm(x)
        return self.prox_map_prm(x, self._u_given.prox)
