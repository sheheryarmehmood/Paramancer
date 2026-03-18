from __future__ import annotations
import torch
from typing import Any

from ..optim.step import (
    GDStep, PolyakStep, NesterovStep,
    ProxGradStep, FISTAStep
)
from ..optim.util import ensure_var_input
from ..operators.grad import gradient
from ..variable import Variable, ParameterBundle
from ..variable.util import flatten, unflatten
from ..variable.types import (
    IndexMapType, ScalarLike, VariableLike, ParameterLike, FlatParameter,
    ApplyType, SpecType,
    ParamSmoothObjType, ParamGradMapType, ParamProxMapType,
    MomentumSchedType, StepsizeSchedType
)

"""
`OptimizerStep` and its child classes are used to implement the following
classes. While `OptimizerStep` has a method `is_markovian` which indicates
whether the corresponding algorithm is memory less or not, the following
steps are implemented in a memory-less or Markovian fashion. Moreover they
also take as input, the parameter of an algorithm. The reason for taking the
additional inputs is to make sure that these classes can be used to implement
ImplicitDifferentiation. As an example, consider Polyak's algorithm.

We define the mapping $A$ such that:
$$(x_{k+1}, x_k) = A(x_k, x_{k-1}, u) := (P(x_k, x_{k-1}, u), x_k)$$
where $P$ is defined as:
$$P(x_k, x_{k-1}, u) := x_k - a \nabla_{x} f (x_k, u) + b (x_k - x_{k-1})$$
where $a$ and $b$ are step size and momentum parameter respectively.
"""


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
        *args: Any, 
        **kwargs: Any
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
        *args: Any, 
        **kwargs: Any
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


class VJPStepMixin:
    def vjp(
        self,
        x_in: VariableLike,
        u_in: ParameterLike,
        grad_out: VariableLike,
    ) -> tuple[VariableLike, ParameterLike]:
        ...

    def vjp_var(
        self,
        x_in: VariableLike,
        u_in: ParameterLike,
        grad_out: VariableLike,
    ) -> VariableLike:
        grad_out_flat, spec = flatten(grad_out)
        def step_var(
            *x_flat: torch.Tensor
        ) -> ApplyType:
            x = unflatten(x_flat, spec)
            out = self.step(x, u_in)
            out_flat, = flatten(out)
            return out_flat
        is_flat = isinstance(u_in, FlatParameter)
        if is_flat:
            u_in = (u_in,)
        (_, vjpfunc) = torch.func.vjp(step_var, *u_in)
        grad_u = vjpfunc(grad_out_flat)
        return grad_u[0] if is_flat else grad_u

    def vjp_prm(
        self,
        x_in: VariableLike,
        u_in: ParameterLike,
        grad_out: VariableLike,
    ) -> ParameterLike:
        if isinstance(u_in, ParameterBundle):
            u_in = u_in.data
        grad_out_flat, = flatten(grad_out)
        def step_prm(
            *u: torch.nn.Parameter
        ) -> ApplyType:
            out = self.step(x_in, u)
            out_flat, = flatten(out)
            return out_flat
        is_flat = isinstance(u_in, torch.nn.Parameter)
        if is_flat:
            u_in = (u_in,)
        (_, vjpfunc) = torch.func.vjp(step_prm, *u_in)
        grad_u = vjpfunc(grad_out_flat)
        return grad_u[0] if is_flat else grad_u
        
        


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
    
    def _grad_map(self, x: VariableLike) -> VariableLike:
        if not self._u_given.takes_params("grad"):
            return self.grad_map_prm(x)
        return self.grad_map_prm(x, self._u_given.grad)


class ParamProxMixin:
    def __init__(self, prox_map_prm: ParamProxMapType):
        self.prox_map_prm = prox_map_prm
    
    def _prox_map(self, x: VariableLike) -> VariableLike:
        if not self._u_given.takes_params("prox"):
            return self.prox_map_prm(x)
        return self.prox_map_prm(x, self._u_given.prox)


class GDParamMarkovStep(ParamGradMixin, ParamMarkovStepMixin, GDStep):
    def __init__(
        self,
        stepsize: ScalarLike,
        smooth_obj_prm: ParamSmoothObjType | None = None,
        grad_map_prm: ParamGradMapType | None = None,
        u_in: ParameterLike | None = None,
        indices: IndexMapType | None = None,
        stepsize_scheduler: StepsizeSchedType | None = None,
        linesearch: bool = True,
        tracking: bool = False
    ):
        ParamGradMixin.__init__(self, smooth_obj_prm, grad_map_prm)
        ParamMarkovStepMixin.__init__(self, u_in, indices)
        GDStep.__init__(
            self, stepsize, grad_map=self._grad_map, linesearch=linesearch,
            stepsize_scheduler=stepsize_scheduler, tracking=tracking
        )


class PolyakParamMarkovStep(ParamGradMixin, ParamMarkovStepMixin, PolyakStep):
    def __init__(
        self,
        stepsize: ScalarLike,
        momentum: ScalarLike,
        smooth_obj_prm: ParamSmoothObjType | None = None,
        grad_map_prm: ParamGradMapType | None = None,
        u_in: ParameterLike | None = None,
        indices: IndexMapType | None = None,
        tracking: bool = False
    ):
        ParamGradMixin.__init__(self, smooth_obj_prm, grad_map_prm)
        ParamMarkovStepMixin.__init__(self, u_in, indices)
        PolyakStep.__init__(
            self, stepsize, momentum, grad_map=self._grad_map,
            tracking=tracking
        )


class NesterovParamMarkovStep(
    ParamGradMixin, ParamMarkovStepMixin, NesterovStep
):
    def __init__(
        self,
        stepsize: ScalarLike,
        smooth_obj_prm: ParamSmoothObjType | None = None,
        grad_map_prm: ParamGradMapType | None = None,
        u_in: ParameterLike | None = None,
        indices: IndexMapType | None = None,
        momentum_scheduler: MomentumSchedType | None = None,
        tracking: bool = False
    ):
        ParamGradMixin.__init__(self, smooth_obj_prm, grad_map_prm)
        ParamMarkovStepMixin.__init__(self, u_in, indices)
        NesterovStep.__init__(
            self, stepsize, grad_map=self._grad_map, tracking=tracking,
            momentum_scheduler=momentum_scheduler
        )


class ProxGradParamMarkovStep(
    ParamProxMixin, ParamGradMixin, ParamMarkovStepMixin, ProxGradStep
):
    def __init__(
        self,
        stepsize: ScalarLike,
        prox_map_prm: ParamProxMapType,
        smooth_obj_prm: ParamSmoothObjType | None = None,
        grad_map_prm: ParamGradMapType | None = None,
        u_in: ParameterLike | None = None,
        indices: IndexMapType | None = None,
        tracking: bool = False
    ):
        ParamProxMixin.__init__(self, prox_map_prm)
        ParamGradMixin.__init__(self, smooth_obj_prm, grad_map_prm)
        ParamMarkovStepMixin.__init__(self, u_in, indices)
        ProxGradStep.__init__(
            self, stepsize, self._prox_map, grad_map=self._grad_map,
            tracking=tracking
        )

class FISTAParamMarkovStep(
    ParamProxMixin, ParamGradMixin, ParamMarkovStepMixin, FISTAStep
):
    def __init__(
        self,
        stepsize: ScalarLike,
        prox_map_prm: ParamProxMapType,
        smooth_obj_prm: ParamSmoothObjType | None = None,
        grad_map_prm: ParamGradMapType | None = None,
        u_in: ParameterLike | None = None,
        indices: IndexMapType | None = None,
        momentum_scheduler: MomentumSchedType | None = None,
        tracking: bool = False
    ):
        ParamProxMixin.__init__(self, prox_map_prm)
        ParamGradMixin.__init__(self, smooth_obj_prm, grad_map_prm)
        ParamMarkovStepMixin.__init__(self, u_in, indices)
        FISTAStep.__init__(
            self, stepsize, self._prox_map, grad_map=self._grad_map,
            tracking=tracking, momentum_scheduler=momentum_scheduler
        )


# Backwards-compatible aliases for the original public API.
GDMarkovParamStep = GDParamMarkovStep
PolyakMarkovParamStep = PolyakParamMarkovStep
NesterovMarkovParamStep = NesterovParamMarkovStep
ProxGradMarkovParamStep = ProxGradParamMarkovStep
FISTAMarkovParamStep = FISTAParamMarkovStep

