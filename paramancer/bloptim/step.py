from __future__ import annotations
import torch
from typing import Any

from ..optim.step import (
    GDStep, PolyakStep, NesterovStep,
    ProxGradStep, FISTAStep
)
from ..optim.util import ensure_var_input
from ..operators.grad import gradient
from ..variable import Variable
from ..variable.types import (
    ScalarLike, VariableLike, ParameterType,
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
    def __init__(self, u_in: ParameterType | None = None):
        self._u_given = u_in
    
    @ensure_var_input
    def step(
        self,
        x_curr: Variable,
        u_in: ParameterType | None = None,
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
        u_given: ParameterType | None = None,
        *args: Any, 
        **kwargs: Any
    ) -> VariableLike:
        return self.step(x_curr, u_given, *args, **kwargs)
    
    @property
    def u_given(self) -> ParameterType:
        if self._u_given is None:
            raise RuntimeError(
                "Parametric step called without setting u_given"
            )
        return self._u_given
    
    @u_given.setter
    def u_given(self, u_in: ParameterType | None):
        """Ignores the incoming `u_in` when it is `None`"""
        if u_in is not None and u_in is not self._u_given:
            self._u_given = u_in


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
        return self.grad_map_prm(x, self.u_given)


class ParamProxMixin:
    def __init__(self, prox_map_prm: ParamProxMapType):
        self.prox_map_prm = prox_map_prm
    
    def _prox_map(self, x: VariableLike) -> VariableLike:
        return self.prox_map_prm(x, self.u_given)


class GDParamMarkovStep(ParamGradMixin, ParamMarkovStepMixin, GDStep):
    def __init__(
        self,
        stepsize: ScalarLike,
        smooth_obj_prm: ParamSmoothObjType | None = None,
        grad_map_prm: ParamGradMapType | None = None,
        u_in: ParameterType | None = None,
        stepsize_scheduler: StepsizeSchedType | None = None,
        linesearch: bool = True,
        tracking: bool = False
    ):
        ParamGradMixin.__init__(self, smooth_obj_prm, grad_map_prm)
        ParamMarkovStepMixin.__init__(self, u_in)
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
        u_in: ParameterType | None = None,
        tracking: bool = False
    ):
        ParamGradMixin.__init__(self, smooth_obj_prm, grad_map_prm)
        ParamMarkovStepMixin.__init__(self, u_in)
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
        u_in: ParameterType | None = None,
        momentum_scheduler: MomentumSchedType | None = None,
        tracking: bool = False
    ):
        ParamGradMixin.__init__(self, smooth_obj_prm, grad_map_prm)
        ParamMarkovStepMixin.__init__(self, u_in)
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
        u_in: ParameterType | None = None,
        tracking: bool = False
    ):
        ParamProxMixin.__init__(self, prox_map_prm)
        ParamGradMixin.__init__(self, smooth_obj_prm, grad_map_prm)
        ParamMarkovStepMixin.__init__(self, u_in)
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
        u_in: ParameterType | None = None,
        momentum_scheduler: MomentumSchedType | None = None,
        tracking: bool = False
    ):
        ParamProxMixin.__init__(self, prox_map_prm)
        ParamGradMixin.__init__(self, smooth_obj_prm, grad_map_prm)
        ParamMarkovStepMixin.__init__(self, u_in)
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

