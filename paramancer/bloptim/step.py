from __future__ import annotations

from ._mixins import (
    ParamMarkovStepMixin, JVPMixin, VJPMixin, ParamGradMixin, ParamProxMixin
)
from ..optim.step import (
    GDStep, PolyakStep, NesterovStep,
    ProxGradStep, FISTAStep
)
from ..variable.types import (
    IndexMapType, ScalarLike, ParameterLike,
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


class GDParamMarkovStep(
    ParamGradMixin, JVPMixin, VJPMixin, ParamMarkovStepMixin, GDStep
):
    def __init__(
        self,
        stepsize: ScalarLike,
        grad_map_prm: ParamGradMapType | None = None,
        smooth_obj_prm: ParamSmoothObjType | None = None,
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


class PolyakParamMarkovStep(
    ParamGradMixin, JVPMixin, VJPMixin, ParamMarkovStepMixin, PolyakStep
):
    def __init__(
        self,
        stepsize: ScalarLike,
        momentum: ScalarLike,
        grad_map_prm: ParamGradMapType | None = None,
        smooth_obj_prm: ParamSmoothObjType | None = None,
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
    ParamGradMixin, JVPMixin, VJPMixin, ParamMarkovStepMixin, NesterovStep
):
    def __init__(
        self,
        stepsize: ScalarLike,
        grad_map_prm: ParamGradMapType | None = None,
        smooth_obj_prm: ParamSmoothObjType | None = None,
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
    ParamProxMixin, ParamGradMixin, JVPMixin, VJPMixin, ParamMarkovStepMixin, 
    ProxGradStep
):
    def __init__(
        self,
        stepsize: ScalarLike,
        prox_map_prm: ParamProxMapType,
        grad_map_prm: ParamGradMapType | None = None,
        smooth_obj_prm: ParamSmoothObjType | None = None,
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
    ParamProxMixin, ParamGradMixin, JVPMixin, VJPMixin, ParamMarkovStepMixin,
    FISTAStep
):
    def __init__(
        self,
        stepsize: ScalarLike,
        prox_map_prm: ParamProxMapType,
        grad_map_prm: ParamGradMapType | None = None,
        smooth_obj_prm: ParamSmoothObjType | None = None,
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
GDParamMarkovStep = GDParamMarkovStep
PolyakMarkovParamStep = PolyakParamMarkovStep
NesterovMarkovParamStep = NesterovParamMarkovStep
ProxGradMarkovParamStep = ProxGradParamMarkovStep
FISTAMarkovParamStep = FISTAParamMarkovStep

