from __future__ import annotations

import abc
from enum import Enum
from typing import Any

from .scheduler import MomentumScheduler
from .util import ensure_flat_input, ensure_pair_input
from ..operators.grad import gradient
from ..operators.linalg import adjoint
from ..variable.flat import FlatVar
from ..variable.pair import PairVar
from ..variable.types import (
    AlgoVarLike,
    FlatLinOpType,
    FlatRawVarType,
    FlatVarLike,
    LinOpType,
    MomentumSchedType,
    PGradMapType,
    ProxMapType,
    PSmoothObjType,
    ScalarLike,
    StepsizeSchedType,
)
from ..variable.util import as_flat_var, as_pair_var, is_flat_var, is_pair_raw_var, is_pair_var


def _wrap_flat_map(fn):
    def wrapped(x: FlatVar, *args: Any, **kwargs: Any) -> FlatVar:
        return FlatVar(fn(x.data, *args, **kwargs))

    return wrapped


def _wrap_pair_map(fn):
    def wrapped(x: PairVar, *args: Any, **kwargs: Any) -> PairVar:
        return PairVar(fn(x.data, *args, **kwargs))

    return wrapped


class MomentumType(Enum):
    Nesterov = "Nesterov"
    Polyak = "Polyak"


class OptimizerStep(abc.ABC):
    def __init__(self, tracking: bool = False):
        self._residual_tracking = tracking
        if self._residual_tracking:
            self._residual: FlatVar | PairVar | None = None
        self._input_is_wrapper = True

    @abc.abstractmethod
    def step(self, x_curr, *args: Any, **kwargs: Any):
        pass

    def __call__(self, x_curr: AlgoVarLike, *args: Any, **kwargs: Any):
        self._input_is_wrapper = is_flat_var(x_curr) or is_pair_var(x_curr)
        return self.step(x_curr, *args, **kwargs)

    @property
    def residual(self) -> AlgoVarLike:
        if not hasattr(self, "_residual"):
            raise RuntimeError(
                "residual tracking is disabled for this step instance."
            )
        if self._residual is None:
            raise AttributeError("`residual` must be set before referencing")
        return (
            self._residual if self._input_is_wrapper else self._residual.data
        )

    @property
    def residual_tracking(self) -> bool:
        return self._residual_tracking

    def is_markovian(self):
        return not hasattr(self, "_x_prev")


class MomentumStep(OptimizerStep):
    def __init__(
        self,
        momentum: ScalarLike,
        strategy: MomentumType = MomentumType.Nesterov,
        momentum_scheduler: MomentumSchedType | None = None,
    ):
        super().__init__(tracking=False)
        if not isinstance(strategy, MomentumType):
            raise TypeError(
                "parameter strategy can only be either "
                "MomentumType.Nesterov or MomentumType.Polyak"
            )
        self.momentum = momentum
        self.strategy = strategy
        self.momentum_scheduler = momentum_scheduler

    @ensure_pair_input
    def step(self, z_curr: PairVar) -> PairVar:
        if self.momentum_scheduler:
            self.momentum = self.momentum_scheduler()
        x_new = self.momentum * (z_curr.first - z_curr.second)
        if self.strategy == MomentumType.Nesterov:
            x_new = z_curr.first + x_new
        return PairVar(x_new, z_curr.first)

    # TODO: This previously returned `False`. But now, the way it is defined, it should return `True`. We should verify that this change does not cause any issues.
    def is_markovian(self):
        return True


class GDStep(OptimizerStep):
    def __init__(
        self,
        stepsize: ScalarLike,
        smooth_obj: PSmoothObjType | None = None,
        grad_map: PGradMapType | None = None,
        stepsize_scheduler: StepsizeSchedType | None = None,
        linesearch: bool = True,
        tracking: bool = False,
    ):
        super().__init__(tracking=tracking)
        self.stepsize = stepsize
        self._init_grad_map(smooth_obj, grad_map)
        self.stepsize_scheduler = stepsize_scheduler
        self.linesearch = linesearch

    @ensure_flat_input
    def step(self, x_curr: FlatVar, *args: Any, **kwargs: Any) -> FlatVar:
        d_curr = -self.grad_map(x_curr, *args, **kwargs)
        self._set_stepsize(x_curr, d_curr)
        x_new = x_curr + self.stepsize * d_curr
        if self._residual_tracking:
            self._residual = x_new - x_curr
        return x_new

    def _init_grad_map(self, smooth_obj, grad_map):
        if (smooth_obj is None) == (grad_map is None):
            raise ValueError(
                "Either `grad_map` should be supplied or `smooth_obj`, but not"
                " both. One of them must be set to `None`."
            )
        if grad_map is None:
            self.grad_map = _wrap_flat_map(gradient(smooth_obj))
        else:
            self.grad_map = _wrap_flat_map(grad_map)

    def _set_stepsize(self, x_curr: FlatVar, d_curr: FlatVar) -> None:
        if not self.stepsize_scheduler:
            return
        if self.linesearch:
            self.stepsize = self.stepsize_scheduler(x_curr.data, d_curr.data)
        else:
            self.stepsize = self.stepsize_scheduler()


class ProxStep(OptimizerStep):
    def __init__(self, prox_map: ProxMapType):
        super().__init__(tracking=False)
        self.prox_map = _wrap_flat_map(prox_map)

    @ensure_flat_input
    def step(self, x_curr: FlatVar) -> FlatVar:
        return self.prox_map(x_curr)

class AffineStep(OptimizerStep):
    def __init__(
        self,
        lin_op: LinOpType,
        vector: AlgoVarLike,
        tracking: bool = False,
    ):
        super().__init__(tracking=tracking)
        self._pair_mode = is_pair_raw_var(vector) or is_pair_var(vector)
        if self._pair_mode:
            self.lin_op = _wrap_pair_map(lin_op)
            self.vector = as_pair_var(vector)
        else:
            self.lin_op = _wrap_flat_map(lin_op)
            self.vector = as_flat_var(vector)

    def step(self, x_curr: AlgoVarLike, *args: Any, **kwargs: Any) -> AlgoVarLike:
        self._input_is_wrapper = is_flat_var(x_curr) or is_pair_var(x_curr)
        x_var = as_pair_var(x_curr) if self._pair_mode else as_flat_var(x_curr)
        x_new = self.lin_op(x_var, *args, **kwargs) + self.vector
        if self._residual_tracking:
            self._residual = x_new - x_var
        return x_new if self._input_is_wrapper else x_new.data


class PolyakStep(OptimizerStep):
    def __init__(
        self,
        stepsize: ScalarLike,
        momentum: ScalarLike,
        smooth_obj: PSmoothObjType | None = None,
        grad_map: PGradMapType | None = None,
        tracking: bool = False,
    ):
        super().__init__(tracking=False)
        self.gd_step = GDStep(
            stepsize, smooth_obj=smooth_obj, grad_map=grad_map, 
            tracking=tracking
        )
        self.mm_step = MomentumStep(momentum, strategy=MomentumType.Polyak)
        self._x_prev = None

    @ensure_flat_input
    def step(self, x_curr: FlatVar, *args: Any, **kwargs: Any) -> FlatVar:
        x_new = self.gd_step(x_curr, *args, **kwargs)
        if self._x_prev is not None:
            z_curr = PairVar(x_curr, self._x_prev)
            x_new = x_new + self.mm_step(z_curr).first
        self._x_prev = x_curr
        return x_new

    @property
    def residual(self) -> FlatVarLike:
        self.gd_step._input_is_wrapper = self._input_is_wrapper
        return self.gd_step.residual

    @property
    def residual_tracking(self) -> bool:
        return self.gd_step.residual_tracking

    @property
    def x_prev(self) -> FlatVar:
        if self._x_prev is None:
            raise RuntimeError("`x_prev` accessed before assignment.")
        return self._x_prev

    @x_prev.setter
    def x_prev(self, x):
        self._x_prev = x


class NesterovStep(OptimizerStep):
    def __init__(
        self,
        stepsize: ScalarLike,
        smooth_obj: PSmoothObjType | None = None,
        grad_map: PGradMapType | None = None,
        momentum_scheduler: MomentumSchedType | None = None,
        tracking: bool = False,
    ):
        super().__init__(tracking=False)
        if momentum_scheduler is None:
            momentum_scheduler = MomentumScheduler()
        self.gd_step = GDStep(
            stepsize, smooth_obj=smooth_obj, grad_map=grad_map, 
            tracking=tracking
        )
        self.mm_step = MomentumStep(
            momentum=None, momentum_scheduler=momentum_scheduler
        )
        self._x_prev = None

    @ensure_flat_input
    def step(self, x_curr: FlatVar, *args: Any, **kwargs: Any) -> FlatVar:
        if self._x_prev is None:
            self._x_prev = x_curr
        z_curr = PairVar(x_curr, self._x_prev)
        x_new = self.gd_step(self.mm_step(z_curr).first, *args, **kwargs)
        self._x_prev = x_curr
        return x_new

    @property
    def residual(self) -> FlatVarLike:
        self.gd_step._input_is_wrapper = self._input_is_wrapper
        return self.gd_step.residual

    @property
    def residual_tracking(self) -> bool:
        return self.gd_step.residual_tracking

    @property
    def x_prev(self) -> FlatVar:
        if self._x_prev is None:
            raise RuntimeError("`x_prev` accessed before assignment.")
        return self._x_prev

    @x_prev.setter
    def x_prev(self, x):
        self._x_prev = x

    def restart(self):
        self._x_prev = None
        self.mm_step.momentum_scheduler.restart()


class ProxGradStep(OptimizerStep):
    def __init__(
        self,
        stepsize: ScalarLike,
        prox_map: ProxMapType,
        smooth_obj: PSmoothObjType | None = None,
        grad_map: PGradMapType | None = None,
        tracking: bool = False,
    ):
        super().__init__(tracking=tracking)
        self.gd_step = GDStep(
            stepsize, smooth_obj=smooth_obj, grad_map=grad_map, tracking=False
        )
        self.prox_step = ProxStep(prox_map)

    @ensure_flat_input
    def step(self, x_curr: FlatVar, *args: Any, **kwargs: Any) -> FlatVar:
        x_new = self.prox_step(self.gd_step(x_curr, *args, **kwargs))
        if self._residual_tracking:
            self._residual = x_new - x_curr
        return x_new


class FISTAStep(OptimizerStep):
    def __init__(
        self,
        stepsize: ScalarLike,
        prox_map: ProxMapType,
        smooth_obj: PSmoothObjType | None = None,
        grad_map: PGradMapType | None = None,
        momentum_scheduler: MomentumSchedType | None = None,
        tracking: bool = False,
    ):
        super().__init__(tracking=False)
        if momentum_scheduler is None:
            momentum_scheduler = MomentumScheduler()
        self.pgd_step = ProxGradStep(
            stepsize,
            prox_map,
            smooth_obj=smooth_obj,
            grad_map=grad_map,
            tracking=tracking,
        )
        self.mm_step = MomentumStep(
            momentum=None, momentum_scheduler=momentum_scheduler
        )
        self._x_prev = None

    @ensure_flat_input
    def step(self, x_curr: FlatVar, *args: Any, **kwargs: Any) -> FlatVar:
        if self._x_prev is None:
            self._x_prev = x_curr
        z_curr = PairVar(x_curr, self._x_prev)
        x_new = self.pgd_step(self.mm_step(z_curr).first, *args, **kwargs)
        self._x_prev = x_curr
        return x_new

    @property
    def residual(self) -> FlatVarLike:
        self.pgd_step._input_is_wrapper = self._input_is_wrapper
        return self.pgd_step.residual

    @property
    def residual_tracking(self) -> bool:
        return self.pgd_step.residual_tracking

    @property
    def x_prev(self) -> FlatVar:
        if self._x_prev is None:
            raise RuntimeError("`x_prev` accessed before assignment.")
        return self._x_prev

    @x_prev.setter
    def x_prev(self, x):
        self._x_prev = x

    def restart(self):
        self._x_prev = None
        self.mm_step.momentum_scheduler.restart()

class PDHGPartialStep(OptimizerStep):
    def __init__(
        self, stepsize: ScalarLike, prox_map: ProxMapType, lin_op: FlatLinOpType
    ):
        super().__init__(tracking=False)
        self.stepsize = stepsize
        self.lin_op = _wrap_flat_map(lin_op)
        self.prox_step = ProxStep(prox_map)

    def __call__(
        self,
        inp_curr: FlatVarLike,
        oth_curr: FlatVarLike,
        *args: Any,
        **kwargs: Any,
    ) -> FlatVarLike:
        return self.step(inp_curr, oth_curr, *args, **kwargs)

    @ensure_flat_input
    def step(
        self,
        inp_curr: FlatVar,
        oth_curr: FlatVarLike,
        *args: Any,
        **kwargs: Any
    ) -> FlatVar:
        oth_var = as_flat_var(oth_curr)
        return self.prox_step(
            inp_curr - self.stepsize * self.lin_op(oth_var, *args, **kwargs)
        )


class PDHGStep(OptimizerStep):
    def __init__(
        self,
        stepsize_primal: ScalarLike,
        stepsize_dual: ScalarLike,
        prox_map_primal: ProxMapType,
        prox_map_dual: ProxMapType,
        lin_op: FlatLinOpType,
        lin_op_adj: FlatLinOpType | None = None,
        zero_el: FlatRawVarType | None = None,
        tracking: bool = False,
    ):
        super().__init__(tracking)
        if (lin_op_adj is None) == (zero_el is None):
            raise ValueError(
                "Either `zero_el` should be supplied or `lin_op_adj`, but not"
                " both. One of them must be set to `None`."
            )
        if lin_op_adj is None:
            lin_op_adj = adjoint(lin_op, zero_el)
        self.primal_step = PDHGPartialStep(
            stepsize_primal, prox_map_primal, lin_op_adj
        )
        self.dual_step = PDHGPartialStep(stepsize_dual, prox_map_dual, lin_op)

    @ensure_pair_input
    def step(self, z_curr: PairVar, *args: Any, **kwargs: Any) -> PairVar:
        x_prim = self.primal_step(z_curr.first, z_curr.second, *args, **kwargs)
        x_prim_mm = 2 * x_prim - z_curr.first
        x_dual = self.dual_step(z_curr.second, -x_prim_mm, *args, **kwargs)
        z_new = PairVar(x_prim, x_dual)
        if self._residual_tracking:
            self._residual = z_new - z_curr
        return z_new
