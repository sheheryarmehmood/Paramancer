from __future__ import annotations

import abc
from enum import Enum
from typing import Any

from ..operators.grad import gradient
from ..operators.linalg import adjoint
from ..variable.flat import FlatVar
from ..variable.pair import PairVar
from ..variable.types import (
    AlgoVarLike,
    FlatRawVarType,
    FlatVarLike,
    LinOpType,
    MomentumSchedType,
    PGradMapType,
    ProxMapType,
    PSmoothObjType,
    RawFlatLinOpType,
    ScalarLike,
    StepsizeSchedType,
)
from ..variable.util import (
    as_flat_var,
    as_pair_var,
    is_flat_var,
    is_pair_raw_var,
    is_pair_var,
)
from .scheduler import MomentumScheduler
from .util import ensure_flat_input, ensure_pair_input


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
        d_curr = -self._wrapped_grad_map(x_curr, *args, **kwargs)
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
            self.grad_map = gradient(smooth_obj)
        else:
            self.grad_map = grad_map

    def _set_stepsize(self, x_curr: FlatVar, d_curr: FlatVar) -> None:
        if not self.stepsize_scheduler:
            return
        if self.linesearch:
            self.stepsize = self.stepsize_scheduler(x_curr.data, d_curr.data)
        else:
            self.stepsize = self.stepsize_scheduler()

    @property
    def grad_map(self):
        return self._raw_grad_map

    @grad_map.setter
    def grad_map(self, grad_map):
        self._raw_grad_map = grad_map
        self._wrapped_grad_map = _wrap_flat_map(grad_map)


class ProxStep(OptimizerStep):
    def __init__(self, prox_map: ProxMapType):
        super().__init__(tracking=False)
        self.prox_map = prox_map

    @ensure_flat_input
    def step(self, x_curr: FlatVar) -> FlatVar:
        return self._wrapped_prox_map(x_curr)

    @property
    def prox_map(self):
        return self._raw_prox_map

    @prox_map.setter
    def prox_map(self, prox_map):
        self._raw_prox_map = prox_map
        self._wrapped_prox_map = _wrap_flat_map(prox_map)


class AffineStep(OptimizerStep):
    def __init__(
        self,
        lin_op: LinOpType,
        vector: AlgoVarLike,
        tracking: bool = False,
    ):
        super().__init__(tracking=tracking)
        self._set_vector(vector, allow_mode_change=True)
        self.lin_op = lin_op

    def step(
        self, x_curr: AlgoVarLike, *args: Any, **kwargs: Any
    ) -> AlgoVarLike:
        if self._wrapped_mode:
            x_new = (
                self._wrapped_lin_op(x_curr, *args, **kwargs) +
                self._wrapped_vector
            )
            if self._residual_tracking:
                self._residual = x_new - x_curr
            return x_new
        x_var = as_pair_var(x_curr) if self._pair_mode else as_flat_var(x_curr)
        x_new = (
            self._wrapped_lin_op(x_var, *args, **kwargs) +
            self._wrapped_vector
        )
        if self._residual_tracking:
            self._residual = x_new - x_var
        return x_new.data

    def replace(
            self,
            lin_op: LinOpType,
            vector: AlgoVarLike,
        ) -> None:
        self._set_vector(vector, allow_mode_change=True)
        self.lin_op = lin_op

    @property
    def vector(self) -> AlgoVarLike:
        return self._raw_vector

    @vector.setter
    def vector(self, vector) -> None:
        self._set_vector(vector, allow_mode_change=False)

    @property
    def lin_op(self) -> LinOpType:
        return self._raw_lin_op

    @lin_op.setter
    def lin_op(self, lin_op) -> None:
        self._raw_lin_op = lin_op
        if self._wrapped_mode:
            self._wrapped_lin_op = lin_op
        elif self._pair_mode:
            self._wrapped_lin_op = _wrap_pair_map(lin_op)
        else:
            self._wrapped_lin_op = _wrap_flat_map(lin_op)

    def _set_vector(self, vector, allow_mode_change: bool) -> None:
        wrapped_mode = is_flat_var(vector) or is_pair_var(vector)
        pair_mode = is_pair_raw_var(vector) or is_pair_var(vector)

        mode_exists = hasattr(self, "_wrapped_mode")
        mode_changed = (
            mode_exists
            and (
                self._wrapped_mode != wrapped_mode
                or self._pair_mode != pair_mode
            )
        )

        if mode_changed and not allow_mode_change:
            raise ValueError(
                "Changing the vector structure also requires replacing the "
                "linear operator. Use `replace()` to set both together."
            )

        self._wrapped_mode = wrapped_mode
        self._pair_mode = pair_mode
        self._raw_vector = vector

        if wrapped_mode:
            self._wrapped_vector = vector
        elif pair_mode:
            self._wrapped_vector = as_pair_var(vector)
        else:
            self._wrapped_vector = as_flat_var(vector)


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

    @property
    def grad_map(self):
        return self.gd_step.grad_map

    @grad_map.setter
    def grad_map(self, grad_map):
        self.gd_step.grad_map = grad_map


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

    @property
    def grad_map(self):
        return self.gd_step.grad_map

    @grad_map.setter
    def grad_map(self, grad_map):
        self.gd_step.grad_map = grad_map

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

    @property
    def grad_map(self):
        return self.gd_step.grad_map

    @grad_map.setter
    def grad_map(self, grad_map):
        self.gd_step.grad_map = grad_map

    @property
    def prox_map(self):
        return self.prox_step.prox_map

    @prox_map.setter
    def prox_map(self, prox_map):
        self.prox_step.prox_map = prox_map


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

    @property
    def grad_map(self):
        return self.pgd_step.grad_map

    @grad_map.setter
    def grad_map(self, grad_map):
        self.pgd_step.grad_map = grad_map

    @property
    def prox_map(self):
        return self.pgd_step.prox_map

    @prox_map.setter
    def prox_map(self, prox_map):
        self.pgd_step.prox_map = prox_map

    def restart(self):
        self._x_prev = None
        self.mm_step.momentum_scheduler.restart()

class PDHGPartialStep(OptimizerStep):
    def __init__(
        self,
        stepsize: ScalarLike,
        prox_map: ProxMapType,
        lin_op: RawFlatLinOpType
    ) -> None:
        super().__init__(tracking=False)
        self.stepsize = stepsize
        self.lin_op = lin_op
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
        lin_out = self._wrapped_lin_op(oth_var, *args, **kwargs)
        return self.prox_step(inp_curr - self.stepsize * lin_out)

    @property
    def lin_op(self) -> RawFlatLinOpType:
        return self._raw_lin_op

    @lin_op.setter
    def lin_op(self, lin_op: RawFlatLinOpType):
        self._raw_lin_op = lin_op
        self._wrapped_lin_op = _wrap_flat_map(lin_op)

    @property
    def prox_map(self) -> ProxMapType:
        return self.prox_step.prox_map

    @prox_map.setter
    def prox_map(self, prox_map: ProxMapType):
        self.prox_step.prox_map = prox_map


class PDHGStep(OptimizerStep):
    def __init__(
        self,
        stepsize_primal: ScalarLike,
        stepsize_dual: ScalarLike,
        prox_map_primal: ProxMapType,
        prox_map_dual: ProxMapType,
        lin_op: RawFlatLinOpType,
        lin_op_adj: RawFlatLinOpType | None = None,
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
        self.dual_step = PDHGPartialStep(
            stepsize_dual, prox_map_dual, lin_op
        )

    @ensure_pair_input
    def step(self, z_curr: PairVar, *args: Any, **kwargs: Any) -> PairVar:
        x_prim = self.primal_step(z_curr.first, z_curr.second, *args, **kwargs)
        x_prim_mm = 2 * x_prim - z_curr.first
        x_dual = self.dual_step(z_curr.second, -x_prim_mm, *args, **kwargs)
        z_new = PairVar(x_prim, x_dual)
        if self._residual_tracking:
            self._residual = z_new - z_curr
        return z_new

    def replace_lin_op(
        self,
        lin_op: RawFlatLinOpType,
        lin_op_adj: RawFlatLinOpType | None = None,
        zero_el: FlatRawVarType | None = None,
    ) -> None:
        if (lin_op_adj is None) == (zero_el is None):
            raise ValueError(
                "Either `zero_el` should be supplied or `lin_op_adj`, but not"
                " both. One of them must be set to `None`."
            )
        if lin_op_adj is None:
            lin_op_adj = adjoint(lin_op, zero_el)
        self.dual_step.lin_op = lin_op
        self.primal_step.lin_op = lin_op_adj

    @property
    def lin_op(self) -> RawFlatLinOpType:
        return self.dual_step.lin_op

    @property
    def lin_op_adj(self) -> RawFlatLinOpType:
        return self.primal_step.lin_op

    @property
    def prox_map_primal(self) -> ProxMapType:
        return self.primal_step.prox_map

    @prox_map_primal.setter
    def prox_map_primal(self, prox_map: ProxMapType) -> None:
        self.primal_step.prox_map = prox_map

    @property
    def prox_map_dual(self) -> ProxMapType:
        return self.dual_step.prox_map

    @prox_map_dual.setter
    def prox_map_dual(self, prox_map: ProxMapType) -> None:
        self.dual_step.prox_map = prox_map
