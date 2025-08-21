import torch
from enum import Enum
import abc
from .scheduler import MomentumScheduler
from typing import Callable, Union


class MomentumType(Enum):
    Nesterov = "Nesterov"
    Polyak = "Polyak"

class OptimizerStep(abc.ABC):
    def __init__(self, tracking: bool=False):
        self._residual_tracking = tracking
        if self._residual_tracking:
            self._residual = None
    
    @abc.abstractmethod
    def step(self, x_curr: torch.Tensor) -> torch.Tensor: pass
    
    def __call__(self, *args, **kwargs):
        return self.step(*args, **kwargs)
    
    @property
    def residual(self):
        if not hasattr(self, '_residual'):
            raise RuntimeError(
                "residual tracking is disabled for this step instance."
            )
        if self._residual is None:
            raise AttributeError("`residual` must be set before referencing")
        return self._residual
    
    @property
    def residual_tracking(self):
        return self._residual_tracking
    
    def is_markovian(self):
        return not hasattr(self, "_x_prev")

class MomentumStep(OptimizerStep):
    def __init__(
        self,
        momentum: torch.Tensor,
        strategy: MomentumType=MomentumType.Nesterov,
        momentum_scheduler: Union[None, Callable]=None
    ):
        super().__init__(tracking=False) # No tracking is needed.
        if not isinstance(strategy, MomentumType):
            raise TypeError(
                "parameter strategy can only be either "
                "MomentumType.Nesterov or MomentumType.Polyak"
            )
        self.momentum = momentum
        self.strategy = strategy
        self.momentum_scheduler = momentum_scheduler
    
    def step(
        self, x_curr: torch.Tensor, x_prev: torch.Tensor
    ) -> torch.Tensor:
        if self.momentum_scheduler:
            self.momentum = self.momentum_scheduler()
        x_new = self.momentum * (x_curr - x_prev)   # torch.Tensor Operation
        if self.strategy == MomentumType.Nesterov:
            x_new = x_curr + x_new                  # torch.Tensor Operation
        return x_new
    
    def is_markovian(self):
        return False

class GDStep(OptimizerStep):
    def __init__(
        self, 
        stepsize: torch.Tensor,
        grad_map: Callable,
        stepsize_scheduler: Union[None, Callable]=None,
        linesearch=True,
        tracking: bool=False
    ):
        super().__init__(tracking=tracking)
        self.stepsize = stepsize
        self.grad_map = grad_map
        self.stepsize_scheduler = stepsize_scheduler
        self.linesearch = linesearch
    
    def step(self, x_curr: torch.Tensor) -> torch.Tensor:
        direction = -self.grad_map(x_curr)
        self._set_stepsize(x_curr, direction)
        x_new = x_curr + self.stepsize * direction  # torch.Tensor Operation
        if self._residual_tracking:
            self._residual = x_new - x_curr         # torch.Tensor Operation
        return x_new
    
    def _set_stepsize(self, x_curr, direction):
        if not self.stepsize_scheduler:
            return
        if self.linesearch:
            self.stepsize = self.stepsize_scheduler(x_curr, direction)
        else:
            self.stepsize = self.stepsize_scheduler()

class ProxStep(OptimizerStep):
    def __init__(
        self,
        prox_map: Callable
    ):
        super().__init__(tracking=False) # No tracking is needed.
        self.prox_map = prox_map
    
    def step(self, x_curr:torch.Tensor) -> torch.Tensor:
        return self.prox_map(x_curr)

class PolyakStep(OptimizerStep):
    def __init__(
        self,
        stepsize: torch.Tensor,
        momentum: torch.Tensor,
        grad_map: Callable,
        tracking: bool=False
    ):
        super().__init__(tracking=False) # Uses the tracking of GDStep
        self.gd_step = GDStep(
            stepsize, grad_map, tracking=tracking
        )
        self.mm_step = MomentumStep(momentum, strategy=MomentumType.Polyak)
        self._x_prev = None
    
    def step(self, x_curr):
        x_new = self.gd_step(x_curr)
        if self._x_prev is not None:
            # vvvvv torch.Tensor Operation vvvvv
            x_new = x_new + self.mm_step(x_curr, self._x_prev)
        self._x_prev = x_curr
        return x_new
    
    @property
    def residual(self):
        return self.gd_step.residual
    
    @property
    def residual_tracking(self):
        return self.gd_step.residual_tracking
    
    @property
    def x_prev(self):
        if self._x_prev is None:
            raise RuntimeError("`x_prev` accessed before assignment.")
        return self._x_prev
    
    @x_prev.setter
    def x_prev(self, x):
        self._x_prev = x

class NesterovStep(OptimizerStep):
    def __init__(
        self,
        stepsize: torch.Tensor,
        grad_map: Callable,
        momentum_scheduler: Union[None, Callable]=None,
        tracking: bool=False
    ):
        super().__init__(tracking=False) # Uses the tracking of GDStep
        if momentum_scheduler is None:
            momentum_scheduler = MomentumScheduler()
        self.gd_step = GDStep(
            stepsize, grad_map, tracking=tracking
        )
        self.mm_step = MomentumStep(
            momentum=None, momentum_scheduler=momentum_scheduler
        )
        self._x_prev = None
    
    def step(self, x_curr):
        if self._x_prev is None:
            x_mm = x_curr
        else:
            x_mm = self.mm_step(x_curr, self._x_prev)
        x_new = self.gd_step(x_mm)
        self._x_prev = x_curr
        return x_new
    
    @property
    def residual(self):
        return self.gd_step.residual
    
    @property
    def residual_tracking(self):
        return self.gd_step.residual_tracking
    
    @property
    def x_prev(self):
        if self._x_prev is None:
            raise RuntimeError("`x_prev` accessed before assignment.")
        return self._x_prev
    
    @x_prev.setter
    def x_prev(self, x):
        self._x_prev = x

class ProxGradStep(OptimizerStep):
    def __init__(
        self,
        stepsize: torch.Tensor,
        grad_map: Callable,
        prox_map: Callable,
        tracking: bool=False
    ):
        super().__init__(tracking=tracking)
        self.gd_step = GDStep(stepsize, grad_map, tracking=False)
        self.prox_step = ProxStep(prox_map)
    
    def step(self, x_curr):
        x_new = self.prox_step(self.gd_step(x_curr))
        if self._residual_tracking:
            self._residual = x_new - x_curr         # torch.Tensor Operation
        return x_new

class FISTAStep(OptimizerStep):
    def __init__(
        self,
        stepsize: torch.Tensor,
        grad_map: Callable,
        prox_map: Callable,
        momentum_scheduler: Union[None, Callable]=None,
        tracking: bool=False
    ):
        super().__init__(tracking=False) # Uses the tracking of ProxGradStep
        if momentum_scheduler is None:
            momentum_scheduler = MomentumScheduler()
        self.pgd_step = ProxGradStep(
            stepsize, grad_map, prox_map, tracking=tracking
        )
        self.mm_step = MomentumStep(
            momentum=None, momentum_scheduler=momentum_scheduler
        )
        self._x_prev = None
    
    def step(self, x_curr):
        if self._x_prev is None:
            self._x_prev = x_curr
        x_new = self.pgd_step(self.mm_step(x_curr, self._x_prev))
        self._x_prev = x_curr
        return x_new
    
    @property
    def residual(self):
        return self.pgd_step.residual
    
    @property
    def residual_tracking(self):
        return self.pgd_step.residual_tracking
    
    @property
    def x_prev(self):
        if self._x_prev is None:
            raise RuntimeError("`x_prev` accessed before assignment.")
        return self._x_prev
    
    @x_prev.setter
    def x_prev(self, x):
        self._x_prev = x


class PDHGPartialStep(OptimizerStep):
    def __init__(
        self,
        stepsize: torch.Tensor,
        lin_op: Callable,
        prox_map: Callable,
        residual_tracking: bool=False
    ):
        super().__init__(tracking=residual_tracking)
        self.stepsize = stepsize
        self.lin_op = lin_op
        self.prox_step = ProxStep(prox_map)
    
    def step(
        self, inp_curr: torch.Tensor, oth_curr: torch.Tensor
    ) -> torch.Tensor:
        # vvvvv torch.Tensor Operation vvvvv
        return self.prox_step(
            inp_curr - self.stepsize * self.lin_op(oth_curr)
        )


class PDHGStep(OptimizerStep):
    def __init__(
        self,
        stepsize_primal: torch.Tensor,
        stepsize_dual: torch.Tensor,
        prox_map_primal: Callable,
        prox_map_dual: Callable,
        lin_op: Callable,
        lin_op_adj: Callable=None,
        residual_tracking: bool=False
    ):
        super().__init__(residual_tracking)
        if lin_op_adj is None:
            lin_op_adj = ... # TODO
        self.primal_step = PDHGPartialStep(
            stepsize_primal, prox_map_primal, lin_op_adj
        )
        self.dual_step = PDHGPartialStep(
            stepsize_dual, prox_map_dual, lin_op
        )
    
    def step(
        self, x_primal_curr: torch.Tensor, x_dual_curr: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        x_primal_next = self.primal_step(x_primal_curr, x_dual_curr)
        # vvvvv torch.Tensor Operation vvvvv
        x_primal_curr = 2 * x_primal_next - x_primal_curr
        x_dual_next = self.dual_step(x_dual_curr, -x_primal_curr)
        return x_primal_next, x_dual_next
    