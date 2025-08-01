import torch
from enum import Enum
import abc
from .scheduler import MomentumScheduler
from typing import Callable


class MomentumType(Enum):
    Nesterov = "Nesterov"
    Polyak = "Polyak"

class OptimizerStep(abc.ABC):
    def __init__(self):
        self.residual = None
    
    @abc.abstractmethod
    def step(self, x_curr: torch.Tensor) -> torch.Tensor: pass
    
    def __call__(self, *args, **kwargs):
        return self.step(*args, **kwargs)

class MomentumStep(OptimizerStep):
    def __init__(
        self,
        momentum: torch.Tensor,
        strategy: MomentumType=MomentumType.Nesterov,
        momentum_scheduler: None | Callable=None
    ):
        super().__init__()
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
        x_new = self.momentum * (x_curr - x_prev)
        if self.strategy == MomentumType.Nesterov:
            x_new = x_curr + x_new
        return x_new

class GDStep(OptimizerStep):
    def __init__(
        self, 
        stepsize: torch.Tensor,
        grad_map: Callable,
        stepsize_scheduler: None | Callable=None,
        linesearch=True
    ):
        super().__init__()
        self.stepsize = stepsize
        self.grad_map = grad_map
        self.stepsize_scheduler = stepsize_scheduler
        self.linesearch = linesearch
    
    def step(self, x_curr: torch.Tensor) -> torch.Tensor:
        direction = -self.grad_map(x_curr)
        if not self.stepsize_scheduler:
            return x_curr + self.stepsize * direction
        if self.linesearch:
            self.stepsize = self.stepsize_scheduler(x_curr, direction)
        else:
            self.stepsize = self.stepsize_scheduler()
        return x_curr + self.stepsize * direction

class ProxStep(OptimizerStep):
    def __init__(
        self,
        prox_map: Callable
    ):
        super().__init__()
        self.prox_map = prox_map
    
    def step(self, x_curr:torch.Tensor) -> torch.Tensor:
        return self.prox_map(x_curr)

class PolyakStep(OptimizerStep):
    def __init__(
        self,
        stepsize: torch.Tensor,
        momentum: torch.Tensor,
        grad_map: Callable
    ):
        super().__init__()
        self.gd_step = GDStep(stepsize, grad_map)
        self.mm_step = MomentumStep(momentum, strategy=MomentumType.Polyak)
        self.x_prev = None
    
    def step(self, x_curr):
        x_new = self.gd_step(x_curr)
        if self.x_prev is not None:
            x_new = x_new + self.mm_step(x_curr, self.x_prev)
        self.x_prev = x_curr
        return x_new

class NesterovStep(OptimizerStep):
    def __init__(
        self,
        stepsize: torch.Tensor,
        grad_map: Callable,
        momentum_scheduler: None | Callable=None
    ):
        super().__init__()
        if momentum_scheduler is None:
            momentum_scheduler = MomentumScheduler()
        self.gd_step = GDStep(stepsize, grad_map)
        self.mm_step = MomentumStep(
            momentum=None, momentum_scheduler=momentum_scheduler
        )
        self.x_prev = None
    
    def step(self, x_curr):
        if self.x_prev is None:
            self.x_prev = x_curr
        x_new = self.gd_step(self.mm_step(x_curr, self.x_prev))
        self.x_prev = x_curr
        return x_new

class ProxGradStep(OptimizerStep):
    def __init__(
        self,
        stepsize: torch.Tensor,
        grad_map: Callable,
        prox_map: Callable
    ):
        super().__init__()
        self.gd_step = GDStep(stepsize, grad_map)
        self.prox_step = ProxStep(prox_map)
    
    def step(self, x_curr):
        return self.prox_step(self.gd_step(x_curr))

class FISTAStep(OptimizerStep):
    def __init__(
        self,
        stepsize: torch.Tensor,
        grad_map: Callable,
        prox_map: Callable,
        momentum_scheduler: None | Callable=None
    ):
        super().__init__()
        if momentum_scheduler is None:
            momentum_scheduler = MomentumScheduler()
        self.nag_step = NesterovStep(
            stepsize, grad_map=grad_map, momentum_scheduler=momentum_scheduler
        )
        self.prox_step = ProxStep(prox_map)
    
    def step(self, x_curr):
        return self.prox_step(self.nag_step(x_curr))


class PDHGPartialStep(OptimizerStep):
    def __init__(
        self,
        stepsize: torch.Tensor,
        lin_op: Callable,
        prox_map: Callable
    ):
        self.stepsize = stepsize
        self.lin_op = lin_op
        self.prox_step = ProxStep(prox_map)
    
    def step(
        self, inp_curr: torch.Tensor, oth_curr: torch.Tensor
    ) -> torch.Tensor:
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
        lin_op_adj: Callable = None
    ):
        super().__init__()
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
        x_primal_curr = 2 * x_primal_next - x_primal_curr
        x_dual_next = self.dual_step(x_dual_curr, -x_primal_curr)
        return x_primal_next, x_dual_next
    