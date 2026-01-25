import torch
from enum import Enum
import abc
from .scheduler import MomentumScheduler
from typing import Callable, Union, Tuple
from .variable import Variable, VariableType
from paramancer.operators.linalg import adjoint


class MomentumType(Enum):
    Nesterov = "Nesterov"
    Polyak = "Polyak"

class OptimizerStep(abc.ABC):
    def __init__(self, tracking: bool=False):
        self._residual_tracking = tracking
        if self._residual_tracking:
            self._residual = None
        self._input_is_variable = True
    
    @abc.abstractmethod
    def step(self, x_curr: Variable) -> Variable: pass
    
    def __call__(
        self, x_curr: Union[Variable, VariableType]
    ) -> Union[Variable, VariableType]:
        return self.step(x_curr)
    
    @property
    def residual(self):
        if not hasattr(self, '_residual'):
            raise RuntimeError(
                "residual tracking is disabled for this step instance."
            )
        if self._residual is None:
            raise AttributeError("`residual` must be set before referencing")
        if self._input_is_variable:
            return self._residual
        else:
            return self._residual.data
    
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
    
    @Variable.ensure_var_input
    def step(self, z_curr: Variable) -> Variable:
        if self.momentum_scheduler:
            self.momentum = self.momentum_scheduler()
        x_new = self.momentum * (z_curr.current - z_curr.previous)
        if self.strategy == MomentumType.Nesterov:
            x_new = z_curr.current + x_new
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
        self.grad_map = Variable.wrap(grad_map)
        self.stepsize_scheduler = stepsize_scheduler
        self.linesearch = linesearch
    
    @Variable.ensure_var_input
    def step(self, x_curr: Variable) -> Variable:
        direction = -self.grad_map(x_curr)
        self._set_stepsize(x_curr, direction)
        x_new = x_curr + self.stepsize * direction
        if self._residual_tracking:
            self._residual = x_new - x_curr
        return x_new
    
    def _set_stepsize(self, x_curr: Variable, direction: Variable):
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
        self.prox_map = Variable.wrap(prox_map)
    
    @Variable.ensure_var_input
    def step(self, x_curr: Variable) -> Variable:
        return self.prox_map(x_curr)

class AffineStep(OptimizerStep):
    def __init__(
        self,
        operator: Callable,
        vector: VariableType,
        residual_tracking: bool=False
    ):
        super().__init__(tracking=residual_tracking)
        self.operator = Variable.wrap(operator)
        self.vector = Variable(vector)
    
    @Variable.ensure_var_input
    def step(self, x_curr: Variable) -> Variable:
        x_new = self.operator(x_curr) + self.vector
        if self._residual_tracking:
            self._residual = x_new - x_curr
        return x_new

class PolyakStep(OptimizerStep):
    def __init__(
        self,
        stepsize: torch.Tensor,
        momentum: torch.Tensor,
        grad_map: Callable,
        tracking: bool=False
    ):
        super().__init__(tracking=False) # Uses the tracking of GDStep
        self.gd_step = GDStep(stepsize, grad_map, tracking=tracking)
        self.mm_step = MomentumStep(momentum, strategy=MomentumType.Polyak)
        self._x_prev = None
    
    @Variable.ensure_var_input
    def step(self, x_curr: Variable) -> Variable:
        x_new = self.gd_step(x_curr)
        if self._x_prev is not None:
            z_curr = Variable.from_momentum(x_curr, self._x_prev)
            # vvvvv torch.Tensor Operation vvvvv
            x_new = x_new + self.mm_step(z_curr)
        self._x_prev = x_curr
        return x_new
    
    @property
    def residual(self):
        self.gd_step._input_is_variable = self._input_is_variable
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
    
    @Variable.ensure_var_input
    def step(self, x_curr: Variable) -> Variable:
        if self._x_prev is None:
            self._x_prev = x_curr
        z_curr = Variable.from_momentum(x_curr, self._x_prev)
        x_new = self.gd_step(self.mm_step(z_curr))
        self._x_prev = x_curr
        return x_new
    
    @property
    def residual(self):
        self.gd_step._input_is_variable = self._input_is_variable
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
    
    def restart(self):
        self._x_prev = None
        self.mm_step.momentum_scheduler.restart()

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
    
    @Variable.ensure_var_input
    def step(self, x_curr: Variable) -> Variable:
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
    
    @Variable.ensure_var_input
    def step(self, x_curr: Variable) -> Variable:
        if self._x_prev is None:
            self._x_prev = x_curr
        z_curr = Variable.from_momentum(x_curr, self._x_prev)
        x_new = self.pgd_step(self.mm_step(z_curr))
        self._x_prev = x_curr
        return x_new
    
    @property
    def residual(self):
        self.pgd_step._input_is_variable = self._input_is_variable
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
    
    def restart(self):
        self._x_prev = None
        self.mm_step.momentum_scheduler.restart()


class PDHGPartialStep(OptimizerStep):
    def __init__(
        self,
        stepsize: torch.Tensor,
        prox_map: Callable,
        lin_op: Callable
    ):
        super().__init__(tracking=False) # No tracking is needed.
        self.stepsize = stepsize
        self.lin_op = Variable.wrap(lin_op)
        self.prox_step = ProxStep(prox_map)
    
    def __call__(
        self,
        inp_curr: Union[Variable, VariableType], 
        oth_curr: Union[Variable, VariableType]
    ) -> Union[Variable, VariableType]:
        return self.step(inp_curr, oth_curr)
    
    @Variable.ensure_var_input
    def step(
        self, inp_curr: Variable, oth_curr: Variable
    ) -> Variable:
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
        zero_el: Union[None, torch.Tensor, Tuple[torch.Tensor, ...]]=None,
        residual_tracking: bool=False
    ):
        super().__init__(residual_tracking)
        if lin_op_adj is None:
            lin_op_adj = self._setup_lin_op_adj(lin_op, zero_el)
        self.primal_step = PDHGPartialStep(
            stepsize_primal, prox_map_primal, lin_op_adj
        )
        self.dual_step = PDHGPartialStep(
            stepsize_dual, prox_map_dual, lin_op
        )
    
    @Variable.ensure_var_input
    def step(self, z_curr: Variable) -> Variable:
        x_prim = self.primal_step(z_curr.primal, z_curr.dual)
        x_prim_mm = 2 * x_prim - z_curr.primal
        x_dual = self.dual_step(z_curr.dual, -x_prim_mm)
        z_new = Variable.from_pdhg(x_prim, x_dual)
        if self._residual_tracking:
            self._residual = z_new - z_curr         # torch.Tensor Operation
        return z_new
    
    def _setup_lin_op_adj(self, lin_op, zero_el):
        if zero_el is None:
            raise ValueError(
                "Either the adjoint linear map must be provided beforehand or"
                " the zero element must be given for automatic computation."
            )
        if isinstance(zero_el, tuple):
            lin_op_ = lambda *args: lin_op(args)
        return adjoint(lin_op_, zero_el)
        