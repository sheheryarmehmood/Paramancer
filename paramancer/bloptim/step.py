import torch
from typing import Callable, Union
from paramancer.optim.variable import VariableType, Variable
from paramancer.bloptim.parameter import ParameterType, Parameter

from paramancer.optim.step import GDStep, PolyakStep, NesterovStep
from paramancer.optim.step import ProxGradStep, FISTAStep

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

class SmoothMarkovParamStepMixin:
    """
    Mixin for converting a base step into Markovian form which also
    accepts a parameter.

    Contract:
    - The base Step must implement:
      - `step(self, x_curr: Variable) -> Variable`
      - `is_markovian(self) -> bool`
      - If `is_markovian` is True, the Step must also define:
          - `x_prev` property with getter and setter
    """
    def __init__(self, grad_map_prm: Callable):
        self._u_given = None
        self.grad_map_prm = grad_map_prm

    @Parameter.ensure_var_param_inputs
    def step(
        self,
        x_curr: Variable,
        u_given: Union[None, Parameter]=None
    ) -> Variable:
        if not self.is_markovian():
            self.x_prev = x_curr.previous
            x_curr = x_curr.current
        self.u_given = u_given
        x_new = super().step(x_curr)
        if not self.is_markovian():
            x_new = Variable.from_momentum(x_new, x_curr)
        return x_new

    def __call__(
        self,
        x_curr: Union[Variable, VariableType],
        u_given: Union[None, Parameter, ParameterType]=None
    ) -> Union[Variable, VariableType]:
        return self.step(x_curr, u_given)
    
    def _grad_map(self, x: VariableType) -> VariableType:
        return self.grad_map_prm(x, self.u_given.data)
    
    @property
    def u_given(self) -> ParameterType:
        if self._u_given is None:
            raise RuntimeError(
                "Parametric step called without setting u_given"
            )
        return self._u_given
    
    @u_given.setter
    def u_given(self, u_given):
        if u_given is not None:
            self._u_given = u_given
    

class GDMarkovParamStep(SmoothMarkovParamStepMixin, GDStep):
    def __init__(
        self,
        stepsize: torch.Tensor,
        grad_map_prm: Callable,
        stepsize_scheduler: Union[None, Callable]=None,
        linesearch=True,
        tracking: bool=False
    ):
        SmoothMarkovParamStepMixin.__init__(self, grad_map_prm)
        GDStep.__init__(
            self, stepsize, self._grad_map, linesearch=linesearch,
            stepsize_scheduler=stepsize_scheduler, tracking=tracking
        )

class PolyakMarkovParamStep(SmoothMarkovParamStepMixin, PolyakStep):
    def __init__(
        self,
        stepsize: torch.Tensor,
        momentum: torch.Tensor,
        grad_map_prm: Callable,
        tracking: bool=False
    ):
        SmoothMarkovParamStepMixin.__init__(self, grad_map_prm)
        PolyakStep.__init__(
            self, stepsize, momentum, self._grad_map, tracking=tracking
        )

class NesterovMarkovParamStep(SmoothMarkovParamStepMixin, NesterovStep):
    def __init__(
        self,
        stepsize: torch.Tensor,
        grad_map_prm: Callable,
        momentum_scheduler: Union[None, Callable]=None,
        tracking: bool=False
    ):
        SmoothMarkovParamStepMixin.__init__(self, grad_map_prm)
        NesterovStep.__init__(
            self, stepsize, self._grad_map, tracking=tracking,
            momentum_scheduler=momentum_scheduler
        )


class NonSmoothMarkovParamStepMixin(SmoothMarkovParamStepMixin):
    def __init__(self, grad_map_prm: Callable, prox_map_prm: Callable):
        super().__init__(grad_map_prm)
        self.prox_map_prm = prox_map_prm
    
    def _prox_map(self, x: torch.Tensor) -> torch.Tensor:
        return self.prox_map_prm(x, self.u_given.data)

class ProxGradMarkovParamStep(NonSmoothMarkovParamStepMixin, ProxGradStep):
    def __init__(
        self,
        stepsize: torch.Tensor,
        grad_map_prm: Callable,
        prox_map_prm: Callable,
        tracking: bool=False
    ):
        NonSmoothMarkovParamStepMixin.__init__(self, grad_map_prm, prox_map_prm)
        ProxGradStep.__init__(
            self, stepsize, self._grad_map, self._prox_map, tracking=tracking
        )

class FISTAMarkovParamStep(NonSmoothMarkovParamStepMixin, FISTAStep):
    def __init__(
        self,
        stepsize: torch.Tensor,
        grad_map_prm: Callable,
        prox_map_prm: Callable,
        momentum_scheduler: Union[None, Callable]=None,
        tracking: bool=False
    ):
        NonSmoothMarkovParamStepMixin.__init__(self, grad_map_prm, prox_map_prm)
        FISTAStep.__init__(
            self, stepsize, self._grad_map, self._prox_map, tracking=tracking,
            momentum_scheduler=momentum_scheduler
        )