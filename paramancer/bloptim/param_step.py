import torch
from typing import Callable, Union, Tuple
from paramancer.optim.step import GDStep, PolyakStep, NesterovStep
from paramancer.optim.step import ProxGradStep, FISTAStep
from paramancer.optim.scheduler import MomentumScheduler

class SmoothParamStepMixin:
    def __init__(self, grad_map_prm: Callable):
        self._u_given = None
        self.grad_map_prm = grad_map_prm

    def _grad_map(self, x: torch.Tensor) -> torch.Tensor:
        return self.grad_map_prm(x, self.u_given)
    
    def step(
        self,
        x_curr: torch.Tensor,
        u_given: Union[None, torch.Tensor, Tuple[torch.Tensor]]=None
    ) -> torch.Tensor:
        self.u_given = u_given
        return super().step(x_curr)

    def __call__(
        self,
        x_curr: torch.Tensor,
        u_given: Union[None, torch.Tensor, Tuple[torch.Tensor]]=None
    ) -> torch.Tensor:
        return self.step(x_curr, u_given)
    
    @property
    def u_given(self):
        if self._u_given is None:
            raise RuntimeError(
                "Parametric step called without setting u_given"
            )
        return self._u_given
    
    @u_given.setter
    def u_given(self, u_given):
        if u_given is not None:
            self._u_given = u_given
    

class GDParamStep(SmoothParamStepMixin, GDStep):
    def __init__(
        self,
        stepsize: torch.Tensor,
        grad_map_prm: Callable,
        stepsize_scheduler: Union[None, Callable]=None,
        linesearch=True,
        tracking: bool=False
    ):
        SmoothParamStepMixin.__init__(self, grad_map_prm)
        GDStep.__init__(
            self, stepsize, self._grad_map, linesearch=linesearch,
            stepsize_scheduler=stepsize_scheduler, tracking=tracking
        )

class PolyakParamStep(SmoothParamStepMixin, PolyakStep):
    def __init__(
        self,
        stepsize: torch.Tensor,
        momentum: torch.Tensor,
        grad_map_prm: Callable,
        tracking: bool=False
    ):
        SmoothParamStepMixin.__init__(self, grad_map_prm)
        PolyakStep.__init__(
            self, stepsize, momentum, self._grad_map, tracking=tracking
        )

class NesterovParamStep(SmoothParamStepMixin, NesterovStep):
    def __init__(
        self,
        stepsize: torch.Tensor,
        grad_map_prm: Callable,
        momentum_scheduler: Union[None, Callable]=None,
        tracking: bool=False
    ):
        SmoothParamStepMixin.__init__(self, grad_map_prm)
        NesterovStep.__init__(
            self, stepsize, self._grad_map, tracking=tracking,
            momentum_scheduler=momentum_scheduler
        )


class NonSmoothParamStepMixin(SmoothParamStepMixin):
    def __init__(self, grad_map_prm: Callable, prox_map_prm: Callable):
        super().__init__(grad_map_prm)
        self.prox_map_prm = prox_map_prm
    
    def _prox_map(self, x: torch.Tensor) -> torch.Tensor:
        return self.prox_map_prm(x, self.u_given)

class ProxGradParamStep(NonSmoothParamStepMixin, ProxGradStep):
    def __init__(
        self,
        stepsize: torch.Tensor,
        grad_map_prm: Callable,
        prox_map_prm: Callable,
        tracking: bool=False
    ):
        NonSmoothParamStepMixin.__init__(self, grad_map_prm, prox_map_prm)
        ProxGradStep.__init__(
            self, stepsize, self._grad_map, self._prox_map, tracking=tracking
        )

class FISTAParamStep(NonSmoothParamStepMixin, FISTAStep):
    def __init__(
        self,
        stepsize: torch.Tensor,
        grad_map_prm: Callable,
        prox_map_prm: Callable,
        momentum_scheduler: Union[None, Callable]=None,
        tracking: bool=False
    ):
        NonSmoothParamStepMixin.__init__(self, grad_map_prm, prox_map_prm)
        FISTAStep.__init__(
            self, stepsize, self._grad_map, self._prox_map, tracking=tracking,
            momentum_scheduler=momentum_scheduler
        )