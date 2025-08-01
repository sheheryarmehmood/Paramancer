import torch
from typing import Callable
from paramancer.optim.step import GDStep, PolyakStep, NesterovStep
from paramancer.optim.step import ProxGradStep, FISTAStep
from paramancer.optim.scheduler import MomentumScheduler

class SmoothParamStepMixin:
    def __init__(self, grad_map_prm: Callable):
        self.u_given = None
        self.grad_map_prm = grad_map_prm

    def _grad_map(self, x: torch.Tensor) -> torch.Tensor:
        if self.u_given is None:
            raise ValueError("'u_given' must be set before calling step.")
        return self.grad_map_prm(x, self.u_given)

    def step(
        self,
        x_curr: torch.Tensor,
        u_given: torch.Tensor | tuple[torch.Tensor]
    ) -> torch.Tensor:
        self.u_given = u_given
        return super().step(x_curr)

    def __call__(
        self,
        x_curr: torch.Tensor,
        u_given: torch.Tensor | tuple[torch.Tensor]
    ) -> torch.Tensor:
        return self.step(x_curr, u_given)

class GDParamStep(SmoothParamStepMixin, GDStep):
    def __init__(self, stepsize: torch.Tensor, grad_map_prm: Callable):
        SmoothParamStepMixin.__init__(self, grad_map_prm)
        GDStep.__init__(self, stepsize, self._grad_map)

class PolyakParamStep(SmoothParamStepMixin, PolyakStep):
    def __init__(
        self,
        stepsize: torch.Tensor,
        momentum: torch.Tensor,
        grad_map_prm: Callable
    ):
        SmoothParamStepMixin.__init__(self, grad_map_prm)
        PolyakStep.__init__(self, stepsize, momentum, self._grad_map)

class NesterovParamStep(SmoothParamStepMixin, NesterovStep):
    def __init__(
        self,
        stepsize: torch.Tensor,
        grad_map_prm: Callable,
        momentum_scheduler: Callable=MomentumScheduler()
    ):
        SmoothParamStepMixin.__init__(self, grad_map_prm)
        NesterovStep.__init__(
            self, stepsize, self._grad_map,
            momentum_scheduler=momentum_scheduler
        )


class NonSmoothParamStepMixin(SmoothParamStepMixin):
    def __init__(self, grad_map_prm: Callable, prox_map_prm: Callable):
        super().__init__(grad_map_prm)
        self.prox_map_prm = prox_map_prm
    
    def _prox_map(self, x: torch.Tensor) -> torch.Tensor:
        if self.u_given is None:
            raise ValueError("'u_given' must be set before calling step.")
        return self.prox_map_prm(x, self.u_given)

class ProxGradParamStep(NonSmoothParamStepMixin, ProxGradStep):
    def __init__(
        self,
        stepsize: torch.Tensor,
        grad_map_prm: Callable,
        prox_map_prm: Callable
    ):
        NonSmoothParamStepMixin.__init__(self, grad_map_prm, prox_map_prm)
        ProxGradStep.__init__(self, stepsize, self._grad_map, self._prox_map)

class FISTAParamStep(NonSmoothParamStepMixin, FISTAStep):
    def __init__(
        self,
        stepsize: torch.Tensor,
        grad_map_prm: Callable,
        prox_map_prm: Callable, 
        momentum_scheduler=MomentumScheduler()
    ):
        NonSmoothParamStepMixin.__init__(self, grad_map_prm, prox_map_prm)
        FISTAStep.__init__(
            self, stepsize, self._grad_map, self._prox_map,
            momentum_scheduler=momentum_scheduler
        )