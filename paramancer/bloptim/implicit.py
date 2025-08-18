import torch
from typing import Callable, Union, Tuple
from paramancer.bloptim.neumann import neumann_series
from paramancer.optim.step import OptimizerStep
import torch.autograd.functional as agF


class ImplicitDifferentiation:
    def __init__(
        self,
        algo_step: OptimizerStep,
        tol: float=1e-5,
        iters: int=100,
        metric: None | str | Callable=None,
        verbose: bool=False
    ):
        self.step = algo_step
        self._xmin = self._u_given = self._xmin_grad = None
        self._adjoint_state = None
        self.tol = tol
        self.iters = iters
        self.metric = metric
        self.verbose = verbose
    
    def __call__(
        self,
        xmin: torch.Tensor,
        u_given: Union[torch.Tensor, Tuple[torch.Tensor]],
        xmin_grad: torch.Tensor
    ) -> Union[torch.Tensor, Tuple[torch.Tensor]]:
        self.setup(xmin, u_given, xmin_grad)
        return self.differentiate()
    
    def setup(
        self,
        xmin: torch.Tensor,
        u_given: Union[torch.Tensor, Tuple[torch.Tensor]],
        xmin_grad: torch.Tensor
    ):
        self._xmin = xmin
        self._u_given = u_given
        self._xmin_grad = xmin_grad
    
    def differentiate(self) -> Union[torch.Tensor, Tuple[torch.Tensor]]:
        self._adjoint_state = neumann_series(
            operator=self.operator, vector=self._xmin_grad, tol=self.tol, 
            iters=self.iters, metric=self.metric, verbose=self.verbose
        )
        return agF.vjp(
            lambda *u: self.step(self._xmin, u), self._u_given, self._adjoint_state
        )[1]
    
    @property
    def adjoint_state(self):
        return self._adjoint_state
    
    def operator(self, y_curr):
        return agF.vjp(
            lambda x: self.step(x, self._u_given), self._xmin, y_curr
        )[1]


