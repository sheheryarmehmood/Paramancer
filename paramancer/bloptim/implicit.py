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
        metric: Union[None, str, Callable]=None,
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
        self.solve_linear_system()
        return self.compute_derivative_from_adjoint_state()
    
    @property
    def adjoint_state(self):
        return self._adjoint_state
    
    def setup(
        self,
        xmin: torch.Tensor,
        u_given: Union[torch.Tensor, Tuple[torch.Tensor]],
        xmin_grad: torch.Tensor
    ):
        self._xmin = xmin
        self._u_given = u_given
        self._xmin_grad = xmin_grad
    
    def solve_linear_system(self):
        if self.step.is_markovian():
            operator = self._operator_markovian
            vector = self._xmin_grad
        else:
            operator = self._operator_non_markovian
            vector = self._xmin_grad, self._xmin_grad
        self._adjoint_state = neumann_series(
            operator=operator, vector=vector, tol=self.tol, iters=self.iters, 
            metric=self.metric, verbose=self.verbose
        )
    
    def compute_derivative_from_adjoint_state(self):
        # -> Union[torch.Tensor, Tuple[torch.Tensor]]
        if self.step.is_markovian():
            func = self._markovian_u
        else:
            func = self._non_markovian_u
        return agF.vjp(
            func, self._u_given, self.adjoint_state
        )[1]
    
    def _operator_markovian(self, yc):
        return agF.vjp(self._markovian_x, self._xmin, yc)[1]
    
    def _operator_non_markovian(self, yc, yp):
        return agF.vjp(
            self._non_markovian_x, (self._xmin, self._xmin), (yc, yp)
        )[1]
    
    def _markovian_x(self, x_curr):
        return self.step(x_curr, self._u_given)
    
    def _non_markovian_x(self, x_curr, x_prev):
        self.step.x_prev = x_prev
        return self.step(x_curr, self._u_given), x_curr
    
    def _markovian_u(self, *u_given):
        if len(u_given) == 1:
            u_given = u_given[0]
        return self.step(self._xmin, u_given)
    
    def _non_markovian_u(self, *u_given):
        self.step.x_prev = self._xmin
        return self._markovian_u(*u_given), self._xmin


