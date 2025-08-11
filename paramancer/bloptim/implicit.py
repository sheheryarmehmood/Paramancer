import torch
from typing import Callable, Union, Tuple
from paramancer.bloptim.neumann import NeumannSeries
from paramancer.optim.step import OptimizerStep
import torch.autograd.functional as agF


class ImplicitDifferentiation:
    def __init__(
        self,
        algo_step: OptimizerStep,
        tol: float=1e-5,
        iters: int=100,
        metric: None | Callable=None,
        verbose: bool=False
    ):
        self.step = algo_step
        self._xmin = self._u_given = self._xmin_grad = None
        self._adjoint = None
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
        neumann = NeumannSeries(
            operator=self.operator, vector=self._xmin_grad, tol=self.tol, 
            iters=self.iters, metric=self.metric, verbose=self.verbose
        )
        self._adjoint = neumann(0*xmin)
        return agF.vjp(
            lambda u: self.step(self._xmin, u), self._u_given, self._adjoint
        )
    
    @property
    def adjoint(self):
        return self._adjoint
    
    def operator(self, y_curr):
        return agF.vjp(
            lambda x: self.step(x, self._u_given), self._xmin, y_curr
        )



from .param_step import GDParamStep
import torch.linalg as la


def grad_fn_prm(x, u):
    A, b = u
    return A.T @ (A @ x - b)

def minimizer(u):
    A, b = u
    return la.solve(A.T @ A, A.T @ b)

M, N, K = 1000, 100, 500
A = torch.rand(M, N)
b = torch.randn(M)
u_given = A, b
xmin = la.solve(A.T @ A, A.T @ b)
xmin_grad = torch.randn(xmin.shape)

u_grad_an = agF.vjp(minimizer, u_given, xmin_grad)

lip = la.matrix_norm(A.T @ A, ord=2)
gd_step = GDParamStep(stepsize=1/lip, grad_map_prm=grad_fn_prm)

imp_diff = ImplicitDifferentiation(gd_step, iters=1000)
u_grad_id = imp_diff(xmin, u_given,xmin_grad)


print(torch.dist(u_grad_id, u_grad_an))
