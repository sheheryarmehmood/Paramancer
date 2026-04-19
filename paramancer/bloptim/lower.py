import torch
from typing import Callable, Union, Tuple


from ..optim.step import OptimizerStep
from ..optim.optimizer import Optimizer as BaseOptimizer
from .implicit import ImplicitDifferentiation
from .implicit.step import GDParamMarkovStep
from ..variable.types import AlgoVarLike


class OptimizerID(torch.autograd.Function):
    @staticmethod
    def forward(ctx, *args) -> torch.Tensor:
        num_u, num_x = args[-2:]
        u_given, x_init = args[:num_u], args[num_u:num_u + num_x]
        if len(u_given) == 1:
            u_given = u_given[0]
        if len(x_init) == 1:
            x_init = x_init[0]
        (
            param_step,
            metric,
            tol_fwd,
            tol_bwd,
            iters_fwd,
            iters_bwd,
        ) = args[num_u + num_x : -2]
        tol_bwd = tol_fwd if tol_bwd is None else tol_bwd
        iters_bwd = iters_fwd if iters_bwd is None else iters_bwd
        param_step.u_given = u_given
        optimizer = Optimizer(param_step, tol_fwd, iters_fwd, metric=metric)
        if not param_step.is_markovian():
            x_init = (x_init, x_init)
        xmin = optimizer(x_init)
        xmin_sol = xmin[0] if not param_step.is_markovian() else xmin
        if torch.is_tensor(u_given):
            u_given = (u_given,)
        ctx.save_for_backward(xmin_sol, *u_given)
        ctx.param_step = param_step
        ctx.tol_bwd = tol_bwd
        ctx.iters_bwd = iters_bwd
        ctx.num_total_args = len(args)
        return xmin_sol
        
    
    @staticmethod
    def backward(
        ctx, xmin_grad: torch.Tensor
    ) -> Union[torch.Tensor, Tuple[torch.Tensor]]:
        xmin, *u_given = ctx.saved_tensors
        u_given = u_given[0] if len(u_given) == 1 else tuple(u_given)
        imp_diff = ImplicitDifferentiation(
            ctx.param_step, tol=ctx.tol_bwd, iters=ctx.iters_bwd, 
            metric="default"
        )
        u_grad = imp_diff(xmin, u_given, xmin_grad)
        if len(u_grad) == 1:
            u_grad = (u_grad,)
        return *u_grad, *((None,) * (ctx.num_total_args - len(u_grad)))


class Optimizer(BaseOptimizer):
    def __init__(
        self,
        step: OptimizerStep,
        tol: float=1e-5,
        iters: int=100,
        metric: Union[None, Callable]=None
    ):
        super().__init__(step, tol=tol, iters=iters, metric=metric)
    
    def __call__(
        self, x_init: AlgoVarLike, iters: None | int=None
    ) -> AlgoVarLike:
        return self.run(x_init, iters=iters)
        


# class GradientDescent:
#     def __init__(
#         self,
#         stepsize: torch.Tensor,
#         grad_map_prm: Callable,
#         tol: float=1e-5,
#         iters: int=100,
#         metric: Union[None, str, Callable]=None
#     ):
#         tracking = metric == "default"
#         param_step = GDMarkovParamStep(
#             stepsize, grad_map_prm, tracking=tracking
#         )
