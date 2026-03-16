import torch
from typing import Callable, Union, Tuple


from ..optim.step import OptimizerStep
from ..optim.optimizer import Optimizer
from .implicit import ImplicitDifferentiation
from .step import GDMarkovParamStep
from ..variable import Variable
from ..variable.types import VariableType


class OptimizerID(torch.autograd.Function):
    @staticmethod
    def forward(ctx, *args) -> torch.Tensor:
        num_u, num_x = args[-2:]
        u_given, x_init = args[:num_u], args[num_u:num_u + num_x]
        param_step, metric, tol_fwd, _, iters_fwd = args[num_u+1:-3]
        tol_bwd = tol_fwd if args[-4] is None else args[-5]
        iters_bwd = iters_fwd if args[-2] is None else args[-3]
        param_step.u_given = u_given
        optimizer = Optimizer(param_step, tol_fwd, iters_fwd, metric=metric)
        xmin = optimizer(x_init)
        if torch.is_tensor(u_given): u_given = u_given,
        ctx.save_for_backward(xmin, *u_given)
        ctx.param_step = param_step
        ctx.tol_bwd = tol_bwd
        ctx.iters_bwd = iters_bwd
        return xmin
        
    
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
        if len(u_grad) == 1: u_grad = u_grad,
        return *u_grad, *((None,) * 8)


class Optimizer:
    def __init__(
        self,
        step: OptimizerStep,
        tol: float=1e-5,
        iters: int=100,
        metric: Union[None, Callable]=None
    ):
        self.step = step
        self.tol = tol
        self.iters = iters
        self.metric = metric
    
    def __call__(
        self, x_init: Union[Variable, VariableType], iters: None | int=None
    ) -> Union[Variable, VariableType]:
        return self.run(x_init, iters)
        


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
