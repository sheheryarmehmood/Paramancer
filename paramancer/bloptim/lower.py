import torch
from paramancer.optim.step import OptimizerStep
from paramancer.optim import Optimizer
from .implicit import ImplicitDifferentiation

from typing import Callable, Union, Tuple

class OptimizerID(torch.autograd.Function):
    @staticmethod
    def forward(ctx, *args) -> torch.Tensor:
        num_u = args[-1]
        u_given, x_init = args[:num_u], args[num_u]
        param_step, metric, tol_fwd, _, iters_fwd = args[num_u+1:-2]
        tol_bwd = tol_fwd if args[-4] is None else args[-4]
        iters_bwd = iters_fwd if args[-2] is None else args[-2]
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
    
