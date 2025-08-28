import torch
from tqdm import tqdm
from typing import Callable, Tuple, Union


from .variable import Variable, VariableType
from .step import OptimizerStep
from .step import GDStep, PolyakStep, NesterovStep
from .step import ProxGradStep, FISTAStep, PDHGStep
from .util import OptimizationResult, default_metric

class Optimizer:
    def __init__(
        self,
        step: OptimizerStep,
        tol: float=1e-5,
        iters: int=100,
        metric: None | Callable=None,
        store_history: bool=False,
        verbose: bool=False
    ):
        self.step = step
        self.tol = tol
        self.iters = iters
        self.metric = metric
        self.verbose = verbose
        self.store_history = store_history
        self.history = [] if store_history else None
        self.result = None
    
    def __call__(
        self, x_init: Union[Variable, VariableType], iters: None | int=None
    ) -> Union[Variable, VariableType]:
        return self.run(x_init, iters)
    
    @Variable.ensure
    def run(self, x_init: Variable, iters: None | int=None) -> Variable:
        
        x_curr = x_init.clone()
        if self.store_history:
            self.history.append(x_curr.clone())
        
        if iters is not None:
            self.iters = iters
        
        converged = False
        pbar = range(self.iters)
        if self.verbose:
            pbar = tqdm(pbar)
        for k in pbar:
            x_curr = self.step(x_curr)
            
            if self.store_history:
                self.history.append(x_curr.clone())
            
            if not self.metric and not self.step.residual_tracking:
                continue
            
            if self.step.residual_tracking:
                # vvvvv torch.Tensor Operation vvvvv
                metric_val = self.step.residual.norm()
            else:
                metric_val = self.metric(x_curr.data)

            if self.verbose:
                pbar.set_description(f"{metric_val:.6e}")
            
            if metric_val < self.tol:
                converged = True
                break
        
        self.result = OptimizationResult(
            solution=x_curr,
            iterations=k + 1,
            metric=metric_val if self.metric else None,
            converged=converged
        )

        return x_curr


class GradientDescent(Optimizer):
    def __init__(
        self,
        stepsize: torch.Tensor,
        grad_map: Callable,
        tol: float=1e-5,
        iters: int=100,
        metric: None | str | Callable=None,
        store_history: bool=False,
        verbose: bool=False
    ):
        tracking = metric == "default"
        step = GDStep(stepsize, grad_map, tracking=tracking)
        super().__init__(step, tol, iters, metric, store_history, verbose)


class HeavyBall(Optimizer):
    def __init__(
        self,
        stepsize: torch.Tensor,
        momentum: torch.Tensor,
        grad_map: Callable,
        tol: float=1e-5,
        iters: int=100,
        metric: None | str | Callable=None,
        store_history: bool=False,
        verbose: bool=False
    ):
        tracking = metric == "default"
        step = PolyakStep(
            stepsize, momentum, grad_map, tracking=tracking
        )
        super().__init__(step, tol, iters, metric, store_history, verbose)

class AcceleratedGradient(Optimizer):
    def __init__(
        self,
        stepsize,
        grad_map,
        momentum_scheduler: None | Callable=None,
        tol: float=1e-5,
        iters: int=100,
        metric: None | str | Callable=None,
        store_history = False,
        verbose = False
    ):
        tracking = metric == "default"
        step = NesterovStep(
            stepsize, grad_map, momentum_scheduler=momentum_scheduler,
            tracking=tracking
        )
        super().__init__(step, tol, iters, metric, store_history, verbose)
    
    def restart(self):
        self.step.restart()


class ProximalGradient(Optimizer):
    def __init__(
        self,
        stepsize: torch.Tensor,
        grad_map: Callable,
        prox_map: Callable,
        tol: float=1e-5,
        iters: int=100,
        metric: None | str | Callable=None,
        store_history = False,
        verbose = False
    ):
        tracking = metric == "default"
        step = ProxGradStep(
            stepsize, grad_map, prox_map, tracking=tracking
        )
        super().__init__(step, tol, iters, metric, store_history, verbose)


class FISTA(Optimizer):
    def __init__(
        self,
        stepsize: torch.Tensor,
        grad_map: Callable,
        prox_map: Callable,
        momentum_scheduler: None | Callable=None,
        tol: float=1e-5,
        iters: int=100,
        metric: None | str | Callable=None,
        store_history = False,
        verbose = False
    ):
        tracking = metric == "default"
        step = FISTAStep(
            stepsize, grad_map, prox_map, tracking=tracking,
            momentum_scheduler=momentum_scheduler
        )
        super().__init__(step, tol, iters, metric, store_history, verbose)
    
    def restart(self):
        self.step.restart()


class PDHG(Optimizer):
    def __init__(
        self,
        stepsize_primal: torch.Tensor,
        stepsize_dual: torch.Tensor,
        prox_map_primal: Callable,
        prox_map_dual: Callable,
        lin_op: Callable,
        lin_op_adj: Callable=None,
        zero_el: Union[None, torch.Tensor, Tuple[torch.Tensor, ...]]=None,
        tol: float=1e-5,
        iters: int=100,
        metric: None | str | Callable=None,
        store_history = False,
        verbose = False
    ):
        """Eactly one out of `zero_el` and `lin_op_adj` should be `None`.
        """
        step = PDHGStep(
            stepsize_primal, stepsize_dual, prox_map_primal, prox_map_dual,
            lin_op, lin_op_adj, zero_el, tracking=(metric == "default")
        )
        super().__init__(step, tol, iters, metric, store_history, verbose)

