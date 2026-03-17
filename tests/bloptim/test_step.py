import torch
import pytest
from typing import TypeAlias

from paramancer.bloptim.step import FISTAParamMarkovStep
from paramancer.optim.step import FISTAStep


def test_fista_param_markov_step():
    Var: TypeAlias = tuple[torch.Tensor, ...]
    Prm: TypeAlias = tuple[torch.Tensor, ...]
    def grad_map(x: Var) -> Var:
        x1, x2 = x
        return x1 - u1, u2 * x2
    
    def prox_map1(x: Var):
        x1, x2 = x
        return x1 * 0.5, x2
    
    def prox_map2(x: Var):
        x1, x2 = x
        return x1 * u1, x2
    
    def grad_map_prm(x: Var, u: Prm) -> Var:
        x1, x2 = x
        u1, u2 = u
        return x1 - u1, u2 * x2
    
    def prox_map1_prm(x: Var):
        x1, x2 = x
        return x1 * 0.5, x2
    
    def prox_map2_prm(x: Var, u1: torch.Tensor):
        x1, x2 = x
        return x1 * u1, x2
    
    N1, N2 = 10, 5
    u1, u2 = torch.randn(N1), torch.rand(N2)
    x1, x2 = torch.randn(N1), torch.randn(N2)
    iters = 5
    ss = torch.Tensor([0.1]).squeeze()
    
    # First, we test using `prox_map1_prm`
    
    step1 = FISTAStep(ss, prox_map1, grad_map=grad_map)
    # vvvvv No indices supplied. Only `grad` defaults to 'all'.
    pm_step1 = FISTAParamMarkovStep(
        ss, prox_map1_prm, grad_map_prm=grad_map_prm
    )
    
    x_curr = x1.clone(), x2.clone()
    for _ in range(iters):
        x_curr = step1(x_curr)
    x1_new, x2_new = x_curr
    
    x_prev = x1.clone(), x2.clone()
    z_curr = x_curr, x_prev
    u = u1, u2
    for _ in range(iters):
        z_curr = pm_step1(z_curr, u)
    x1_new_pm, x2_new_pm = z_curr[0]
    
    assert torch.allclose(x1_new_pm, x1_new)
    assert torch.allclose(x2_new_pm, x2_new)
    
    
    # Similarly, we test using `prox_map2_prm`
    
    step2 = FISTAStep(ss, prox_map2, grad_map=grad_map)
    pm_step2 = FISTAParamMarkovStep(
        ss, prox_map2_prm, grad_map_prm=grad_map_prm,
        indices={"grad": "all", "prox": (0)}
    )
    
    x_curr = x1.clone(), x2.clone()
    for _ in range(iters):
        x_curr = step2(x_curr)
    x1_new, x2_new = x_curr
    
    x_prev = x1.clone(), x2.clone() 
    z_curr = x_curr, x_prev
    u = u1, u2
    for _ in range(iters):
        z_curr = pm_step2(z_curr, u)
    x1_new_pm, x2_new_pm = z_curr[0]
    
    assert torch.allclose(x1_new_pm, x1_new)
    assert torch.allclose(x2_new_pm, x2_new)
    

