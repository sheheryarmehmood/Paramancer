import torch
import pytest
from typing import TypeAlias

from paramancer.bloptim.step import FISTAParamMarkovStep, GDMarkovParamStep
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
    

def test_gradient_descent_vjp():
    def grad_lr(
        x: torch.Tensor,
        u: tuple[torch.Tensor, torch.Tensor]
    ) -> torch.Tensor:
        A, b = u
        return A.T @ (A @ x - b)
    
    def grad_des(
        x: torch.Tensor,
        u: tuple[torch.Tensor, torch.Tensor],
        ss: torch.Tensor
    ) -> torch.Tensor:
        return x - ss * grad_lr(x, u)
    
    def vjp_grad_lr(
        x: torch.Tensor,
        u: tuple[torch.Tensor, torch.Tensor],
        grad_out: torch.Tensor
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        A, b = u
        err = A @ x - b
        adj_out = A @ grad_out
        grad_x = A.T @ adj_out
        grad_A = torch.outer(err, grad_out) + torch.outer(adj_out, x)
        grad_b = -adj_out
        return grad_x, (grad_A, grad_b)
    
    def vjp_grad_des(
        x: torch.Tensor,
        u: tuple[torch.Tensor, torch.Tensor],
        grad_out: torch.Tensor,
        ss: torch.Tensor
    ):
        grad_x_lr, (grad_A_lr, grad_b_lr) = vjp_grad_lr(x, u, grad_out)
        grad_x = grad_out - ss * grad_x_lr
        grad_A = -ss * grad_A_lr
        grad_b = -ss * grad_b_lr
        return grad_x, (grad_A, grad_b)
    
    ss = torch.tensor(1.)
    M, N = 10, 5
    A, b = torch.rand(M, N), torch.randn(M)
    xk = torch.randn(N)
    grad_xkp = torch.randn(N)
    
    u = (A, b)
    xkp_pm = grad_des(xk, u, ss)
    grad_x, (grad_A, grad_b) = vjp_grad_des(xk, u, grad_xkp, ss)
    
    u_pm = (A, b)
    pm_step = GDMarkovParamStep(
        ss, grad_map_prm=grad_lr, u_in=(A, b)
    )
    xkp_pm = pm_step(xk, u_pm)
    grad_x_pm, (grad_A_pm, grad_b_pm) = pm_step.vjp(xk, u_pm, grad_xkp)
    
    assert torch.allclose(grad_x, grad_x_pm)
    assert torch.allclose(grad_A, grad_A_pm)
    assert torch.allclose(grad_b, grad_b_pm)


def test_gradient_descent_jvp():
    pass
        
    