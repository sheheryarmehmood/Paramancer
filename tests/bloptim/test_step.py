import torch
import torch.linalg as la
import pytest
from typing import TypeAlias

from paramancer.bloptim.implicit.step import (
    FISTAParamMarkovStep, GDParamMarkovStep, PolyakParamMarkovStep
)
from paramancer.optim.step import FISTAStep
from paramancer.optim.optimizer import NeumannSeries


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
        z_curr = pm_step1.markov_step(z_curr, u)
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
        z_curr = pm_step2.markov_step(z_curr, u)
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
    pm_step = GDParamMarkovStep(
        ss, grad_map_prm=grad_lr, u_in=(A, b)
    )
    xkp_pm = pm_step.markov_step(xk, u_pm)
    grad_x_pm, (grad_A_pm, grad_b_pm) = pm_step.vjp(xk, u_pm, grad_xkp)
    
    assert torch.allclose(grad_x, grad_x_pm)
    assert torch.allclose(grad_A, grad_A_pm)
    assert torch.allclose(grad_b, grad_b_pm)


def test_implicit_differentiation_with_polyak_step():
    def grad_map(x, u):
        A, b, r = u
        return A.T @ (A @ x - b) + r * x
    
    def minimizer(A, b, r):
        return la.solve(A.T @ A + r * torch.eye(A.shape[1]), A.T @ b)
    
    M, N = 5, 2
    A, b = torch.rand(M, N), torch.randn(M)
    r = torch.rand([])
    Q = A.T @ A +  r * torch.eye(N)
    A_tan, b_tan = torch.randn_like(A), torch.randn_like(b)
    r_tan = torch.randn_like(r)
    grad_xm = torch.randn(N)
    u = (A, b, r)
    u_tan = (A_tan, b_tan, r_tan)
    xm = minimizer(A, b, r)
    
    grad_u_anl = torch.func.vjp(minimizer, *u)[1](grad_xm)
    xm_tan_anl = torch.func.jvp(minimizer, u, u_tan)[1]
    
    lip, mu = la.norm(Q, ord=2), la.norm(Q, ord=-2)
    rt_lip, rt_mu = lip.sqrt(), mu.sqrt()
    ss = 4 / (rt_lip + rt_mu)**2
    mm = (rt_lip - rt_mu)**2 / (rt_lip + rt_mu)**2
    pm_step = PolyakParamMarkovStep(ss, mm, grad_map_prm=grad_map, u_in=u)
    zm = (xm, xm)
    
    neumann_jvp = NeumannSeries(
        lambda v: pm_step.jvp_var(zm, u, v),
        pm_step.jvp_par(zm, u, u_tan),
        tol=1e-4, iters=1000, metric="default"
    )
    zm_tan_neu = neumann_jvp()
    assert torch.allclose(xm_tan_anl, zm_tan_neu[0], atol=1e-4)
    
    grad_out = (grad_xm, torch.zeros_like(grad_xm))
    neumann_vjp = NeumannSeries(
        lambda v: pm_step.vjp_var(zm, u, v), grad_out, tol=1e-4, iters=1000,
        metric="default"
    )
    grad_u_neu = pm_step.vjp_par(zm, u, neumann_vjp())
    assert torch.allclose(grad_u_anl[0], grad_u_neu[0], atol=1e-4)
    assert torch.allclose(grad_u_anl[1], grad_u_neu[1], atol=1e-4)
    assert torch.allclose(grad_u_anl[2], grad_u_neu[2], atol=1e-4)


def test_implicit_differentiation_with_GD_step():
    # TODO: Figure out why the test is failing for JVP.
    
    def grad_map(x, u):
        A, b, r = u
        return A.T @ (A @ x - b) + r * x
    
    def minimizer(A, b, r):
        return la.solve(A.T @ A + r * torch.eye(A.shape[1]), A.T @ b)
    
    M, N = 5, 2
    A, b = torch.rand(M, N), torch.randn(M)
    r = torch.rand([])
    Q = A.T @ A +  r * torch.eye(N)
    A_tan, b_tan = torch.randn_like(A), torch.randn_like(b)
    r_tan = torch.randn_like(r)
    grad_xm = torch.randn(N)
    u = (A, b, r)
    u_tan = (A_tan, b_tan, r_tan)
    xm = minimizer(A, b, r)
    
    grad_u_anl = torch.func.vjp(minimizer, *u)[1](grad_xm)
    xm_tan_anl = torch.func.jvp(minimizer, u, u_tan)[1]
    
    lip, mu = la.norm(Q, ord=2), la.norm(Q, ord=-2)
    ss = 2 / (lip + mu)
    pm_step = GDParamMarkovStep(ss, grad_map_prm=grad_map, u_in=u)
    
    neumann_jvp = NeumannSeries(
        lambda v: pm_step.jvp_var(xm, u, v),
        pm_step.jvp_par(xm, u, u_tan),
        tol=1e-4, iters=10000, metric="default"
    )
    xm_tan_neu = neumann_jvp()
    assert torch.allclose(xm_tan_anl, xm_tan_neu, atol=1e-4)
    
    neumann_vjp = NeumannSeries(
        lambda v: pm_step.vjp_var(xm, u, v), grad_xm, tol=1e-4, iters=10000,
        metric="default"
    )
    grad_u_neu = pm_step.vjp_par(xm, u, neumann_vjp())
    assert torch.allclose(grad_u_anl[0], grad_u_neu[0], atol=1e-4)
    assert torch.allclose(grad_u_anl[1], grad_u_neu[1], atol=1e-4)
    assert torch.allclose(grad_u_anl[2], grad_u_neu[2], atol=1e-4)
