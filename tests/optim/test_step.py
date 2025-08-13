import torch
import pytest

from paramancer.optim.step import GDStep, PolyakStep
from paramancer.optim.step import ProxStep, ProxGradStep, FISTAStep

from tests.util import prox_l1, lin_reg, grad_lin_reg


def test_gd_step_squrared_euclidean():
    num_examples = 50
    stepsize = torch.linspace(0.1, 0.5, num_examples)
    x_curr = torch.linspace(1, 10, num_examples)
    gd_step = GDStep(stepsize=stepsize, grad_map=lambda x: x)
    
    x_step = gd_step(x_curr)
    
    x_direct = (1 - stepsize) * x_curr
    
    assert torch.allclose(x_step, x_direct, atol=1e-5)


def test_prox_step():
    reg = torch.rand(10)
    ss = torch.linspace(0.1, 0.5, 10)
    prox_map = prox_l1(reg*ss)
    
    x_curr = torch.randn(10)
    prox_step = ProxStep(prox_map)
    
    x_step = prox_step(x_curr)
    x_direct = prox_map(x_curr)
    
    assert torch.allclose(x_step, x_direct, atol=1e-5)


def test_polyak_step():
    M, N, K = 1000, 10, 5
    A, b, _, lip, _ = lin_reg(M, N)
    grad_map = grad_lin_reg(A, b)
    
    ss, mm = 1 / lip, 0.5
    x_curr = torch.randn(N)
    
    polyak_step = PolyakStep(ss, mm, grad_map)
    x_step = x_curr.clone()
    
    for _ in range(K): x_step = polyak_step(x_step)
    
    x_direct = x_curr.clone()
    x_prev = x_curr.clone()
    for _ in range(K):
        x_next = x_direct - ss * grad_map(x_direct) + mm * (x_direct - x_prev)
        x_direct, x_prev = x_next, x_direct
    
    assert torch.allclose(x_step, x_direct, atol=1e-5)


def test_pgd_and_fista_step():
    M, N, K = 1000, 10, 5
    A, b, _, lip, _ = lin_reg(M, N)
    reg = torch.rand(1)
    grad_map = grad_lin_reg(A, b)
    ss = 1 / lip
    prox_map = prox_l1(reg*ss)
    
    x_curr = torch.randn(N)
    
    pgd_step = ProxGradStep(ss, grad_map, prox_map)
    x_pgd = x_curr.clone()
    for _ in range(K):
        x_pgd = pgd_step(x_pgd)
    
    x_pgd_dr = x_curr.clone()
    for _ in range(K):
        x_pgd_dr = prox_map(x_pgd_dr - ss * grad_map(x_pgd_dr))
    
    
    fista_step = FISTAStep(ss, grad_map, prox_map)
    x_fista = x_curr.clone()
    for _ in range(K):
        x_fista = fista_step(x_fista)
    
    x_fista_dr = x_curr.clone()
    x_prev = x_curr.clone()
    t_prev = t = torch.tensor(0.)
    for _ in range(K):
        t = (1 + (1 + 4 * t_prev ** 2).sqrt()) / 2
        mm = (t_prev - 1) / t
        t_prev = t
        x_next = x_fista_dr + mm * (x_fista_dr - x_prev)
        x_next = prox_map(x_next - ss * grad_map(x_next))
        x_fista_dr, x_prev = x_next, x_fista_dr
    
    assert torch.allclose(x_pgd, x_pgd_dr, atol=1e-5)
    assert torch.allclose(x_fista, x_fista_dr, atol=1e-5)


def test_gd_step_differentiation():
    A, b, _, lip, _ = lin_reg(100, 10)
    x_curr = torch.randn(10)
    y_grad = torch.randn(10)
    
    A_direct = A.detach().clone().requires_grad_()
    b_direct = b.detach().clone().requires_grad_()
    x_direct = x_curr.detach().clone().requires_grad_()
    y_direct = x_direct - A_direct.T @ (A_direct @ x_direct - b_direct) / lip
    y_direct.backward(y_grad)
    
    A_step = A.detach().clone().requires_grad_()
    b_step = b.detach().clone().requires_grad_()
    x_step = x_curr.detach().clone().requires_grad_()
    grad_map = grad_lin_reg(A_step, b_step)
    gd_step = GDStep(1 / lip, grad_map)
    y_step = gd_step(x_step)
    y_step.backward(y_grad)
    
    assert torch.allclose(y_step, y_direct, atol=1e-5)
    assert torch.allclose(x_step.grad, x_direct.grad, atol=1e-5)
    assert torch.allclose(A_step.grad, A_direct.grad, atol=1e-5)
    assert torch.allclose(b_step.grad, b_direct.grad, atol=1e-5)
    

