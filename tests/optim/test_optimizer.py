import torch
import torch.linalg as la
import pytest
from paramancer.optim import GradientDescent, HeavyBall, AcceleratedGradient
from paramancer.optim import ProximalGradient, FISTA
from tests.util import grad_lin_reg, lin_reg, lin_reg_grad_sol, prox_sq_l2


# Testing Gradient Descent and Heavy-ball
def test_gd_and_hb():
    M, N = 1000, 10
    A, b, xm, lip, mu = lin_reg(M, N)
    ss_gd = 2 / (lip + mu)
    sql, sqm = lip.sqrt(), mu.sqrt()
    ss_hb = (2 / (sql + sqm)) ** 2
    mm = ((sql - sqm) / (sql + sqm)) ** 2
    
    grad_map = grad_lin_reg(A, b)
    
    optim_gd = GradientDescent(ss_gd, grad_map)
    optim_hb = HeavyBall(ss_hb, mm, grad_map)
    
    x_init = torch.randn(N)
    xm_gd = optim_gd(x_init, iters=10000)
    xm_hb = optim_hb(x_init, iters=10000)
    
    assert torch.allclose(xm, xm_gd, rtol=1e-3, atol=1e-6)
    assert torch.allclose(xm, xm_hb, rtol=1e-4, atol=1e-6)


# Unrolling of Accelerated Gradient
def test_nag_unrolling():
    M, N = 5, 3
    A, b, xm, lip, _ = lin_reg(M, N)
    xm_grad = torch.randn(xm.shape)
    A_grad, b_grad = lin_reg_grad_sol(A, b, xm_grad)
    
    ss = 1 / lip
    
    A = A.detach().clone().requires_grad_()
    b = b.detach().clone().requires_grad_()
    grad_map = grad_lin_reg(A, b)
    optim_nag = AcceleratedGradient(ss, grad_map)
    
    x_sol = torch.randn(N)
    
    x_sol = optim_nag(x_sol, 10000)
    x_sol.backward(xm_grad)
    
    assert torch.allclose(A.grad, A_grad, rtol=1e-3, atol=1e-5)
    assert torch.allclose(b.grad, b_grad, rtol=1e-3, atol=1e-5)

def test_prox_methods():
    M, N = 20, 5
    A, b, _, lip, _ = lin_reg(M, N)
    reg = torch.rand(1)
    xm = la.solve(A.T @ A + reg * torch.eye(N), A.T @ b)
    
    ss = 1 / lip
    
    grad_map = grad_lin_reg(A, b)
    prox_map = prox_sq_l2(reg * ss)
    
    optim_pgd = ProximalGradient(ss, grad_map, prox_map)
    optim_fista = FISTA(ss, grad_map, prox_map)
    
    x_init = torch.randn(N)
    xm_pgd = optim_pgd(x_init, iters=10000)
    xm_fista = optim_fista(x_init, iters=10000)
    
    assert torch.allclose(xm, xm_pgd, rtol=1e-3, atol=1e-6)
    assert torch.allclose(xm, xm_fista, rtol=1e-3, atol=1e-6)
    