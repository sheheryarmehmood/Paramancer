import torch
import torch.autograd.functional as agF
import torch.linalg as la
import pytest
from paramancer.operators.norms import l2
from paramancer.optim import GradientDescent, HeavyBall, AcceleratedGradient
from paramancer.optim import ProximalGradient, FISTA


# Testing Gradient Descent and Heavy-ball
def test_gd_and_hb():
    M, N = 10, 5
    A, b = torch.rand(M, N), torch.randn(M)
    lip = la.matrix_norm(A.T @ A, ord=2)
    mu = la.matrix_norm(A.T @ A, ord=-2)
    xm = la.solve(A.T @ A, A.T @ b)
    
    ss_gd = 2 / (lip + mu)
    sql, sqm = lip.sqrt(), mu.sqrt()
    ss_hb = (2 / (sql + sqm)) ** 2
    mm = ((sql - sqm) / (sql + sqm)) ** 2
    
    def grad_map(x): return A.T @ (A @ x - b)
    
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
    A, b = torch.rand(M, N), torch.randn(M)
    lip = la.matrix_norm(A.T @ A, ord=2)
    xm_grad = torch.randn(N)
    
    A_grad, b_grad = agF.vjp(
        lambda A, b: la.solve(A.T @ A, A.T @ b), (A, b), xm_grad
    )[1]
    
    ss = 1 / lip
    
    A = A.detach().clone().requires_grad_()
    b = b.detach().clone().requires_grad_()
    def grad_map(x): return A.T @ (A @ x - b)
    optim_nag = AcceleratedGradient(ss, grad_map)
    
    x_sol = torch.randn(N)
    
    x_sol = optim_nag(x_sol, 10000)
    x_sol.backward(xm_grad)
    
    assert torch.allclose(A.grad, A_grad, rtol=1e-3, atol=1e-5)
    assert torch.allclose(b.grad, b_grad, rtol=1e-3, atol=1e-5)

def test_prox_methods():
    M, N = 20, 5
    A, b = torch.rand(M, N), torch.randn(M)
    lip = la.matrix_norm(A.T @ A, ord=2)
    reg = torch.rand(1)
    xm = la.solve(A.T @ A + reg * torch.eye(N), A.T @ b)
    
    ss = 1 / lip
    
    def grad_map(x): return A.T @ (A @ x - b)
    def prox_map(x): return x / (1 + reg * ss)
    
    optim_pgd = ProximalGradient(ss, grad_map, prox_map)
    optim_fista = FISTA(ss, grad_map, prox_map)
    
    x_init = torch.randn(N)
    xm_pgd = optim_pgd(x_init, iters=10000)
    xm_fista = optim_fista(x_init, iters=10000)
    
    assert torch.allclose(xm, xm_pgd, rtol=1e-3, atol=1e-6)
    assert torch.allclose(xm, xm_fista, rtol=1e-3, atol=1e-6)


def test_hb_with_default_metric():
    M, N = 10, 5
    A, b = torch.rand(M, N), torch.randn(M)
    xm = la.solve(A.T @ A, A.T @ b)
    lip = la.matrix_norm(A.T @ A, ord=2)
    mu = la.matrix_norm(A.T @ A, ord=-2)
    
    sql, sqm = lip.sqrt(), mu.sqrt()
    ss = (2 / (sql + sqm)) ** 2
    mm = ((sql - sqm) / (sql + sqm)) ** 2
    
    def grad_map(x): return A.T @ (A @ x - b)
    
    heavy_ball = HeavyBall(ss, mm, grad_map, tol=1e-6, metric="default")
    
    x_hb = heavy_ball(torch.randn(N), iters=10000)
    
    # For that many iterations, the algorithm should converge
    assert heavy_ball.result.converged
    assert torch.allclose(x_hb, xm, atol=1e-4)


def test_nag_with_gradient_metric():
    M, N = 10, 5
    A, b = torch.rand(M, N), torch.randn(M)
    xm = la.solve(A.T @ A, A.T @ b)
    lip = la.matrix_norm(A.T @ A, ord=2)
    ss = 1 / lip
    
    def grad_map(x): return A.T @ (A @ x - b)
    def metric(x): return l2(grad_map(x))
    
    nag_optim = AcceleratedGradient(ss, grad_map, tol=1e-6, metric=metric)
    
    x_nag = nag_optim(torch.randn(N), iters=10000)
    
    # For that many iterations, the algorithm should converge
    assert nag_optim.result.converged
    assert torch.allclose(x_nag, xm, atol=1e-4)
    
    