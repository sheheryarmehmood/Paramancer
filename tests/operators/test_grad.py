import torch
import pytest
from paramancer.operators import gradient
from paramancer.operators.norms import l2_sq


def test_gradient():
    def fn(x, A, b):
        return 0.5 * l2_sq(A @ x - b)
    
    A = torch.rand(100, 10)
    b = torch.randn(100)
    x = torch.randn(10)
    
    grad_x = A.T @ (A @ x - b)
    grad_b = b - A @ x
    grad_A = torch.outer(A @ x - b, x)
    
    gradx_fn = gradient(fn, 0)
    gradAb_fn = gradient(fn, 1, 2)
    
    assert all([
        torch.allclose(grad_x, gradx_fn(x, A, b)),
        torch.allclose(grad_b, gradAb_fn(x, A, b)[1]),
        torch.allclose(grad_A, gradAb_fn(x, A, b)[0])
    ])


def test_gradient_vhp():
    def fn(x, A, b):
        return 0.5 * l2_sq(A @ x - b)
    
    A = torch.rand(100, 10)
    b = torch.randn(100, requires_grad=True)
    x = torch.randn(10, requires_grad=True)
    v = torch.randn(10)
    
    gradxA_fn = gradient(fn, 0)
    gx = gradxA_fn(x, A, b)
    
    (gx @ v).backward()
    
    assert all([
        A.grad is None,
        torch.allclose(x.grad, A.T @ A @ v),
        torch.allclose(b.grad, -A @ v)
    ])
    
    
    