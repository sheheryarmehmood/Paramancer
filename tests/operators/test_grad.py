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
    
    grad_x_tr = A.T @ (A @ x - b)
    grad_b_tr = b - A @ x
    grad_A_tr = torch.outer(A @ x - b, x)
    
    gradx_fn = gradient(fn, 0)
    gradAb_fn = gradient(fn, 1, 2)
    
    assert torch.allclose(grad_x_tr, gradx_fn(x, A, b), atol=1e-5)
    assert torch.allclose(grad_b_tr, gradAb_fn(x, A, b)[1], atol=1e-5)
    assert torch.allclose(grad_A_tr, gradAb_fn(x, A, b)[0], atol=1e-5)


def test_gradient_vhp():
    def fn(x, A, b):
        return 0.5 * l2_sq(A @ x - b)
    
    A = torch.rand(20, 10)
    b = torch.randn(20, requires_grad=True)
    x = torch.randn(10, requires_grad=True)
    v = torch.randn(10)
    
    gradxA_fn = gradient(fn, 0)
    gx = gradxA_fn(x, A, b)
    
    (gx @ v).backward()
    
    v_d2x = A.T @ A @ v
    v_dbx = -A @ v
    
    assert A.grad is None
    assert torch.allclose(x.grad, v_d2x, atol=1e-5)
    assert torch.allclose(b.grad, v_dbx, atol=1e-5)
    
    
    