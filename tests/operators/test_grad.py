import torch
import pytest

from paramancer.operators.grad import _gradient, gradient
from paramancer.operators.norms import l2_sq
from paramancer.optim.variable import Variable
from paramancer.operators.norms import inner_product, l2_sq


def test_gradient():
    def smooth(x, u):
        x_data = x.data if isinstance(x, Variable) else x
        x1, x2 = x_data
        return 0.5 * l2_sq(x1) + inner_product(x2, u)
    
    x1 = torch.randn(5)
    x2 = torch.randn(7)
    u = torch.randn(7)
    
    grad_fn = gradient(smooth)
    gd = grad_fn((x1, x2), u)
    
    assert torch.allclose(gd[0], x1)
    assert torch.allclose(gd[1], u)
    
    x_var = Variable((x1, x2))
    gd_var = grad_fn(x_var, u)
    
    assert isinstance(gd_var, Variable)
    assert torch.allclose(gd_var.data[0], x1)
    assert torch.allclose(gd_var.data[1], u)


def test_backend_gradient():
    def fn(x, A, b):
        return 0.5 * l2_sq(A @ x - b)
    
    A = torch.rand(100, 10)
    b = torch.randn(100)
    x = torch.randn(10)
    
    grad_x_tr = A.T @ (A @ x - b)
    grad_b_tr = b - A @ x
    grad_A_tr = torch.outer(A @ x - b, x)
    
    gradx_fn = _gradient(fn, 0)
    gradAb_fn = _gradient(fn, 1, 2)
    
    assert torch.allclose(grad_x_tr, gradx_fn(x, A, b), atol=1e-5)
    assert torch.allclose(grad_b_tr, gradAb_fn(x, A, b)[1], atol=1e-5)
    assert torch.allclose(grad_A_tr, gradAb_fn(x, A, b)[0], atol=1e-5)


def test_backend_gradient_vhp():
    def fn(x, A, b):
        return 0.5 * l2_sq(A @ x - b)
    
    A = torch.rand(20, 10)
    b = torch.randn(20, requires_grad=True)
    x = torch.randn(10, requires_grad=True)
    v = torch.randn(10)
    
    gradxA_fn = _gradient(fn, 0)
    gx = gradxA_fn(x, A, b)
    
    (gx @ v).backward()
    
    v_d2x = A.T @ A @ v
    v_dbx = -A @ v
    
    assert A.grad is None
    assert torch.allclose(x.grad, v_d2x, atol=1e-5)
    assert torch.allclose(b.grad, v_dbx, atol=1e-5)
    
    
    
