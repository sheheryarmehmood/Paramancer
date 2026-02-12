import torch
import pytest

from paramancer.operators.grad import _gradient, gradient
from paramancer.operators.norms import l2_sq
from paramancer.variable import Variable
from paramancer.operators.norms import inner_product, l2_sq


def test_gradient():
    def smooth(x, u, a1, a2): # a1 and a2 can be non-tensors
        x_data = x.data if isinstance(x, Variable) else x
        x1, x2 = x_data
        u1, u2 = u
        return 0.5 * a1 * u2.sum() * l2_sq(x1) + a2 * inner_product(x2, u1)
    grad_fn = gradient(smooth)
    
    x1, x2 = torch.randn(5), torch.randn(7)
    u1, u2 = torch.randn(7), torch.randn(10)
    a1, a2 = 4, 10
    
    x = (x1, x2)
    u = (u1, u2)
    
    gd = grad_fn(x, u, a1, a2)
    
    assert torch.allclose(gd[0], a1 * u2.sum() * x1)
    assert torch.allclose(gd[1], a2 * u1)
    
    x_var = Variable(x)
    gd_var = grad_fn(x_var, u, a1, a2)
    
    assert isinstance(gd_var, Variable)
    assert torch.allclose(gd_var.data[0], a1 * u2.sum() * x1)
    assert torch.allclose(gd_var.data[1], a2 * u1)
    
    u_par = torch.nn.ParameterList(u)
    x1.requires_grad = True     # x_var should get updated automatically?
    gd_var = grad_fn(x_var, u_par, a1, a2)
    sum(tuple(g.sum() for g in gd_var.data)).backward()
    
    assert torch.allclose(
        x_var.data[0].grad, a1 * u2.sum() * torch.ones_like(x1)
    )
    assert torch.allclose(u_par[0].grad, a2 * torch.ones_like(u1))
    assert torch.allclose(u_par[1].grad, a1 * x1.sum() * torch.ones_like(u2))


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
    
    
    
