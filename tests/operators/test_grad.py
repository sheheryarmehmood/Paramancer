from __future__ import annotations
import torch
import pytest

from paramancer.operators.grad import _gradient, gradient
from paramancer.operators.norms import l2_sq
from paramancer.variable import Variable
from paramancer.operators.norms import inner_product, l2_sq
from paramancer.variable.types import (
    BaseVariableLike, ParameterType, ScalarLike
)


def test_gradient():
    def smooth(x, u, a1, a2): # a1 and a2 can be non-tensors
        x_data = x.data if isinstance(x, Variable) else x
        x1, x2 = x_data
        u1, u2 = u
        return 0.5 * a1 * u2.sum() * l2_sq(x1) + a2 * inner_product(x2, u1)
    # If `smooth` can handle different kinds of inputs, then so can `grad_fn`.
    # We need to construct `grad_fn` only once.
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
    x1.requires_grad = True     # x_var should get updated automatically.
    gd_var = grad_fn(x_var, u_par, a1, a2)
    sum(tuple(g.sum() for g in gd_var.data)).backward()
    
    assert torch.allclose(
        x_var.data[0].grad, a1 * u2.sum() * torch.ones_like(x1)
    )
    assert torch.allclose(u_par[0].grad, a2 * torch.ones_like(u1))
    assert torch.allclose(u_par[1].grad, a1 * x1.sum() * torch.ones_like(u2))
    assert x1.requires_grad
    assert not x2.requires_grad


def test_gradient_differentiability_with_args_and_kwargs():
    def smooth_bvar(
        x: BaseVariableLike, a: torch.Tensor, b: ScalarLike, c: ScalarLike = 4
    ) -> torch.Tensor:
        x1, x2, x3 = x
        return b * x1.sum() + 0.5 * c * (x2 ** 2).sum() + inner_product(a, x3)
    
    def smooth_var(
        x: Variable, u: ParameterType, c: ScalarLike = 4
    ) -> torch.Tensor:
        x1, x2, x3 = x.data
        a, b = tuple(u)
        return b * x1.sum() + 0.5 * c * (x2 ** 2).sum() + inner_product(a, x3)
    
    def grad(
        x: BaseVariableLike, a: torch.Tensor, b: ScalarLike, c: ScalarLike = 4
    ):
        x1, x2, _ = x
        return b * torch.ones_like(x1), c * x2, a
    
    def vHp(
        x: BaseVariableLike, a: torch.Tensor, b: ScalarLike, c: ScalarLike,
        gd1_grad, gd2_grad, gd3_grad
    ):
        x1, _, x3 = x
        zx1, zx3 = torch.zeros_like(x1), torch.zeros_like(x3)
        return (zx1, c * gd2_grad, zx3)
    
    def vJgp(
        x: BaseVariableLike, a: torch.Tensor, b: ScalarLike, c: ScalarLike,
        gd1_grad, gd2_grad, gd3_grad
    ):
        return gd3_grad, gd1_grad.sum()
    
    x1 = torch.rand(4, 3)
    x2 = torch.randn(10, 5)
    x3 = torch.randn(10)
    a = torch.randn(10)
    b = torch.rand(1).squeeze()
    c = 15.
    gd = grad((x1, x2, x3), a, b, c=c)
    gd_grad = tuple(
        torch.randn(gx.shape) for gx in gd
    )
    
    grad_bvar = gradient(smooth_bvar)
    grad_var = gradient(smooth_var)
    vhp = vHp((x1, x2, x3), a, b, c, *gd_grad)
    vJgp_a, vJgp_b = vJgp((x1, x2, x3), a, b, c, *gd_grad)
    
    a_bvar = torch.nn.Parameter(a)
    u_var = torch.nn.ParameterList((a, b))
    x_bvar = tuple(xk.clone().requires_grad_(True) for xk in (x1, x2, x3))
    x_var = Variable(
        tuple(xk.clone().requires_grad_(True) for xk in (x1, x2, x3))
    )
    
    # Within no_grad, 
    with torch.no_grad():
        gd_bvar = grad_bvar(x_bvar, a_bvar, b, c=c)
        out_bvar = sum(
            inner_product(gdk, gdk_grad) 
            for gdk, gdk_grad in zip(gd_bvar, gd_grad)
        )
        gd_var = grad_var(x_var, u_var, c=c).data
        out_var = sum(
            inner_product(gdk, gdk_grad) 
            for gdk, gdk_grad in zip(gd_var, gd_grad)
        )
    assert all(torch.allclose(gdk_b, gdk) for gdk_b, gdk in zip(gd_bvar, gd))
    assert all(torch.allclose(gdk_v, gdk) for gdk_v, gdk in zip(gd_var, gd))
    assert out_bvar.requires_grad is False
    assert out_var.requires_grad is False
    
    
    # Outside no_grad
    gd_bvar = grad_bvar(x_bvar, a_bvar, b, c=c)
    out_bvar = sum(
        inner_product(gdk, gdk_grad) 
        for gdk, gdk_grad in zip(gd_bvar, gd_grad)
    )
    gd_var = grad_var(x_var, u_var, c=c).data
    out_var = sum(
        inner_product(gdk, gdk_grad) 
        for gdk, gdk_grad in zip(gd_var, gd_grad)
    )
    out_bvar.backward()
    out_var.backward()
    
    assert x_var.data[0].grad is None and torch.norm(vhp[0]) < 1e-6
    assert torch.allclose(x_var.data[1].grad, vhp[1])
    assert x_var.data[2].grad is None and torch.norm(vhp[2]) < 1e-6
    
    assert x_bvar[0].grad is None and torch.norm(vhp[0]) < 1e-6
    assert torch.allclose(x_bvar[1].grad, vhp[1])
    assert x_bvar[2].grad is None and torch.norm(vhp[2]) < 1e-6
    
    assert torch.allclose(a_bvar.grad, vJgp_a)
    assert torch.allclose(u_var[0].grad, vJgp_a)
    assert torch.allclose(u_var[1].grad, vJgp_b)
    


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
    
    
    
