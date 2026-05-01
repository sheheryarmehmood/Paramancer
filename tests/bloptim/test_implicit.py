import torch
import torch.autograd.functional as agF
import torch.linalg as la
import pytest

from paramancer.bloptim.implicit.step import (
    NesterovMarkovParamStep, PolyakParamMarkovStep
)
from paramancer.bloptim.implicit import VJP, OptimizerID
from paramancer.variable import AlgoParam, FlatVar, PairVar
from paramancer.variable import FlatVar, PairVar, ParamBundle


def test_VJP():
    def grad_map(x, u):
        return x - u

    stepsize = 0.5
    u = torch.tensor(3.0)
    u_in = AlgoParam(u)
    x_min = u.clone()
    x_grad = torch.tensor(2.0)
    expected_grad_u = x_grad.clone()

    step = NesterovMarkovParamStep(
        stepsize,
        grad_map_prm=grad_map,
        u_in=u_in,
        momentum_scheduler=lambda: torch.tensor(0.0),
    )
    z_root = PairVar(x_min, x_min)
    z_grad = PairVar(x_grad, torch.zeros_like(x_grad))
    grad_u = VJP(step, tol=1e-8, iters=20)(z_root, u_in, z_grad)

    assert torch.allclose(grad_u.data, expected_grad_u)


def test_implicit_differentiation_with_OptimizerID():
    def grad_map(x, u):
        A, b = u
        return A.T @ (A @ x - b) + reg * x

    def minimizer(A, b):
        return la.solve(A.T @ A + reg * torch.eye(A.shape[1]), A.T @ b)

    M, N = 5, 2
    reg = torch.tensor(0.5)
    A = torch.rand(M, N, requires_grad=True)
    b = torch.randn(M, requires_grad=True)
    grad_xm = torch.randn(N)
    
    x_init = FlatVar(torch.randn(N))
    u = ParamBundle(torch.nn.ParameterList([A.clone(), b.clone()]))

    Q = A.T @ A + reg * torch.eye(N)
    lip, mu = la.norm(Q, ord=2), la.norm(Q, ord=-2)
    rt_lip, rt_mu = lip.sqrt(), mu.sqrt()
    ss = 4 / (rt_lip + rt_mu) ** 2
    mm = (rt_lip - rt_mu) ** 2 / (rt_lip + rt_mu) ** 2
    pm_step = PolyakParamMarkovStep(ss, mm, grad_map_prm=grad_map, u_in=u)
    
    u_flat, u_spec = u.flatten()
    x_init_flat, x_spec = x_init.flatten()
    xm = OptimizerID.apply(
        pm_step,
        u_spec,
        x_spec,
        u.indices,
        1e-6,
        1000,
        "default",
        False,
        *u_flat,
        *x_init_flat,
    )
    assert torch.allclose(xm.detach(), minimizer(A, b), atol=1e-4)

    xm.backward(grad_xm)
    grad_u_anl = torch.func.vjp(minimizer, A, b)[1](grad_xm)
    assert torch.allclose(u.data[0].grad, grad_u_anl[0], atol=1e-4)
    assert torch.allclose(u.data[1].grad, grad_u_anl[1], atol=1e-4)
