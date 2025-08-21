import torch
import pytest
import torch.nn.functional as nnF

from paramancer.optim.step import ProxStep, GDStep, MomentumStep
from paramancer.optim.step import PolyakStep, NesterovStep
from paramancer.optim.step import ProxGradStep, FISTAStep


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
    def prox_map(x):
        return x.sign() * torch.maximum(torch.zeros(1), x.abs() - ss*reg)
    
    x_curr = torch.randn(10)
    prox_step = ProxStep(prox_map)
    
    x_step = prox_step(x_curr)
    x_direct = prox_map(x_curr)
    
    assert torch.allclose(x_step, x_direct, atol=1e-5)


def test_polyak_step():
    M, N = 10, 6
    A, b = torch.rand(M, N), torch.randn(M)
    def grad_map(x): return A.T @ (A @ x - b)
    
    ss, mm = 0.001, 0.5
    x_curr = torch.randn(N)
    
    
    # 3 steps using an object of `PolyakStep`.
    polyak_step = PolyakStep(ss, mm, grad_map)
    x_step = polyak_step(polyak_step(polyak_step(x_curr.clone())))
    
    
    # Custom Implementation of Polyak's Heavy Ball
    x_direct = x_curr.clone()
    x_prev = x_curr.clone()
    
    # Iter 0
    x_next = x_direct - ss * grad_map(x_direct) + mm * (x_direct - x_prev)
    x_direct, x_prev = x_next, x_direct
    
    # Iter 1
    x_next = x_direct - ss * grad_map(x_direct) + mm * (x_direct - x_prev)
    x_direct, x_prev = x_next, x_direct
    
    # Iter 2
    x_next = x_direct - ss * grad_map(x_direct) + mm * (x_direct - x_prev)
    x_direct, x_prev = x_next, x_direct
    
    assert torch.allclose(x_step, x_direct, atol=1e-5)


def test_pgd_and_fista_step():
    M, N = 10, 5
    A, b = torch.rand(M, N), torch.randn(M)
    reg = torch.tensor(0.02)
    
    def grad_map(x): return A.T @ (A @ x - b)
    def prox_map(x): return nnF.softshrink(x, lambd=float(ss*reg))
    
    ss = 1 / torch.linalg.norm(A.T @ A, ord=2)
    x_curr = torch.randn(N)
    
    
    # 3 Iterations of `ProxGradStep`
    pgd_step = ProxGradStep(ss, grad_map, prox_map)
    x_pgd = pgd_step(pgd_step(pgd_step(x_curr.clone())))
    
    
    # Custom Implemented Proximal Gradient Steps
    x_pgd_dr = x_curr.clone()
    
    # Iter 0
    x_pgd_dr = prox_map(x_pgd_dr - ss * grad_map(x_pgd_dr))
    
    # Iter 1
    x_pgd_dr = prox_map(x_pgd_dr - ss * grad_map(x_pgd_dr))
    
    # Iter 2
    x_pgd_dr = prox_map(x_pgd_dr - ss * grad_map(x_pgd_dr))
    
    
    # 3 Iterations of `FISTAStep`
    fista_step = FISTAStep(ss, grad_map, prox_map)
    x_fista = fista_step(fista_step(fista_step(x_curr.clone())))
    
    # Custom Implemented FISTA Steps
    
    # Iter 0
    t1 = torch.tensor(1.)
    y1 = x0 = x_curr.clone()
    x1 = prox_map(y1 - ss * grad_map(y1))
    
    # Iter 1
    t2 = (1 + torch.sqrt(1 + 4 * t1 ** 2)) / 2
    y2 = x1 + (t1 - 1) * (x1 - x0) / t2
    x2 = prox_map(y2 - ss * grad_map(y2))
    
    # Iter 2
    t3 = (1 + torch.sqrt(1 + 4 * t2 ** 2)) / 2
    y3 = x2 + (t2 - 1) * (x2 - x1) / t3
    x3 = prox_map(y3 - ss * grad_map(y3))
    
    x_fista_dr = x3
    
    assert torch.allclose(x_pgd, x_pgd_dr, atol=1e-5)
    assert torch.allclose(x_fista, x_fista_dr, atol=1e-5)


def test_gd_step_differentiation():
    M, N = 10, 5
    A, b = torch.rand(M, N), torch.randn(M)
    ss = 0.1
    x_curr = torch.randn(N)
    y_grad = torch.randn(N)
    
    A_direct = A.detach().clone().requires_grad_()
    b_direct = b.detach().clone().requires_grad_()
    x_direct = x_curr.detach().clone().requires_grad_()
    y_direct = x_direct - ss * A_direct.T @ (A_direct @ x_direct - b_direct)
    y_direct.backward(y_grad)
    
    A_step = A.detach().clone().requires_grad_()
    b_step = b.detach().clone().requires_grad_()
    x_step = x_curr.detach().clone().requires_grad_()
    
    def grad_map(x):
        return A_step.T @ (A_step @ x - b_step)
    gd_step = GDStep(ss, grad_map)
    y_step = gd_step(x_step)
    y_step.backward(y_grad)
    
    assert torch.allclose(y_step, y_direct, atol=1e-5)
    assert torch.allclose(x_step.grad, x_direct.grad, atol=1e-5)
    assert torch.allclose(A_step.grad, A_direct.grad, atol=1e-5)
    assert torch.allclose(b_step.grad, b_direct.grad, atol=1e-5)
    

def test_fista_residual_value():
    def grad_fn(x): return -1/x     # fn(x) = -ln(x)
    def prox_fn(x): return nnF.softshrink(x, lambd=ss*reg)
    
    ss = 0.01
    reg = 0.5
    
    # ----- manual FISTA computation -----
    t1 = torch.tensor(1.)
    y1 = x0 = torch.tensor(10.)
    x1 = prox_fn(y1 - ss * grad_fn(y1))
    res1 = x1 - y1
    
    t2 = (1 + torch.sqrt(1 + 4 * t1 ** 2)) / 2
    y2 = x1 + (t1 - 1) * (x1 - x0) / t2
    x2 = prox_fn(y2 - ss * grad_fn(y2))
    res2 = x2 - y2
    
    t3 = (1 + torch.sqrt(1 + 4 * t2 ** 2)) / 2
    y3 = x2 + (t2 - 1) * (x2 - x1) / t3
    x3 = prox_fn(y3 - ss * grad_fn(y3))
    res3 = x3 - y3
    
    fista_step = FISTAStep(ss, grad_fn, prox_fn, tracking=True)
    x_curr = x0
    
    # Iter 1
    x_curr = fista_step(x_curr)
    assert torch.allclose(fista_step.residual, res1, atol=1e-5)
    
    # Iter 2
    x_curr = fista_step(x_curr)
    assert torch.allclose(fista_step.residual, res2, atol=1e-5)
    
    # Iter 3
    x_curr = fista_step(x_curr)
    assert torch.allclose(fista_step.residual, res3, atol=1e-5)

def test_fista_residual_attribute_duplication():
    def grad_fn(x): return -1/x     # fn(x) = -ln(x)
    def prox_fn(x): return nnF.softshrink(x, lambd=ss*reg)
    
    ss = 0.01
    reg = 0.5
    
    fista_step = FISTAStep(ss, grad_fn, prox_fn, tracking=True)
    _ = fista_step(fista_step(fista_step(torch.tensor(10.))))
    
    # FISTAStep residual must come from its ProxGradStep.
    assert torch.allclose(
        fista_step.residual, fista_step.pgd_step._residual, atol=1e-5
    )
    
    def check_if_has_attr_prop(
        step, must_raise_Exp_if_not_have, raise_Exp_if_has, message, attr=True
    ):
        try:
            _ = (step._residual if attr else step.residual)
        except must_raise_Exp_if_not_have:
            pass
        else:
            raise raise_Exp_if_has(message)
    
    # FISTAStep must not have its own attribute named `_residual`.
    check_if_has_attr_prop(
        fista_step, AttributeError, AssertionError,
        "fista_step._residual exists!", attr=True
    )
    
    # Momentum SubStep must not have its own attribute named `_residual` and
    # a property named `residual`.
    check_if_has_attr_prop(
        fista_step.mm_step, AttributeError, AssertionError,
        "fista_step.mm_step._residual exists!", attr=True
    )
    check_if_has_attr_prop(
        fista_step.mm_step, RuntimeError, AssertionError,
        "fista_step.mm_step.residual exists!", attr=False
    )
    
    # GD SubStep must not have its own attribute named `_residual` and
    # a property named `residual`.
    check_if_has_attr_prop(
        fista_step.pgd_step.gd_step, AttributeError, AssertionError,
        "fista_step.pgd_step.gd_step._residual exists!", attr=True
    )
    check_if_has_attr_prop(
        fista_step.pgd_step.gd_step, RuntimeError, AssertionError,
        "fista_step.pgd_step.gd_step.residual exists!", attr=False
    )
    
    # Prox SubStep must not have its own attribute named `_residual` and
    # a property named `residual`.
    check_if_has_attr_prop(
        fista_step.pgd_step.prox_step, AttributeError, AssertionError,
        "fista_step.pgd_step.prox_step._residual exists!", attr=True
    )
    check_if_has_attr_prop(
        fista_step.pgd_step.prox_step, RuntimeError, AssertionError,
        "fista_step.pgd_step.prox_step.residual exists!", attr=False
    )
    
def test_markovian_property():
    def grad_map(x): x
    def prox_map(x): x
    
    ss = torch.tensor(0.1)
    mm = torch.tensor(0.1)
    
    mm_step = MomentumStep(mm)
    gd_step = GDStep(ss, grad_map)
    prox_step = ProxStep(prox_map)
    hb_step = PolyakStep(ss, mm, grad_map)
    nag_step = NesterovStep(ss, grad_map)
    pgd_step = ProxGradStep(ss, grad_map, prox_map)
    fista_step = FISTAStep(ss, grad_map, prox_map)
    
    assert not mm_step.is_markovian()
    assert gd_step.is_markovian()
    assert prox_step.is_markovian()
    assert not hb_step.is_markovian()
    assert not nag_step.is_markovian()
    assert pgd_step.is_markovian()
    assert not fista_step.is_markovian()