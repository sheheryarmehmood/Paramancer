import torch
from paramancer.operators import adjoint
import pytest


def test_adjoint_single_input():
    A = torch.tensor([[1., 2.], [3., 4.]])
    def lin_op(x): return A @ x
    zero_el = torch.zeros(2)
    lin_op_adj = adjoint(lin_op, zero_el)
    y = torch.tensor([1., 1.])
    assert torch.allclose(lin_op_adj(y), A.T @ y, atol=1e-5)

def test_adjoint_multiple_input():
    def lin_op(*args): return sum(args)
    num_inputs = 5
    input_size = 15
    zero_el = tuple(torch.zeros(input_size) for _ in range(num_inputs))
    lin_op_adj = adjoint(lin_op, zero_el)
    z = torch.randn(input_size)
    adj_out = tuple(z.clone() for _ in range(num_inputs))
    assert all(
        torch.allclose(x, y, atol=1e-5) for x, y in zip(adj_out, lin_op_adj(z))
    )

def test_adjoint_differentiablity():
    rows, cols = 15, 10
    A = torch.rand(rows, cols)
    def lin_op(x): return A @ x
    zero_el = torch.zeros(cols)
    lin_op_adj = adjoint(lin_op, zero_el)
    
    A.requires_grad = True
    y = torch.randn(rows, requires_grad=True)
    lin_op_adj(y).sum().backward()
    
    ones = torch.ones(cols)
    
    assert torch.allclose(y.grad, A @ ones, atol=1e-5)
    assert torch.allclose(A.grad, torch.outer(y, ones), atol=1e-5)

def test_parametric_adjoint():
    rows, cols, cols1, cols2 = 15, 20, 10, 5
    A = torch.rand(rows, cols)
    A1 = torch.rand(rows, cols1)
    A2 = torch.rand(rows, cols2)
    def lin_op_single(x, A): return A @ x
    def lin_op_multi(x1, x2, A1, A2): return A1 @ x1 + A2 @ x2
    zero_el_single = torch.zeros(cols)
    zero_el_multi = torch.zeros(cols1), torch.zeros(cols2)
    lin_op_adj_single = adjoint(lin_op_single, zero_el_single)
    lin_op_adj_multi = adjoint(lin_op_multi, zero_el_multi)
    y = torch.randn(rows)
    
    x = lin_op_adj_single(y, A)
    x1, x2 = lin_op_adj_multi(y, A1, A2)
    
    assert torch.allclose(x, A.T @ y, atol=1e-5)
    assert torch.allclose(x1, A1.T @ y, atol=1e-5)
    assert torch.allclose(x2, A2.T @ y, atol=1e-5)
    