import torch
import pytest
from torch import nn

from paramancer.variable import ParamBundle
from paramancer.operators.prox import l2_norm, l1_norm


def test_parameter():
    def grad_lin_reg(x: torch.Tensor, prms: tuple[torch.Tensor, ...]):
        A, b = prms
        return A.T @ (A @ x - b)
    
    def prox_l1(x: torch.Tensor, prms: torch.Tensor):
        reg, = prms
        return l1_norm(x, reg)
    
    def prox_l2(x: torch.Tensor, prms: tuple[torch.Tensor, ...]):
        reg = prms
        return l2_norm(x, reg)
    
    def grad_lin_reg_scaled(x: torch.Tensor, prms: tuple[torch.Tensor, ...]):
        A, b, ss = prms
        return ss * A.T @ (A @ x - b)
    
    def prox_l2_scaled(x: torch.Tensor, prms: tuple[torch.Tensor, ...]):
        reg, ss = prms
        return l2_norm(x, reg * ss)
    
    M, N = 10, 5
    x = torch.randn(N)
    A, b = torch.rand(M, N), torch.randn(M)
    reg = torch.rand(1).squeeze()
    ss = torch.rand(1).squeeze()
    param_l1_reg = ParamBundle(
        nn.ParameterList((A.clone(), b.clone(), reg.clone())),
        {"grad": (0, 1), "prox": (2,)}
    )
    param_l2_reg = ParamBundle(
        nn.ParameterList((A.clone(), b.clone(), reg.clone())),
        {"grad": (0, 1), "prox": 2}
    )
    param_l2_reg_scaled = ParamBundle(
        nn.ParameterList((A.clone(), b.clone(), reg.clone(), ss.clone())),
        {"grad": (0, 1, 3), "prox": (2, 3)}
    )

    assert torch.allclose(
        grad_lin_reg(x, (A, b)), grad_lin_reg(x, param_l2_reg.grad)
    )

    assert torch.allclose(
        prox_l1(x, (reg,)), prox_l1(x, param_l1_reg.prox)
    )

    assert torch.allclose(
        prox_l2(x, reg), prox_l2(x, param_l2_reg.prox)
    )

    assert torch.allclose(
        grad_lin_reg_scaled(x, (A, b, ss)),
        grad_lin_reg_scaled(x, param_l2_reg_scaled.grad)
    )

    assert torch.allclose(
        prox_l2_scaled(x, (reg, ss)),
        prox_l2_scaled(x, param_l2_reg_scaled.prox)
    )
