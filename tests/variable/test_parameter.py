import torch
import pytest

from paramancer.variable import ParameterBundle
from paramancer.operators.prox import l2_norm, l1_norm


def test_parameter():
    def grad_lin_reg(x: torch.Tensor, prms: tuple[torch.Tensor, ...]):
        A, b = prms
        return A.T @ (A @ x - b)
    
    def prox_l2(x: torch.Tensor, prms: tuple[torch.Tensor, ...]):
        reg = prms
        return l2_norm(x, reg)
    
    def prox_l1(x: torch.Tensor, prms: torch.Tensor):
        reg, = prms
        return l1_norm(x, reg)
    
    M, N = 10, 5
    A, b = torch.rand(M, N), torch.randn(M)
    reg = torch.rand(1).squeeze()
    param_group = ParameterBundle()
    param_lasso = ...


