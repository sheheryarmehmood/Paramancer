import torch
import torch.autograd.functional as agF
import torch.linalg as la
import pytest

from paramancer.bloptim.implicit.step import (
    NesterovMarkovParamStep, ProxGradParamMarkovStep
)
from paramancer.bloptim.implicit import VJP, JVP
from paramancer.variable import AlgoParam, FlatVar, PairVar


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
