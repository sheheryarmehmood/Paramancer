import torch
import torch.autograd.functional as agF
import torch.linalg as la
import pytest

from paramancer.bloptim.step import (
    NesterovMarkovParamStep, ProxGradParamMarkovStep
)
from paramancer.bloptim.implicit import VJP, JVP


def test_VJP():
    def grad_map(x, u):
        return x - u

    stepsize = 0.5
    u = torch.tensor(3.0)
    x_min = u.clone()
    x_grad = torch.tensor(2.0)
    expected_grad_u = x_grad.clone()

    step = NesterovMarkovParamStep(
        stepsize,
        grad_map_prm=grad_map,
        u_in=u,
        momentum_scheduler=lambda: torch.tensor(0.0),
    )
    z_root = (x_min, x_min)
    z_grad = (x_grad, torch.zeros_like(x_grad))
    grad_u = VJP(step, tol=1e-8, iters=20)(z_root, u, z_grad)

    assert torch.allclose(grad_u, expected_grad_u)
