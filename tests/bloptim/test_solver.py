import pytest
import torch

from paramancer.bloptim.solver import Solver
from paramancer.bloptim.util import Algorithm


def quadratic_grad(x, target):
    return x - target


@pytest.mark.parametrize("backend", ["unrolled", "implicit"])
@pytest.mark.parametrize(
    ("algorithm", "momentum"),
    [
        (Algorithm.GD, None),
        (Algorithm.HB, 0.2),
    ],
)
def test_solver_quadratic_forward_and_gradient(
    backend,
    algorithm,
    momentum,
):
    target = torch.tensor(
        [1.5, -0.5, 2.0],
        dtype=torch.float64,
        requires_grad=True,
    )
    grad_out = torch.tensor([0.3, -1.0, 0.7], dtype=target.dtype)
    solver = Solver(
        backend=backend,
        algorithm=algorithm,
        iters=60,
        tol=0.0,
    )

    solution = solver(
        init=torch.zeros_like(target),
        outer_vars=target,
        stepsize=0.5,
        momentum=momentum,
        grad_map=quadratic_grad,
    )

    assert torch.allclose(solution, target, atol=1e-10, rtol=1e-10)

    (solution * grad_out).sum().backward()

    assert torch.allclose(target.grad, grad_out, atol=1e-10, rtol=1e-10)

    if backend == "implicit":
        adjoint_state = solver.adjoint_state
        if algorithm == Algorithm.GD:
            assert isinstance(adjoint_state, torch.Tensor)
        else:
            assert isinstance(adjoint_state, tuple)
            assert len(adjoint_state) == 2
