import pytest
import torch

from paramancer.bloptim.solver import Solver
from paramancer.bloptim.util import Algorithm, optimal_algorithm_parameters


def quadratic_grad(x, target):
    return x - target


def quadratic_objective(x, target):
    return 0.5 * (x - target).square().sum()


def test_optimal_algorithm_parameters():
    lipschitz = torch.tensor(9.0)
    strong_convexity = torch.tensor(1.0)

    gd_step, gd_momentum = optimal_algorithm_parameters(
        Algorithm.GD,
        lipschitz,
        strong_convexity,
    )
    hb_step, hb_momentum = optimal_algorithm_parameters(
        Algorithm.HB,
        lipschitz,
        strong_convexity,
    )
    nag_step, nag_momentum = optimal_algorithm_parameters(
        Algorithm.NAG,
        lipschitz,
        strong_convexity,
    )

    assert torch.allclose(gd_step, torch.tensor(0.2))
    assert gd_momentum is None
    assert torch.allclose(hb_step, torch.tensor(0.25))
    assert torch.allclose(hb_momentum, torch.tensor(0.25))
    assert torch.allclose(nag_step, torch.tensor(1 / 9))
    assert nag_momentum is None


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


def test_implicit_solver_with_smooth_objective():
    target = torch.tensor([1.5, -0.5, 2.0], dtype=torch.float64, requires_grad=True)
    grad_out = torch.tensor([0.3, -1.0, 0.7], dtype=target.dtype)
    solver = Solver(
        backend="implicit",
        algorithm=Algorithm.GD,
        iters=60,
        tol=0.0,
    )

    solution = solver(
        init=torch.zeros_like(target),
        outer_vars=target,
        stepsize=0.5,
        smooth_obj=quadratic_objective,
    )

    assert torch.allclose(solution, target, atol=1e-10, rtol=1e-10)

    (solution * grad_out).sum().backward()

    assert torch.allclose(target.grad, grad_out, atol=1e-10, rtol=1e-10)


def test_unrolled_solver_exposes_idle_solution():
    target = torch.tensor([1.0, -2.0], dtype=torch.float64)
    initial = torch.zeros_like(target)
    solver = Solver(
        backend="unrolled",
        algorithm=Algorithm.GD,
        truncation=2,
        omega=2,
        iters=5,
        tol=0.0,
    )

    solver(
        init=initial,
        outer_vars=target,
        stepsize=0.5,
        grad_map=quadratic_grad,
    )

    expected = target * (1 - 0.5**6)
    assert torch.allclose(solver.idle_solution, expected)
