from __future__ import annotations

import math
from enum import Enum
from numbers import Real
from typing import TypeVar

import torch

from ..operators.grad import gradient
from ..operators.linalg import adjoint
from ..optim import optimizer
from ..optim.scheduler import MomentumScheduler
from ..variable.types import (
    FlatRawVarType,
    ParamGradMapType,
    ParamLinOpType,
    ParamProxMapType,
    ParamSmoothObjType,
    PGradMapType,
    ProxMapType,
    RawFlatLinOpType,
    RawParamType,
    ScalarLike,
)

T = TypeVar("T")


class Algorithm(Enum):
    GD = "gd"
    HB = "hb"
    NAG = "nag"
    PGD = "pgd"
    FISTA = "fista"
    PDHG = "pdhg"


def _required(value: T | None, name: str, algorithm: Algorithm) -> T:
    if value is None:
        raise ValueError(
            f"{algorithm.value.upper()} requires prepared `{name}`."
        )
    return value


def create_optimizer(
    algorithm: Algorithm,
    stepsize: ScalarLike,
    momentum: ScalarLike | None,
    grad_map: PGradMapType | None,
    prox_map: ProxMapType | None,
    stepsize_dual: ScalarLike | None,
    prox_map_dual: ProxMapType | None,
    lin_op: RawFlatLinOpType | None,
    lin_op_adj: RawFlatLinOpType | None,
    iters: int,
    tol: float,
) -> optimizer.Optimizer:
    if algorithm == Algorithm.HB:
        if momentum is None:
            raise ValueError("HB requires `momentum`.")
    elif momentum is not None:
        raise ValueError(f"{algorithm.value.upper()} does not use `momentum`.")

    if algorithm == Algorithm.PDHG:
        if stepsize_dual is None:
            raise ValueError("PDHG requires `stepsize_dual`.")
    elif stepsize_dual is not None:
        raise ValueError(
            f"{algorithm.value.upper()} does not use `stepsize_dual`."
        )

    match algorithm:
        case Algorithm.GD:
            return optimizer.GradientDescent(
                stepsize=stepsize,
                grad_map=_required(grad_map, "grad_map", algorithm),
                tol=tol,
                iters=iters,
            )
        case Algorithm.HB:
            return optimizer.HeavyBall(
                stepsize=stepsize,
                momentum=_required(momentum, "momentum", algorithm),
                grad_map=_required(grad_map, "grad_map", algorithm),
                tol=tol,
                iters=iters,
            )
        case Algorithm.NAG:
            return optimizer.AcceleratedGradient(
                stepsize=stepsize,
                grad_map=_required(grad_map, "grad_map", algorithm),
                momentum_scheduler=MomentumScheduler(),
                tol=tol,
                iters=iters,
            )
        case Algorithm.PGD:
            return optimizer.ProximalGradient(
                stepsize=stepsize,
                prox_map=_required(prox_map, "prox_map", algorithm),
                grad_map=_required(grad_map, "grad_map", algorithm),
                tol=tol,
                iters=iters,
            )
        case Algorithm.FISTA:
            return optimizer.FISTA(
                stepsize=stepsize,
                prox_map=_required(prox_map, "prox_map", algorithm),
                grad_map=_required(grad_map, "grad_map", algorithm),
                momentum_scheduler=MomentumScheduler(),
                tol=tol,
                iters=iters,
            )
        case Algorithm.PDHG:
            return optimizer.PDHG(
                stepsize_primal=stepsize,
                stepsize_dual=_required(
                    stepsize_dual, "stepsize_dual", algorithm
                ),
                prox_map_primal=_required(prox_map, "prox_map", algorithm),
                prox_map_dual=_required(
                    prox_map_dual, "prox_map_dual", algorithm
                ),
                lin_op=_required(lin_op, "lin_op", algorithm),
                lin_op_adj=_required(lin_op_adj, "lin_op_adj", algorithm),
                tol=tol,
                iters=iters,
            )
        case _:
            raise ValueError("Not supported")


def requires_grad_map(algorithm: Algorithm) -> bool:
    return algorithm != Algorithm.PDHG

def requires_prox_map(algorithm: Algorithm) -> bool:
    return algorithm in [Algorithm.PGD, Algorithm.FISTA, Algorithm.PDHG]

def requires_lin_op(algorithm: Algorithm) -> bool:
    return algorithm == Algorithm.PDHG

def requires_momentum(algorithm: Algorithm) -> bool:
    return algorithm == Algorithm.HB


def optimal_algorithm_parameters(
    algorithm: Algorithm,
    lipschitz: ScalarLike,
    strong_convexity: ScalarLike,
) -> tuple[ScalarLike, ScalarLike | None]:
    """Return classical parameters from smooth strongly-convex bounds."""
    _validate_problem_constant(lipschitz, "lipschitz", positive=True)
    _validate_problem_constant(
        strong_convexity,
        "strong_convexity",
        positive=algorithm == Algorithm.HB,
    )

    if _as_float(strong_convexity) > _as_float(lipschitz):
        raise ValueError("`strong_convexity` must not exceed `lipschitz`.")

    if algorithm == Algorithm.GD:
        return 2 / (lipschitz + strong_convexity), None
    if algorithm == Algorithm.HB:
        sqrt_l = _sqrt(lipschitz)
        sqrt_mu = _sqrt(strong_convexity)
        denominator = sqrt_l + sqrt_mu
        return (
            4 / denominator**2,
            ((sqrt_l - sqrt_mu) / denominator) ** 2,
        )
    if algorithm == Algorithm.NAG:
        return 1 / lipschitz, None
    if algorithm in {Algorithm.PGD, Algorithm.FISTA}:
        return 1 / lipschitz, None
    raise ValueError(
        "Automatic parameters from `(L, mu)` are not available for PDHG."
    )


def _sqrt(value: ScalarLike) -> ScalarLike:
    return (
        value.sqrt()
        if isinstance(value, torch.Tensor)
        else math.sqrt(value)
    )


def _as_float(value: ScalarLike) -> float:
    if isinstance(value, torch.Tensor):
        return float(value.detach().item())
    return float(value)


def _validate_problem_constant(
    value: ScalarLike,
    name: str,
    *,
    positive: bool,
) -> None:
    if isinstance(value, torch.Tensor):
        if value.numel() != 1:
            raise ValueError(f"`{name}` must be scalar.")
        scalar = float(value.detach().item())
    elif isinstance(value, Real) and not isinstance(value, bool):
        scalar = float(value)
    else:
        raise TypeError(f"`{name}` must be a real scalar or scalar tensor.")

    lower_bound_violated = scalar <= 0 if positive else scalar < 0
    if not math.isfinite(scalar) or lower_bound_violated:
        qualifier = "positive" if positive else "non-negative"
        raise ValueError(f"`{name}` must be finite and {qualifier}.")


def bind_gradient(
    grad_map: ParamGradMapType,
    params: RawParamType,
) -> PGradMapType:
    def bound_grad_map(x, *args, **kwds):
        return grad_map(x, params, *args, **kwds)

    return bound_grad_map


def bind_prox_map(
    prox_map: ParamProxMapType,
    params: RawParamType,
) -> ProxMapType:
    def bound_prox_map(x):
        return prox_map(x, params)

    return bound_prox_map


def bind_lin_op(
    lin_op: ParamLinOpType,
    params: RawParamType,
) -> RawFlatLinOpType:
    def bound_lin_op(x):
        return lin_op(x, params)

    return bound_lin_op


def prepare_gradient(
    smooth_obj: ParamSmoothObjType | None,
    grad_map: ParamGradMapType | None,
    params: RawParamType,
    algorithm: Algorithm,
) -> tuple[ParamGradMapType | None, PGradMapType | None]:
    if not requires_grad_map(algorithm):
        if smooth_obj is not None or grad_map is not None:
            raise ValueError("PDHG does not use `smooth_obj` or `grad_map`.")
        return None, None

    if (smooth_obj is None) == (grad_map is None):
        raise ValueError(
            f"{algorithm.value.upper()} requires exactly one of "
            "`smooth_obj` and `grad_map`."
        )

    if grad_map is None:
        if smooth_obj is None:
            raise RuntimeError("Gradient input validation failed.")
        canonical_grad_map = gradient(smooth_obj, create_graph=True)
    else:
        canonical_grad_map = grad_map
    return canonical_grad_map, bind_gradient(canonical_grad_map, params)


def prepare_prox_maps(
    prox_map: ParamProxMapType | None,
    prox_map_dual: ParamProxMapType | None,
    params_primal: RawParamType,
    params_dual: RawParamType,
    algorithm: Algorithm,
) -> tuple[ProxMapType | None, ProxMapType | None]:
    if not requires_prox_map(algorithm):
        if prox_map is not None or prox_map_dual is not None:
            raise ValueError(
                f"{algorithm.value.upper()} does not use proximal maps."
            )
        return None, None

    if prox_map is None:
        raise ValueError(f"{algorithm.value.upper()} requires `prox_map`.")

    bound_prox_map = bind_prox_map(
        prox_map,
        params_primal,
    )
    if algorithm != Algorithm.PDHG:
        if prox_map_dual is not None:
            raise ValueError(
                f"{algorithm.value.upper()} does not use `prox_map_dual`."
            )
        return bound_prox_map, None

    if prox_map_dual is None:
        raise ValueError("PDHG requires `prox_map_dual`.")
    return (
        bound_prox_map,
        bind_prox_map(prox_map_dual, params_dual),
    )


def prepare_pdhg_linear_operators(
    lin_op: ParamLinOpType | None,
    lin_op_adj: ParamLinOpType | None,
    zero_el: FlatRawVarType,
    params: RawParamType,
    algorithm: Algorithm,
) -> tuple[
    ParamLinOpType | None,
    ParamLinOpType | None,
    RawFlatLinOpType | None,
    RawFlatLinOpType | None,
]:
    if not requires_lin_op(algorithm):
        if lin_op is not None or lin_op_adj is not None:
            raise ValueError(
                f"{algorithm.value.upper()} does not use linear operators."
            )
        return None, None, None, None

    if lin_op is None:
        raise ValueError("PDHG requires `lin_op`.")

    canonical_lin_op_adj = (
        lin_op_adj
        if lin_op_adj is not None
        else adjoint(lin_op, zero_el, create_graph=True)
    )
    return (
        lin_op,
        canonical_lin_op_adj,
        bind_lin_op(lin_op, params),
        bind_lin_op(canonical_lin_op_adj, params),
    )
