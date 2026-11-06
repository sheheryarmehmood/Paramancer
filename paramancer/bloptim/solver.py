from __future__ import annotations

from typing import Any

from ..variable.flat import FlatVar
from ..variable.pair import PairVar
from ..variable.parameter import ParamBundle
from ..variable.types import (
    AlgoVarLike,
    AlgoVarType,
    IndexMapType,
    OptVarType,
    ParamGradMapType,
    ParamLinOpType,
    ParamProxMapType,
    ParamSmoothObjType,
    RawParamType,
    ScalarLike,
)
from ..variable.util import zeros_like
from .implicit.backend import ImplicitBackend
from .unrolled.backend import UnrolledBackend
from .util import (
    Algorithm,
    create_optimizer,
    prepare_gradient,
    prepare_pdhg_linear_operators,
    prepare_prox_maps,
)


class Solver:
    def __init__(
        self,
        backend: str,
        algorithm: Algorithm,
        truncation: int = 0,
        omega: float = 1,
        tol: float = 1e-6,
        iters: int = 100,
    ):
        if backend not in {"unrolled", "implicit"}:
            raise ValueError(
                f"Unsupported backend {backend!r}. "
                f"Expected one of: unrolled, implicit."
            )
        self._backend_name = backend
        if not isinstance(algorithm, Algorithm):
            raise TypeError("`algorithm` must be an `Algorithm`.")
        if backend == "unrolled":
            self._backend = UnrolledBackend(
                truncation=truncation,
                omega=omega,
            )
        else:
            self._backend = ImplicitBackend()
        self._algorithm = algorithm
        self._tol = tol
        self._iters = iters

    @property
    def algorithm(self) -> Algorithm:
        return self._algorithm

    @property
    def backend(self) -> str:
        return self._backend_name
    
    def solve(
        self,
        init: OptVarType,
        outer_vars: RawParamType,
        stepsize: ScalarLike,
        *args: Any,
        outer_var_indices: IndexMapType | None = None,
        momentum: ScalarLike | None = None,
        smooth_obj: ParamSmoothObjType | None = None,
        grad_map: ParamGradMapType | None = None,
        prox_map: ParamProxMapType | None = None,
        stepsize_dual: ScalarLike | None = None,
        prox_map_dual: ParamProxMapType | None = None,
        lin_op: ParamLinOpType | None = None,
        lin_op_adj: ParamLinOpType | None = None,
        adjoint_init: AlgoVarLike | None = None,
        **kwds: Any,
    ) -> OptVarType:
        params = ParamBundle(outer_vars, outer_var_indices)
        grad_map, bound_grad_map = prepare_gradient(
            smooth_obj,
            grad_map,
            params.grad,
            self._algorithm,
        )

        prox_params = (
            params.primal
            if self._algorithm == Algorithm.PDHG
            else params.prox
        )
        bound_prox_map, bound_prox_map_dual = prepare_prox_maps(
            prox_map,
            prox_map_dual,
            prox_params,
            params.dual,
            self._algorithm,
        )

        (
            lin_op,
            lin_op_adj,
            bound_lin_op,
            bound_lin_op_adj,
        ) = prepare_pdhg_linear_operators(
            lin_op,
            lin_op_adj,
            zeros_like(init),
            params.lin_op,
            self._algorithm,
        )

        optimizer_init = FlatVar(init)
        if bound_lin_op is not None:
            optimizer_init = PairVar(init, zeros_like(bound_lin_op(init)))

        optimizer = create_optimizer(
            self._algorithm,
            stepsize,
            momentum,
            grad_map=bound_grad_map,
            prox_map=bound_prox_map,
            stepsize_dual=stepsize_dual,
            prox_map_dual=bound_prox_map_dual,
            lin_op=bound_lin_op,
            lin_op_adj=bound_lin_op_adj,
            tol=self._tol,
            iters=self._iters,
        )

        if isinstance(self._backend, ImplicitBackend):
            self._backend(
                optimizer,
                optimizer_init,
                params,
                *args,
                grad_map=grad_map,
                prox_map=prox_map,
                prox_map_dual=prox_map_dual,
                lin_op=lin_op,
                lin_op_adj=lin_op_adj,
                adjoint_init=adjoint_init,
                **kwds,
            )
        else:
            if adjoint_init is not None:
                raise ValueError(
                    "`adjoint_init` is only supported by the implicit backend."
                )
            self._backend(
                optimizer,
                optimizer_init,
                *args,
                **kwds,
            )
        return self.solution

    def __call__(
        self,
        init: OptVarType,
        outer_vars: RawParamType,
        stepsize: ScalarLike,
        *args: Any,
        outer_var_indices: IndexMapType | None = None,
        momentum: ScalarLike | None = None,
        smooth_obj: ParamSmoothObjType | None = None,
        grad_map: ParamGradMapType | None = None,
        prox_map: ParamProxMapType | None = None,
        stepsize_dual: ScalarLike | None = None,
        prox_map_dual: ParamProxMapType | None = None,
        lin_op: ParamLinOpType | None = None,
        lin_op_adj: ParamLinOpType | None = None,
        adjoint_init: AlgoVarLike | None = None,
        **kwds: Any,
    ) -> OptVarType:
        return self.solve(
            init,
            outer_vars,
            stepsize,
            *args,
            outer_var_indices=outer_var_indices,
            momentum=momentum,
            smooth_obj=smooth_obj,
            grad_map=grad_map,
            prox_map=prox_map,
            stepsize_dual=stepsize_dual,
            prox_map_dual=prox_map_dual,
            lin_op=lin_op,
            lin_op_adj=lin_op_adj,
            adjoint_init=adjoint_init,
            **kwds,
        )

    @property
    def solution(self) -> OptVarType:
        sol = self._backend.fwd_sol
        return (
            sol.first.data
            if hasattr(sol, "first")
            else sol.data
        )

    @property
    def idle_solution(self) -> OptVarType:
        if not isinstance(self._backend, UnrolledBackend):
            raise TypeError(
                "Idle solution is only available with the unrolled backend."
            )
        sol = self._backend.idle_sol
        return sol.first.data if hasattr(sol, "first") else sol.data

    @property
    def adjoint_state(self) -> AlgoVarType:
        if not isinstance(self._backend, ImplicitBackend):
            raise TypeError(
                "Adjoint state is only available with the implicit backend."
            )
        return self._backend.adjoint_state.data
