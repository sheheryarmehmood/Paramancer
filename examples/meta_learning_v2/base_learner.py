from __future__ import annotations

from typing import Any, TypeAlias

import torch
import torch.nn.functional as F

from paramancer.bloptim.solver import Solver
from paramancer.bloptim.util import (
    optimal_algorithm_parameters,
    requires_momentum,
)
from paramancer.variable.types import ScalarLike

BaseLearnerParamType: TypeAlias = tuple[torch.Tensor, torch.Tensor]
BaseLearnerParamPairType: TypeAlias = tuple[
    BaseLearnerParamType, BaseLearnerParamType
]
BaseLearnerAdjointType: TypeAlias = (
    BaseLearnerParamType | BaseLearnerParamPairType
)


class BaseLearner:
    def __init__(
        self,
        solver: Solver,
        num_classes: int,
        reg_strength: float | torch.Tensor,
    ) -> None:
        self._solver = solver
        self._wb_init: BaseLearnerParamType | None = None
        self._wb: BaseLearnerParamType | None = None
        self._num_classes = num_classes
        self._reg_strength = reg_strength
    
    def fit(
        self,
        features: torch.Tensor,
        targets: torch.Tensor,
        wb_init: BaseLearnerParamType | None = None,
        *,
        adjoint_init: BaseLearnerAdjointType | None = None,
        stepsize: ScalarLike | None = None,
        **solver_kwargs: Any,
    ) -> None:
        if wb_init is not None:
            self._wb = tuple(
                tensor.detach().to(
                    device=features.device,
                    dtype=features.dtype,
                ) for tensor in wb_init
            )
        else:
            weight = torch.zeros(
                features.shape[0],
                self._num_classes,
                features.shape[-1],
                device=features.device,
                dtype=features.dtype,
            )
            bias = torch.zeros(
                features.shape[0],
                self._num_classes,
                device=features.device,
                dtype=features.dtype,
            )
            self._wb = (weight, bias)

        self._wb_init = tuple(tensor.detach().clone() for tensor in self._wb)
        
        momentum = solver_kwargs.get("momentum")
        needs_momentum = requires_momentum(self._solver.algorithm)
        if stepsize is None or (needs_momentum and momentum is None):
            lipschitz, strong_convexity = self._problem_constants(features)
            optimal_stepsize, optimal_momentum = optimal_algorithm_parameters(
                self._solver.algorithm,
                lipschitz,
                strong_convexity,
            )
            if stepsize is None:
                stepsize = optimal_stepsize
            if needs_momentum and momentum is None:
                solver_kwargs["momentum"] = optimal_momentum

        self._wb = self._solver(
            self._wb,
            features,
            stepsize,
            targets,
            smooth_obj=self._loss,
            adjoint_init=adjoint_init,
            **solver_kwargs,
        )

    def _problem_constants(
        self,
        features: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if features.ndim != 3:
            raise ValueError(
                "`features` must have shape (meta_batch, samples, features)."
            )

        with torch.no_grad():
            regularization = torch.as_tensor(
                self._reg_strength,
                device=features.device,
                dtype=features.dtype,
            )
            if regularization.numel() != 1 or regularization.item() <= 0:
                raise ValueError("`reg_strength` must be a positive scalar.")

            ones = features.new_ones((*features.shape[:-1], 1))
            augmented_features = torch.cat((features.detach(), ones), dim=-1)
            spectral_norms = torch.linalg.matrix_norm(
                augmented_features,
                ord=2,
            )
            data_lipschitz = (
                spectral_norms.square() / (2 * features.shape[1])
            )
            lipschitz = (data_lipschitz + regularization).max()
            strong_convexity = regularization

        return lipschitz, strong_convexity
    
    def predict(
        self,
        features: torch.Tensor
    ) -> torch.Tensor:
        if self._wb is None:
            raise RuntimeError(
                "BaseLearner.fit() must be called before predict()."
            )
        
        return self.predict_with(features, self._wb)

    def predict_with(
        self,
        features: torch.Tensor,
        wb: BaseLearnerParamType,
    ) -> torch.Tensor:
        return self._predict(features, wb)
    
    def _loss(
        self,
        wb_curr: BaseLearnerParamType,
        features,
        targets,
    ) -> torch.Tensor:
        logits = self._predict(features, wb_curr)
        weight, bias = wb_curr
        reg_term = self._reg_strength * (
            weight.square().sum() + bias.square().sum()
        ) / 2
        return F.cross_entropy(
            logits.flatten(0, 1),
            targets.long().flatten(),
            reduction="sum",
        ) / targets.shape[1] + reg_term
    
    def _predict(
        self,
        features: torch.Tensor,
        wb_curr: BaseLearnerParamType,
    ) -> torch.Tensor:
        W, b = wb_curr
        return torch.einsum("bse,bce->bsc", features, W) + b[:, None, :]
    
    @property
    def wb_opt(self) -> BaseLearnerParamType:
        if self._wb is None:
            raise RuntimeError("No solution is available before fit().")
        
        return tuple(w.detach() for w in self._wb)

    @property
    def initial_params(self) -> BaseLearnerParamType:
        if self._wb_init is None:
            raise RuntimeError("No initialization is available before fit().")
        return self._wb_init

    @property
    def idle_params(self) -> BaseLearnerParamType:
        if self._solver.backend == "implicit":
            return self.initial_params
        weight, bias = self._solver.idle_solution
        return weight.detach(), bias.detach()

    @property
    def adjoint_state(self) -> BaseLearnerAdjointType:
        return self._solver.adjoint_state
