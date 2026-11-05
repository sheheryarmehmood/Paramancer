from __future__ import annotations

from typing import Any

import torch

from paramancer.variable.types import ScalarLike

from .base_learner import (
    BaseLearner,
    BaseLearnerAdjointType,
    BaseLearnerParamType,
)


class MetaLearner(torch.nn.Module):
    def __init__(
        self,
        encoder: torch.nn.Module,
        base_learner: BaseLearner,
    ) -> None:
        super().__init__()
        self._encoder = encoder
        self._base_learner = base_learner
    
    def forward(
        self,
        support_images: torch.Tensor,
        support_targets: torch.Tensor,
        query_images: torch.Tensor,
        initial_params: BaseLearnerParamType | None = None,
        adjoint_init: BaseLearnerAdjointType | None = None,
        *,
        stepsize: ScalarLike | None = None,
        **solver_kwargs: Any,
    ) -> torch.Tensor:
        support_feats = self.encode(support_images)
        query_feats = self.encode(query_images)
        return self.forward_features(
            support_feats,
            support_targets,
            query_feats,
            initial_params,
            adjoint_init,
            stepsize=stepsize,
            **solver_kwargs,
        )

    def encode(self, images: torch.Tensor) -> torch.Tensor:
        return self._encoder(images)

    def forward_features(
        self,
        support_features: torch.Tensor,
        support_targets: torch.Tensor,
        query_features: torch.Tensor,
        initial_params: BaseLearnerParamType | None = None,
        adjoint_init: BaseLearnerAdjointType | None = None,
        *,
        stepsize: ScalarLike | None = None,
        **solver_kwargs: Any,
    ) -> torch.Tensor:
        self._base_learner.fit(
            support_features,
            support_targets,
            initial_params,
            adjoint_init=adjoint_init,
            stepsize=stepsize,
            **solver_kwargs,
        )

        return self._base_learner.predict(query_features)

    def predict_features_with(
        self,
        features: torch.Tensor,
        params: BaseLearnerParamType,
    ) -> torch.Tensor:
        return self._base_learner.predict_with(features, params)
    
    @property
    def base_learner_params(self) -> BaseLearnerParamType:
        return self._base_learner.wb_opt

    @property
    def base_learner_initial_params(self) -> BaseLearnerParamType:
        return self._base_learner.initial_params

    @property
    def base_learner_idle_params(self) -> BaseLearnerParamType:
        return self._base_learner.idle_params

    @property
    def base_learner_adjoint_state(
        self
    ) -> BaseLearnerAdjointType:
        return self._base_learner.adjoint_state
