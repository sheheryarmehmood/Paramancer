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
        support_feats = self._encoder(support_images)
        query_feats = self._encoder(query_images)
        
        self._base_learner.fit(
            support_feats,
            support_targets,
            initial_params,
            adjoint_init,
            stepsize=stepsize,
            **solver_kwargs,
        )
        
        return self._base_learner.predict(query_feats)
    
    @property
    def base_learner_params(self) -> BaseLearnerParamType:
        return self._base_learner.wb_opt

    @property
    def base_learner_adjoint_state(
        self
    ) -> BaseLearnerAdjointType:
        return self._base_learner.adjoint_state
