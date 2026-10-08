from __future__ import annotations
import torch

from .base_learner import BaseLearner


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
        initial_params: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        support_feats = self._encoder(support_images)
        query_feats = self._encoder(query_images)
        
        self._base_learner.fit(support_feats, support_targets, initial_params)
        
        return self._base_learner.predict(query_feats)
    
    @property
    def base_learner_params(self) -> tuple[torch.Tensor, torch.Tensor]:
        return self._base_learner.wb_opt
        