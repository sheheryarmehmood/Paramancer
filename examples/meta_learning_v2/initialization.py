from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable
from typing import TypeAlias

import torch

from .base_learner import BaseLearnerAdjointType, BaseLearnerParamType

EpisodeIds: TypeAlias = torch.Tensor | Iterable[int]
InitialStateType: TypeAlias = tuple[
    BaseLearnerParamType | None,
    BaseLearnerAdjointType | None,
]


class StateStrategy(ABC):
    @abstractmethod
    def get(self, episode_ids: EpisodeIds) -> InitialStateType:
        pass

    @abstractmethod
    def update_forward(
        self,
        episode_ids: EpisodeIds,
        params: BaseLearnerParamType,
    ) -> None:
        pass

    @abstractmethod
    def update_adjoint(
        self,
        episode_ids: EpisodeIds,
        adjoint_state: BaseLearnerAdjointType,
    ) -> None:
        pass

    @abstractmethod
    def clear(self) -> None:
        pass


class ColdStart(StateStrategy):
    def get(self, episode_ids: EpisodeIds) -> InitialStateType:
        return None, None

    def update_forward(
        self,
        episode_ids: EpisodeIds,
        params: BaseLearnerParamType,
    ) -> None:
        pass

    def update_adjoint(
        self,
        episode_ids: EpisodeIds,
        adjoint_state: BaseLearnerAdjointType,
    ) -> None:
        pass

    def clear(self) -> None:
        pass


class ForwardWarmStart(StateStrategy):
    """CPU cache of per-episode base-learner solutions."""

    def __init__(self) -> None:
        self._states: dict[int, BaseLearnerParamType] = {}

    def get(
        self,
        episode_ids: EpisodeIds,
    ) -> InitialStateType:
        ids = self._normalize_ids(episode_ids)
        states = [self._states.get(episode_id) for episode_id in ids]

        if not states or any(state is None for state in states):
            return None, None
        known_states = [state for state in states if state is not None]

        params = (
            torch.stack([state[0] for state in known_states]),
            torch.stack([state[1] for state in known_states]),
        )
        return params, None

    def update_forward(
        self,
        episode_ids: EpisodeIds,
        params: BaseLearnerParamType,
    ) -> None:
        ids = self._normalize_ids(episode_ids)
        weight, bias = params

        if weight.ndim == 0 or bias.ndim == 0:
            raise ValueError("Warm-start tensors must have a batch dimension.")
        if weight.shape[0] != len(ids) or bias.shape[0] != len(ids):
            raise ValueError(
                "The warm-start batch size must match the number of episode "
                "IDs."
            )

        weight = weight.detach().cpu()
        bias = bias.detach().cpu()
        for index, episode_id in enumerate(ids):
            self._states[episode_id] = (
                weight[index].clone(),
                bias[index].clone(),
            )

    def update_adjoint(
        self,
        episode_ids: EpisodeIds,
        adjoint_state: BaseLearnerAdjointType,
    ) -> None:
        pass

    def clear(self) -> None:
        self._states.clear()

    def __len__(self) -> int:
        return len(self._states)

    @staticmethod
    def _normalize_ids(
        episode_ids: EpisodeIds,
    ) -> list[int]:
        if isinstance(episode_ids, torch.Tensor):
            if episode_ids.ndim > 1:
                raise ValueError(
                    "`episode_ids` must be a scalar or 1-D tensor."
                )
            ids = episode_ids.detach().cpu().reshape(-1)
            return [int(value) for value in ids]

        return [int(value) for value in episode_ids]
