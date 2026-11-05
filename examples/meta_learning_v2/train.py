from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any, TypeAlias

import torch
import torch.nn.functional as F

from .initialization import StateStrategy
from .meta_learner import MetaLearner
from .reference import ReferenceEvaluator
from .base_learner import BaseLearnerParamType

EpisodeBatch: TypeAlias = tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]


@dataclass(frozen=True)
class BatchMetrics:
    loss: float
    correct: int
    num_queries: int
    num_episodes: int
    objective_errors: ObjectiveErrors | None = None


@dataclass(frozen=True)
class ObjectiveErrors:
    initial: float
    idle: float
    final: float


@dataclass(frozen=True)
class EpochMetrics:
    loss: float
    accuracy: float
    num_episodes: int
    num_batches: int
    objective_errors: ObjectiveErrors | None = None


class MetaTrain:
    def __init__(
        self,
        meta_learner: MetaLearner,
        outer_optimizer: torch.optim.Optimizer,
        state_strategy: StateStrategy,
        device: torch.device | str,
        inner_solver_kwargs: dict[str, Any] | None = None,
        reference: ReferenceEvaluator | None = None,
    ) -> None:
        self._meta_learner = meta_learner
        self._outer_optimizer = outer_optimizer
        self._state_strategy = state_strategy
        self._device = torch.device(device)
        if self._device.type == "cuda" and self._device.index is None:
            self._device = torch.device("cuda", torch.cuda.current_device())
        self._inner_solver_kwargs = dict(inner_solver_kwargs or {})
        self._reference = reference
        self._validate_model_device()

    def train_epoch(
        self,
        loader: Iterable[EpisodeBatch],
    ) -> EpochMetrics:
        self._meta_learner.train()

        total_loss = 0.0
        total_correct = 0
        total_queries = 0
        total_episodes = 0
        num_batches = 0
        objective_error_totals = [0.0, 0.0, 0.0]

        for batch in loader:
            metrics = self.train_step(batch)
            total_loss += metrics.loss * metrics.num_queries
            total_correct += metrics.correct
            total_queries += metrics.num_queries
            total_episodes += metrics.num_episodes
            num_batches += 1
            if metrics.objective_errors is not None:
                errors = metrics.objective_errors
                objective_error_totals[0] += errors.initial * metrics.num_queries
                objective_error_totals[1] += errors.idle * metrics.num_queries
                objective_error_totals[2] += errors.final * metrics.num_queries

        if num_batches == 0:
            raise ValueError("Cannot train on an empty episode loader.")

        objective_errors = None
        if self._reference is not None:
            objective_errors = ObjectiveErrors(
                *(total / total_queries for total in objective_error_totals)
            )

        return EpochMetrics(
            loss=total_loss / total_queries,
            accuracy=total_correct / total_queries,
            num_episodes=total_episodes,
            num_batches=num_batches,
            objective_errors=objective_errors,
        )

    def train_step(self, batch: EpisodeBatch) -> BatchMetrics:
        (
            episode_ids,
            support_images,
            support_targets,
            query_images,
            query_targets,
            _,
        ) = batch
        initial_params, adjoint_init = self._state_strategy.get(episode_ids)

        support_images = self._to_device(support_images)
        support_targets = self._to_device(support_targets)
        query_images = self._to_device(query_images)
        query_targets = self._to_device(query_targets)

        self._outer_optimizer.zero_grad(set_to_none=True)
        support_features = self._meta_learner.encode(support_images)
        query_features = self._meta_learner.encode(query_images)
        query_logits = self._meta_learner.forward_features(
            support_features,
            support_targets,
            query_features,
            initial_params,
            adjoint_init,
            **self._inner_solver_kwargs,
        )
        loss = F.cross_entropy(
            query_logits.flatten(0, 1),
            query_targets.long().flatten(),
        )

        objective_errors = self._objective_errors(
            support_features,
            support_targets,
            query_features,
            query_targets,
            loss,
        )
        loss.backward()

        self._state_strategy.update_forward(
            episode_ids,
            self._meta_learner.base_learner_params,
        )
        self._outer_optimizer.step()

        predictions = query_logits.detach().argmax(dim=-1)
        correct = int((predictions == query_targets).sum().item())
        return BatchMetrics(
            loss=float(loss.detach().item()),
            correct=correct,
            num_queries=query_targets.numel(),
            num_episodes=episode_ids.numel(),
            objective_errors=objective_errors,
        )

    def _objective_errors(
        self,
        support_features: torch.Tensor,
        support_targets: torch.Tensor,
        query_features: torch.Tensor,
        query_targets: torch.Tensor,
        final_loss: torch.Tensor,
    ) -> ObjectiveErrors | None:
        if self._reference is None:
            return None

        reference = self._reference(
            support_features,
            support_targets,
            query_features,
            query_targets,
            self._meta_learner.parameters(),
        )
        with torch.no_grad():
            initial_loss = self._query_loss(
                query_features,
                query_targets,
                self._meta_learner.base_learner_initial_params,
            )
            idle_loss = self._query_loss(
                query_features,
                query_targets,
                self._meta_learner.base_learner_idle_params,
            )
            reference_loss = reference.loss
            return ObjectiveErrors(
                initial=float((initial_loss - reference_loss).abs().item()),
                idle=float((idle_loss - reference_loss).abs().item()),
                final=float((final_loss.detach() - reference_loss).abs().item()),
            )

    def _query_loss(
        self,
        query_features: torch.Tensor,
        query_targets: torch.Tensor,
        params: BaseLearnerParamType,
    ) -> torch.Tensor:
        logits = self._meta_learner.predict_features_with(
            query_features,
            params,
        )
        return F.cross_entropy(
            logits.flatten(0, 1),
            query_targets.long().flatten(),
        )

    def _to_device(self, tensor: torch.Tensor) -> torch.Tensor:
        return tensor.to(self._device, non_blocking=True)

    def _validate_model_device(self) -> None:
        devices = {
            parameter.device for parameter in self._meta_learner.parameters()
        }
        devices.update(
            buffer.device for buffer in self._meta_learner.buffers()
        )
        if devices and devices != {self._device}:
            raise ValueError(
                "Move `meta_learner` to the training device before creating "
                "its outer optimizer and `MetaTrain`."
            )
