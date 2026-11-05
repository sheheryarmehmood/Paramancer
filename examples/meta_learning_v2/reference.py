from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import TypeAlias

import torch
import torch.nn.functional as F

from paramancer.bloptim.solver import Solver
from paramancer.bloptim.util import Algorithm

from .base_learner import BaseLearner, BaseLearnerParamType

OuterGradient: TypeAlias = tuple[torch.Tensor, ...]


@dataclass(frozen=True)
class ReferenceResult:
    loss: torch.Tensor
    logits: torch.Tensor
    base_params: BaseLearnerParamType
    outer_gradient: OuterGradient


class ReferenceEvaluator:
    """Compute an accurate lower-level solution and outer gradient.

    Support and query features should come from the same encoder pass used by
    the method being evaluated. This keeps both outer gradients tied to the
    same encoder parameters and episode.
    """

    def __init__(
        self,
        num_classes: int,
        reg_strength: float | torch.Tensor,
        *,
        iters: int = 2_000,
        tol: float = 1e-10,
    ) -> None:
        solver = Solver(
            backend="implicit",
            algorithm=Algorithm.HB,
            iters=iters,
            tol=tol,
        )
        self._base_learner = BaseLearner(
            solver,
            num_classes=num_classes,
            reg_strength=reg_strength,
        )

    def __call__(
        self,
        support_features: torch.Tensor,
        support_targets: torch.Tensor,
        query_features: torch.Tensor,
        query_targets: torch.Tensor,
        outer_parameters: Iterable[torch.Tensor],
        *,
        retain_graph: bool = True,
    ) -> ReferenceResult:
        parameters = tuple(outer_parameters)
        if not parameters:
            raise ValueError("`outer_parameters` must not be empty.")

        self._base_learner.fit(support_features, support_targets)
        logits = self._base_learner.predict(query_features)
        loss = F.cross_entropy(
            logits.flatten(0, 1),
            query_targets.long().flatten(),
        )
        outer_gradient = torch.autograd.grad(
            loss,
            parameters,
            retain_graph=retain_graph,
        )

        return ReferenceResult(
            loss=loss.detach(),
            logits=logits.detach(),
            base_params=self._base_learner.wb_opt,
            outer_gradient=tuple(grad.detach() for grad in outer_gradient),
        )
