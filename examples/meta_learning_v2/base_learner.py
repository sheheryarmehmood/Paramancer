from __future__ import annotations

import torch
import torch.nn.functional as F



class BaseLearner:
    def __init__(
        self,
        solver,
        num_classes: int,
        reg_strength: float | torch.Tensor,
    ) -> None:
        self._solver = solver
        self._wb = None
        self._num_classes = num_classes
        self._reg_strength = reg_strength        # We can learn this later.
    
    def fit(
        self,
        features: torch.Tensor,
        targets: torch.Tensor,
        wb_init: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> None:
        if wb_init is not None:
            self._wb = tuple(
                tensor.detach().detach().to(
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
        
        # This is to be dissected later.
        self._wb = self._solver(
            init=self._wb,          # The first input. Should allow the solver to identify the inner optimization variable.
            outer_vars=(features,), # The second input. Should allow the solver to identify the outer optimization 
            args=(targets,),        # Part of `*args` to be passed to `obj_map` or `grad_map`.
            obj_map=self._loss,     # `obj_map` xor `grad_map` should be passed to the solver.
            # `solver` could also support `*kwds`. Not needed here, I guess.
        )
    
    def predict(
        self,
        features: torch.Tensor
    ) -> torch.Tensor:
        if self._wb is None:
            raise RuntimeError(
                "BaseLearner.fit() must be called before predict()."
            )
        
        return self._predict(features, self._wb)
    
    def _loss(
        self,
        wb_curr: tuple[torch.Tensor, torch.Tensor],
        features,
        targets,
    ) -> torch.Tensor:
        logits = self._predict(features, wb_curr)
        reg_term = self._reg_strength * wb_curr[0].square().sum() / 2
        return F.cross_entropy(
            logits.flatten(0, 1),
            targets.long().flatten(),
            reduction="sum",
        ) / targets.shape[1] + reg_term
    
    def _predict(
        self,
        features: torch.Tensor,
        wb_curr: tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        W, b = wb_curr
        return torch.einsum("bse,bce->bsc", features, W) + b[:, None, :]
    
    @property
    def wb_opt(self) -> tuple[torch.Tensor, torch.Tensor]:
        if self._wb is None:
            raise RuntimeError("No solution is available before fit().")
        
        return tuple(w.detach() for w in self._wb)