from __future__ import annotations

from numbers import Real
from typing import Any

import torch

from ...optim import Optimizer
from ...variable import AlgoVar


class UnrolledBackend:
    def __init__(
        self,
        truncation: int = 0,
        omega: float = 1,
    ):
        if isinstance(truncation, bool) or not isinstance(truncation, int):
            raise TypeError("`truncation` must be an integer.")
        if truncation < 0:
            raise ValueError("`truncation` must be non-negative.")
        if isinstance(omega, bool) or not isinstance(omega, Real):
            raise TypeError("`omega` must be a real number.")
        if not 1 <= omega <= 4:
            raise ValueError("`omega` must lie in the interval [1, 4].")

        self._truncation = truncation
        self._omega = float(omega)
        self._idle_sol: AlgoVar | None = None
        self._fwd_sol: AlgoVar | None = None

    def __call__(
        self,
        optimizer: Optimizer,
        init: AlgoVar,
        *args: Any,
        **kwds: Any,
    ) -> AlgoVar:
        total_iters = optimizer.iters
        if self._truncation > total_iters:
            raise ValueError("`truncation` must not exceed `optimizer.iters`.")

        self._idle_sol = None
        self._fwd_sol = None
        idle_iters = self._idle_iters()
        unrolled_iters = self._unrolled_iters(total_iters)

        try:
            new_init = init
            if idle_iters > 0:
                with torch.no_grad():
                    new_init = optimizer(
                        init,
                        *args,
                        iters=idle_iters,
                        **kwds,
                    )
            self._idle_sol = new_init.clone()
            self._fwd_sol = optimizer(
                new_init,
                *args,
                iters=unrolled_iters,
                **kwds,
            )
        finally:
            optimizer.iters = total_iters
        return self.fwd_sol

    def _idle_iters(self) -> int:
        idle_iters = (1 + self._omega) * self._truncation
        if isinstance(idle_iters, float) and not idle_iters.is_integer():
            raise ValueError(
                "`(1 + omega) * truncation` must be an integer-valued "
                "iteration count."
            )
        return int(idle_iters)

    def _unrolled_iters(self, total_iters: int) -> int:
        return total_iters - self._truncation

    @property
    def fwd_sol(self) -> AlgoVar:
        if self._fwd_sol is None:
            raise RuntimeError(
                "Forward solution accessed before `forward` ran."
            )
        return self._fwd_sol

    @property
    def idle_sol(self) -> AlgoVar:
        if self._idle_sol is None:
            raise RuntimeError(
                "Idle solution accessed before the backend ran."
            )
        return self._idle_sol
