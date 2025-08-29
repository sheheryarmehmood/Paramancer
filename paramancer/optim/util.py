from dataclasses import dataclass
import torch
from typing import Optional, Union
from paramancer.operators.norms import l2


@dataclass
class OptimizationResult:
    solution: Union[torch.Tensor, tuple[torch.Tensor]]
    iterations: int
    metric: Optional[float]
    converged: bool
