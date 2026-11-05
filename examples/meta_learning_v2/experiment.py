from __future__ import annotations

import argparse
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from paramancer.bloptim.solver import Solver
from paramancer.bloptim.util import Algorithm

from .base_learner import BaseLearner
from .encoders import Conv2OmniEncoder
from .episodes import EpisodeDataset
from .initialization import ColdStart, ForwardWarmStart, StateStrategy
from .meta_learner import MetaLearner
from .train import EpochMetrics, MetaTrain
from .util import build_class_index

NUM_EPOCHS = 5
NUM_EPISODES = 20
META_BATCH_SIZE = 2
N_WAY = 5
N_SUPPORT_PER_CLASS = 1
N_QUERY_PER_CLASS = 2
EMBEDDING_DIM = 64
INNER_ITERS = 300
INNER_TOL = 1e-5
REG_STRENGTH = 1e-2
OUTER_LR = 1e-3

Backend = Literal["unrolled", "implicit"]
Initialization = Literal["cold", "forward_warm"]


@dataclass(frozen=True)
class ExperimentConfig:
    backend: Backend = "unrolled"
    algorithm: Algorithm = Algorithm.GD
    initialization: Initialization = "cold"
    truncation: int = 0
    seed: int = 0


def run_experiment(
    config: ExperimentConfig,
    *,
    device: torch.device | str | None = None,
) -> list[EpochMetrics]:
    device = _resolve_device(device)
    _seed_everything(config.seed)

    loader = _build_loader(config.seed, device)
    meta_learner = _build_meta_learner(config).to(device)
    outer_optimizer = torch.optim.Adam(
        meta_learner.parameters(),
        lr=OUTER_LR,
    )
    trainer = MetaTrain(
        meta_learner,
        outer_optimizer,
        _build_state_strategy(config.initialization),
        device,
    )

    print(
        f"device={device} backend={config.backend} "
        f"algorithm={config.algorithm.value} "
        f"initialization={config.initialization}"
    )
    history = []
    for epoch in range(1, NUM_EPOCHS + 1):
        metrics = trainer.train_epoch(loader)
        history.append(metrics)
        print(
            f"epoch={epoch} "
            f"loss={metrics.loss:.6f} "
            f"accuracy={100 * metrics.accuracy:.2f}%"
        )

    return history


def _build_meta_learner(config: ExperimentConfig) -> MetaLearner:
    solver = Solver(
        backend=config.backend,
        algorithm=config.algorithm,
        truncation=config.truncation,
        tol=INNER_TOL,
        iters=INNER_ITERS,
    )
    base_learner = BaseLearner(
        solver,
        num_classes=N_WAY,
        reg_strength=REG_STRENGTH,
    )
    return MetaLearner(
        Conv2OmniEncoder(embedding_dim=EMBEDDING_DIM),
        base_learner,
    )


def _build_loader(seed: int, device: torch.device) -> DataLoader:
    data_root = Path(__file__).resolve().parent / "data"
    dataset = datasets.Omniglot(
        root=data_root,
        background=True,
        download=True,
        transform=transforms.Compose(
            [
                transforms.Resize((28, 28)),
                transforms.ToTensor(),
            ]
        ),
    )
    episodes = EpisodeDataset(
        dataset,
        build_class_index(dataset),
        num_episodes=NUM_EPISODES,
        n_way=N_WAY,
        n_support_per_class=N_SUPPORT_PER_CLASS,
        n_query_per_class=N_QUERY_PER_CLASS,
        seed=seed,
    )
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        episodes,
        batch_size=META_BATCH_SIZE,
        shuffle=True,
        generator=generator,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )


def _build_state_strategy(initialization: Initialization) -> StateStrategy:
    if initialization == "cold":
        return ColdStart()
    if initialization == "forward_warm":
        return ForwardWarmStart()
    raise ValueError(f"Unsupported initialization strategy {initialization!r}.")


def _resolve_device(
    device: torch.device | str | None,
) -> torch.device:
    if device is None:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    resolved = torch.device(device)
    if resolved.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    return resolved


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--backend",
        choices=("unrolled", "implicit"),
        default="unrolled",
    )
    parser.add_argument(
        "--initialization",
        choices=("cold", "forward_warm"),
        default="cold",
    )
    parser.add_argument("--truncation", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    run_experiment(
        ExperimentConfig(
            backend=args.backend,
            initialization=args.initialization,
            truncation=args.truncation,
            seed=args.seed,
        ),
        device=args.device,
    )


if __name__ == "__main__":
    main()
