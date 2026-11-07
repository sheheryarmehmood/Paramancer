from __future__ import annotations

import argparse
import csv
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
from .reference import ReferenceEvaluator
from .train import EpochMetrics, MetaTrain
from .util import build_class_index

NUM_EPOCHS = 5
NUM_EPISODES = 20
META_BATCH_SIZE = 2
N_WAY = 5
N_SUPPORT_PER_CLASS = 1
N_QUERY_PER_CLASS = 2
EMBEDDING_DIM = 64
INNER_ITERS = 150
INNER_TOL = 1e-5
REG_STRENGTH = 1e-3
OUTER_LR = 1e-3
REFERENCE_ITERS = 2_000
REFERENCE_TOL = 1e-10

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
    results_dir: Path | str | None = None,
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
        reference=ReferenceEvaluator(
            num_classes=N_WAY,
            reg_strength=REG_STRENGTH,
            iters=REFERENCE_ITERS,
            tol=REFERENCE_TOL,
        ),
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
        errors = metrics.objective_errors
        error_stds = metrics.objective_error_stds
        if (
            errors is None
            or error_stds is None
            or metrics.reverse_error is None
            or metrics.reverse_error_std is None
        ):
            raise RuntimeError("Reference errors are missing.")
        print(
            f"epoch={epoch} "
            f"loss={metrics.loss:.6f} "
            f"accuracy={100 * metrics.accuracy:.2f}% "
            f"objective_error_initial={errors.initial:.6e}"
            f"±{error_stds.initial:.2e} "
            f"objective_error_idle={errors.idle:.6e}"
            f"±{error_stds.idle:.2e} "
            f"objective_error_final={errors.final:.6e}±{error_stds.final:.2e} "
            f"reverse_error={metrics.reverse_error:.6e}"
            f"±{metrics.reverse_error_std:.2e} "
            f"forward_time={metrics.forward_time:.6f}"
            f"±{metrics.forward_time_std:.2e}s "
            f"backward_time={metrics.backward_time:.6f}"
            f"±{metrics.backward_time_std:.2e}s"
        )

    output_path = _save_history(config, history, results_dir)
    print(f"results={output_path}")
    return history


def _save_history(
    config: ExperimentConfig,
    history: list[EpochMetrics],
    results_dir: Path | str | None,
) -> Path:
    directory = (
        Path(results_dir)
        if results_dir is not None
        else Path(__file__).resolve().parent / "results"
    )
    directory.mkdir(parents=True, exist_ok=True)
    output_path = directory / _result_filename(config)
    fieldnames = [
        "epoch",
        "batch",
        "loss",
        "accuracy",
        "objective_error_initial",
        "objective_error_idle",
        "objective_error_final",
        "reverse_error",
        "forward_time_seconds",
        "backward_time_seconds",
    ]
    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for epoch, metrics in enumerate(history, start=1):
            for batch, batch_metrics in enumerate(
                metrics.batch_metrics,
                start=1,
            ):
                errors = batch_metrics.objective_errors
                if errors is None or batch_metrics.reverse_error is None:
                    raise RuntimeError("Reference errors are missing.")
                writer.writerow(
                    {
                        "epoch": epoch,
                        "batch": batch,
                        "loss": batch_metrics.loss,
                        "accuracy": (
                            batch_metrics.correct / batch_metrics.num_queries
                        ),
                        "objective_error_initial": errors.initial,
                        "objective_error_idle": errors.idle,
                        "objective_error_final": errors.final,
                        "reverse_error": batch_metrics.reverse_error,
                        "forward_time_seconds": batch_metrics.forward_time,
                        "backward_time_seconds": batch_metrics.backward_time,
                    }
                )
    return output_path


def _result_filename(config: ExperimentConfig) -> str:
    return (
        f"backend={config.backend}_"
        f"algorithm={config.algorithm.value}_"
        f"initialization={config.initialization}_"
        f"truncation={config.truncation}_"
        f"seed={config.seed}.csv"
    )


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
    parser.add_argument(
        "--algorithm",
        choices=("gd", "hb", "nag"),
        default="gd",
    )
    parser.add_argument("--truncation", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default=None)
    parser.add_argument("--results-dir", default=None)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    run_experiment(
        ExperimentConfig(
            backend=args.backend,
            algorithm=Algorithm(args.algorithm),
            initialization=args.initialization,
            truncation=args.truncation,
            seed=args.seed,
        ),
        device=args.device,
        results_dir=args.results_dir,
    )


if __name__ == "__main__":
    main()
