from __future__ import annotations

import argparse
import csv
from itertools import product
from pathlib import Path
import re
from typing import Iterable

from paramancer.bloptim.util import Algorithm

from .experiment import (
    INNER_ITERS,
    ExperimentConfig,
    _result_filename,
    run_experiment,
)

SUPPORTED_ALGORITHMS = (Algorithm.GD, Algorithm.HB, Algorithm.NAG)
RESULT_PATTERN = re.compile(
    r"^backend=(unrolled|implicit)_"
    r"algorithm=(gd|hb|nag)_"
    r"initialization=(cold|forward_warm)_"
    r"truncation=(\d+)_seed=(-?\d+)\.csv$"
)


def build_configs(
    backends: Iterable[str],
    algorithms: Iterable[Algorithm],
    initializations: Iterable[str],
    truncations: Iterable[int],
    seeds: Iterable[int],
) -> list[ExperimentConfig]:
    backends = tuple(dict.fromkeys(backends))
    algorithms = tuple(dict.fromkeys(algorithms))
    initializations = tuple(dict.fromkeys(initializations))
    truncations = tuple(dict.fromkeys(truncations))
    seeds = tuple(dict.fromkeys(seeds))

    if any(truncation < 0 or truncation > INNER_ITERS for truncation in truncations):
        raise ValueError(f"Truncations must lie between 0 and {INNER_ITERS}.")

    configs = []
    for backend, algorithm, initialization, seed in product(
        backends,
        algorithms,
        initializations,
        seeds,
    ):
        backend_truncations = truncations if backend == "unrolled" else (0,)
        configs.extend(
            ExperimentConfig(
                backend=backend,
                algorithm=algorithm,
                initialization=initialization,
                truncation=truncation,
                seed=seed,
            )
            for truncation in backend_truncations
        )
    return configs


def run_sweep(
    configs: Iterable[ExperimentConfig],
    *,
    device: str | None = None,
    results_dir: Path | str | None = None,
) -> Path:
    directory = _results_directory(results_dir)
    configs = list(configs)
    for index, config in enumerate(configs, start=1):
        print(f"run={index}/{len(configs)}")
        run_experiment(config, device=device, results_dir=directory)
    return combine_results(directory, configs)


def combine_results(
    results_dir: Path | str,
    configs: Iterable[ExperimentConfig] | None = None,
) -> Path:
    directory = Path(results_dir)
    if configs is None:
        input_paths = sorted(directory.glob("backend=*.csv"))
        path_configs = [
            (path, _config_from_result_filename(path.name))
            for path in input_paths
        ]
    else:
        path_configs = [
            (directory / _result_filename(config), config)
            for config in configs
        ]
        input_paths = [path for path, _ in path_configs]

    missing = [path for path in input_paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing result file: {missing[0]}")
    if not input_paths:
        raise ValueError(f"No per-run CSV files found in {directory}.")

    rows: list[dict[str, str | int]] = []
    fieldnames: list[str] | None = None
    for path, config in path_configs:
        with path.open(newline="", encoding="utf-8") as file:
            reader = csv.DictReader(file)
            current_fieldnames = reader.fieldnames
            if current_fieldnames is None:
                raise ValueError(f"CSV file has no header: {path}")
            if fieldnames is None:
                fieldnames = current_fieldnames
            elif current_fieldnames != fieldnames:
                raise ValueError(f"CSV columns do not match: {path}")
            for row in reader:
                rows.append(
                    {
                        "backend": config.backend,
                        "algorithm": config.algorithm.value,
                        "initialization": config.initialization,
                        "truncation": config.truncation,
                        "seed": config.seed,
                        **row,
                    }
                )

    if fieldnames is None:
        raise RuntimeError("Failed to read the result columns.")

    output_path = directory / "summary.csv"
    temporary_path = directory / ".summary.csv.tmp"
    summary_fieldnames = [
        "backend",
        "algorithm",
        "initialization",
        "truncation",
        "seed",
        *fieldnames,
    ]
    with temporary_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=summary_fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    temporary_path.replace(output_path)
    print(f"summary={output_path}")
    return output_path


def _config_from_result_filename(filename: str) -> ExperimentConfig:
    match = RESULT_PATTERN.fullmatch(filename)
    if match is None:
        raise ValueError(f"Unrecognized result filename: {filename}")
    backend, algorithm, initialization, truncation, seed = match.groups()
    return ExperimentConfig(
        backend=backend,
        algorithm=Algorithm(algorithm),
        initialization=initialization,
        truncation=int(truncation),
        seed=int(seed),
    )


def _results_directory(results_dir: Path | str | None) -> Path:
    return (
        Path(results_dir)
        if results_dir is not None
        else Path(__file__).resolve().parent / "results"
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--backends",
        nargs="+",
        choices=("unrolled", "implicit"),
        default=("unrolled", "implicit"),
    )
    parser.add_argument(
        "--algorithms",
        nargs="+",
        choices=tuple(algorithm.value for algorithm in SUPPORTED_ALGORITHMS),
        default=("gd",),
    )
    parser.add_argument(
        "--initializations",
        nargs="+",
        choices=("cold", "forward_warm"),
        default=("cold",),
    )
    parser.add_argument("--truncations", nargs="+", type=int, default=(0,))
    parser.add_argument("--seeds", nargs="+", type=int, default=(0,))
    parser.add_argument("--device", default=None)
    parser.add_argument("--results-dir", default=None)
    parser.add_argument("--task-index", type=int, default=None)
    parser.add_argument("--list-configs", action="store_true")
    parser.add_argument("--combine-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    configs = build_configs(
        args.backends,
        (Algorithm(value) for value in args.algorithms),
        args.initializations,
        args.truncations,
        args.seeds,
    )
    directory = _results_directory(args.results_dir)

    if args.list_configs:
        for index, config in enumerate(configs):
            print(index, config)
        return
    if args.combine_only:
        combine_results(directory, configs)
        return
    if args.task_index is not None:
        if not 0 <= args.task_index < len(configs):
            raise ValueError(
                f"`task-index` must lie between 0 and {len(configs) - 1}."
            )
        run_experiment(
            configs[args.task_index],
            device=args.device,
            results_dir=directory,
        )
        return

    run_sweep(configs, device=args.device, results_dir=directory)


if __name__ == "__main__":
    main()
