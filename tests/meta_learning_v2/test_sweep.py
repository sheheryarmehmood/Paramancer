import csv

from examples.meta_learning_v2.experiment import (
    ExperimentConfig,
    _save_history,
)
from examples.meta_learning_v2.sweep import build_configs, combine_results
from examples.meta_learning_v2.train import (
    BatchMetrics,
    EpochMetrics,
    ObjectiveErrors,
)
from paramancer.bloptim.util import Algorithm


def test_build_configs_uses_truncation_only_for_unrolled():
    configs = build_configs(
        ("unrolled", "implicit"),
        (Algorithm.GD, Algorithm.HB),
        ("cold",),
        (0, 10),
        (0, 1),
    )

    assert len(configs) == 12
    assert {
        config.truncation for config in configs if config.backend == "implicit"
    } == {0}


def test_combine_results_writes_summary(tmp_path):
    configs = [
        ExperimentConfig(backend="unrolled", seed=0),
        ExperimentConfig(backend="implicit", seed=1),
    ]
    metrics = EpochMetrics(
        loss=1.0,
        accuracy=0.5,
        num_episodes=2,
        num_batches=1,
        objective_errors=ObjectiveErrors(0.3, 0.3, 0.1),
        objective_error_stds=ObjectiveErrors(0.03, 0.03, 0.01),
        reverse_error=0.2,
        reverse_error_std=0.02,
        forward_time=0.4,
        forward_time_std=0.04,
        backward_time=0.6,
        backward_time_std=0.06,
        batch_metrics=(
            BatchMetrics(
                loss=1.0,
                correct=1,
                num_queries=2,
                num_episodes=2,
                objective_errors=ObjectiveErrors(0.3, 0.3, 0.1),
                reverse_error=0.2,
                forward_time=0.4,
                backward_time=0.6,
            ),
        ),
    )
    for config in configs:
        _save_history(config, [metrics], tmp_path)

    summary = combine_results(tmp_path, configs)

    with summary.open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    assert len(rows) == 2
    assert {row["backend"] for row in rows} == {"unrolled", "implicit"}
