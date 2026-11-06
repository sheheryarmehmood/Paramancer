import csv

from examples.meta_learning_v2.experiment import (
    ExperimentConfig,
    _save_history,
)
from examples.meta_learning_v2.train import (
    BatchMetrics,
    EpochMetrics,
    ObjectiveErrors,
)
from paramancer.bloptim.util import Algorithm


def test_save_history_writes_one_row_per_batch(tmp_path):
    config = ExperimentConfig(
        backend="implicit",
        algorithm=Algorithm.GD,
        initialization="cold",
        truncation=0,
        seed=3,
    )
    history = [
        EpochMetrics(
            loss=1.2,
            accuracy=0.4,
            num_episodes=10,
            num_batches=5,
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
                    loss=1.2,
                    correct=4,
                    num_queries=10,
                    num_episodes=2,
                    objective_errors=ObjectiveErrors(0.3, 0.3, 0.1),
                    reverse_error=0.2,
                    forward_time=0.4,
                    backward_time=0.6,
                ),
            ),
        ),
        EpochMetrics(
            loss=0.9,
            accuracy=0.6,
            num_episodes=10,
            num_batches=5,
            objective_errors=ObjectiveErrors(0.4, 0.4, 0.05),
            objective_error_stds=ObjectiveErrors(0.04, 0.04, 0.005),
            reverse_error=0.1,
            reverse_error_std=0.01,
            forward_time=0.3,
            forward_time_std=0.03,
            backward_time=0.5,
            backward_time_std=0.05,
            batch_metrics=(
                BatchMetrics(
                    loss=0.9,
                    correct=6,
                    num_queries=10,
                    num_episodes=2,
                    objective_errors=ObjectiveErrors(0.4, 0.4, 0.05),
                    reverse_error=0.1,
                    forward_time=0.3,
                    backward_time=0.5,
                ),
            ),
        ),
    ]

    output_path = _save_history(config, history, tmp_path)

    with output_path.open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))

    assert len(rows) == 2
    assert rows[0]["epoch"] == "1"
    assert rows[1]["epoch"] == "2"
    assert rows[0]["batch"] == "1"
    assert "backend" not in rows[0]
    assert "seed" not in rows[0]
    assert rows[1]["objective_error_final"] == "0.05"
    assert rows[1]["reverse_error"] == "0.1"
    assert rows[0]["forward_time_seconds"] == "0.4"
    assert rows[0]["backward_time_seconds"] == "0.6"
