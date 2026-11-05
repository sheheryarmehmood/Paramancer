import csv

from examples.meta_learning_v2.experiment import (
    ExperimentConfig,
    _save_history,
)
from examples.meta_learning_v2.train import EpochMetrics, ObjectiveErrors
from paramancer.bloptim.util import Algorithm


def test_save_history_writes_one_row_per_epoch(tmp_path):
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
        ),
    ]

    output_path = _save_history(config, history, tmp_path)

    with output_path.open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))

    assert len(rows) == 2
    assert rows[0]["epoch"] == "1"
    assert rows[1]["epoch"] == "2"
    assert "backend" not in rows[0]
    assert "seed" not in rows[0]
    assert rows[1]["objective_error_final_mean"] == "0.05"
    assert rows[1]["objective_error_final_std"] == "0.005"
    assert rows[1]["reverse_error_mean"] == "0.1"
    assert rows[0]["forward_time_seconds_mean"] == "0.4"
    assert rows[0]["backward_time_seconds_mean"] == "0.6"
