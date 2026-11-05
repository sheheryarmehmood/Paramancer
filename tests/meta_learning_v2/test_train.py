import torch

from examples.meta_learning_v2.base_learner import BaseLearner
from examples.meta_learning_v2.initialization import ForwardWarmStart
from examples.meta_learning_v2.meta_learner import MetaLearner
from examples.meta_learning_v2.reference import ReferenceEvaluator
from examples.meta_learning_v2.train import MetaTrain
from paramancer.bloptim.solver import Solver
from paramancer.bloptim.util import Algorithm


def test_meta_train_runs_epoch_and_updates_forward_cache():
    torch.manual_seed(0)
    encoder = torch.nn.Linear(3, 3, bias=False)
    solver = Solver(
        backend="unrolled",
        algorithm=Algorithm.GD,
        iters=5,
        tol=0.0,
    )
    base_learner = BaseLearner(solver, num_classes=2, reg_strength=0.5)
    meta_learner = MetaLearner(encoder, base_learner)
    optimizer = torch.optim.SGD(meta_learner.parameters(), lr=0.01)
    strategy = ForwardWarmStart()
    trainer = MetaTrain(
        meta_learner,
        optimizer,
        strategy,
        device="cpu",
        reference=ReferenceEvaluator(
            num_classes=2,
            reg_strength=0.5,
            iters=50,
            tol=1e-10,
        ),
    )

    batch = (
        torch.tensor([3, 1]),
        torch.randn(2, 4, 3),
        torch.tensor([[0, 1, 0, 1], [1, 0, 1, 0]]),
        torch.randn(2, 2, 3),
        torch.tensor([[0, 1], [1, 0]]),
        torch.tensor([[10, 11], [20, 21]]),
    )
    metrics = trainer.train_epoch([batch])

    assert torch.isfinite(torch.tensor(metrics.loss))
    assert 0.0 <= metrics.accuracy <= 1.0
    assert metrics.num_episodes == 2
    assert metrics.num_batches == 1
    assert metrics.objective_errors is not None
    assert metrics.objective_errors.initial >= 0.0
    assert metrics.objective_errors.idle >= 0.0
    assert metrics.objective_errors.final >= 0.0
    assert len(strategy) == 2

    initial_weight, initial_bias = base_learner.initial_params
    idle_weight, idle_bias = base_learner.idle_params
    query_features = encoder(batch[3])

    assert torch.count_nonzero(initial_weight) == 0
    assert torch.count_nonzero(initial_bias) == 0
    assert base_learner.predict_with(
        query_features,
        (idle_weight, idle_bias),
    ).shape == (2, 2, 2)
