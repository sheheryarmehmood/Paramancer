import torch

from examples.meta_learning_v2.reference import ReferenceEvaluator


def test_reference_evaluator_returns_solution_and_outer_gradient():
    scale = torch.nn.Parameter(torch.tensor(1.0, dtype=torch.float64))
    support_inputs = torch.tensor(
        [[[-1.0, 0.0], [-0.8, 0.1], [0.8, -0.1], [1.0, 0.0]]],
        dtype=torch.float64,
    )
    query_inputs = torch.tensor(
        [[[-0.9, 0.0], [0.9, 0.0]]],
        dtype=torch.float64,
    )
    support_targets = torch.tensor([[0, 0, 1, 1]])
    query_targets = torch.tensor([[0, 1]])
    evaluator = ReferenceEvaluator(
        num_classes=2,
        reg_strength=0.1,
        iters=100,
        tol=1e-10,
    )

    result = evaluator(
        scale * support_inputs,
        support_targets,
        scale * query_inputs,
        query_targets,
        (scale,),
    )

    weight, bias = result.base_params
    assert result.loss.ndim == 0
    assert result.logits.shape == (1, 2, 2)
    assert weight.shape == (1, 2, 2)
    assert bias.shape == (1, 2)
    assert len(result.outer_gradient) == 1
    assert result.outer_gradient[0].shape == scale.shape
    assert torch.isfinite(result.outer_gradient[0])
