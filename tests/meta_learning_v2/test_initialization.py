import torch

from examples.meta_learning_v2.warm_start import ColdStart, ForwardWarmStart


def test_forward_warm_start_cache_reorders_episode_states():
    strategy = ForwardWarmStart()
    episode_ids = torch.tensor([4, 2])
    weight = torch.arange(12.0).reshape(2, 2, 3)
    bias = torch.arange(4.0).reshape(2, 2)

    strategy.update_forward(episode_ids, (weight, bias))
    warm_start, adjoint_init = strategy.get(torch.tensor([2, 4]))

    assert warm_start is not None
    assert adjoint_init is None
    warm_weight, warm_bias = warm_start
    assert torch.equal(warm_weight, weight.flip(0))
    assert torch.equal(warm_bias, bias.flip(0))
    assert warm_weight.device.type == "cpu"
    assert warm_bias.device.type == "cpu"


def test_forward_warm_start_cache_returns_none_for_unseen_episode():
    strategy = ForwardWarmStart()
    strategy.update_forward(
        [0],
        (torch.zeros(1, 2, 3), torch.zeros(1, 2)),
    )

    assert strategy.get([0, 1]) == (None, None)

    strategy.clear()
    assert len(strategy) == 0


def test_cold_start_is_stateless():
    strategy = ColdStart()
    params = (torch.zeros(1, 2, 3), torch.zeros(1, 2))

    strategy.update_forward([0], params)
    strategy.update_adjoint([0], params)

    assert strategy.get([0]) == (None, None)
