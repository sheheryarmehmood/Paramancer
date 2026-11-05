import torch

from paramancer.operators.prox import l2_ball


def test_l2_ball_preserves_device_and_dtype():
    point = torch.tensor([3.0, 4.0], dtype=torch.float64)
    radius = point.new_tensor(2.0)

    projected = l2_ball(point, radius)

    assert projected.device == point.device
    assert projected.dtype == point.dtype
    assert torch.allclose(projected, point * (2.0 / 5.0))
