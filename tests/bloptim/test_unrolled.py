import torch
import pytest

from paramancer.bloptim import unrolled
from paramancer.variable import FlatVar


def test_unrolled_gradient_descent_tracks_forward_state_and_autograd():
    def grad_map(x, u, shift):
        return x - (u + shift)

    shift = torch.randn(4)
    grad_out = torch.randn(4)
    model = unrolled.GradientDescent(
        torch.tensor(1.0),
        u=torch.randn(4),
        grad_map_prm=grad_map,
        iters=5,
        truncation=2,
    )

    with pytest.raises(RuntimeError, match="Forward initialization is unset"):
        _ = model.fwd_init

    z_init = torch.randn(4)
    z_root = model.u.detach() + shift
    xm = model(shift, z_init=z_init)

    assert isinstance(model.u, torch.nn.Parameter)
    assert isinstance(xm, FlatVar)
    assert torch.allclose(xm.data, z_root)
    assert torch.allclose(model.fwd_init.data, z_init)
    assert torch.allclose(model.fwd_sol.data, z_root)

    loss = (xm.data * grad_out).sum()
    loss.backward()

    assert torch.allclose(model.u.grad, grad_out)
