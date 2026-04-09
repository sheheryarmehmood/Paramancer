import torch

from paramancer.operators.imaging import conv_op
from paramancer.variable import FlatVar, PairVar


def test_variable_data():
    x = torch.randn(10)
    x_curr_var = FlatVar(x)
    assert torch.allclose(x_curr_var.data, x)

    a = torch.randn(100)
    b = torch.rand(20)
    c = torch.randn(5)
    x_curr_var = FlatVar((a, b, c))
    assert x_curr_var.data[0] is a
    assert x_curr_var.data[1] is b
    assert x_curr_var.data[2] is c

    ker_d1 = torch.rand(2, 1, 3, 3)
    imgs = torch.rand(10, 3, 64, 64)
    imgs_d1 = conv_op(imgs, ker_d1)
    x_curr_var = PairVar(FlatVar(imgs), FlatVar(imgs_d1))
    assert x_curr_var.first.data is imgs
    assert x_curr_var.second.data is imgs_d1

    ker_d1 = torch.rand(2, 1, 3, 3)
    ker_d2 = torch.rand(4, 1, 3, 3)
    ker_d3 = torch.rand(8, 1, 3, 3)

    imgs = torch.rand(10, 3, 64, 64)
    imgs_d1 = conv_op(imgs, ker_d1)
    imgs_d2 = conv_op(imgs_d1, ker_d2)

    edgs = torch.rand(10, 6, 64, 64)
    edgs_d1 = conv_op(edgs, ker_d2)
    edgs_d2 = conv_op(edgs_d1, ker_d3)

    primal = (imgs, imgs_d1, imgs_d2)
    dual = (edgs, edgs_d1, edgs_d2)
    x_curr_var = PairVar(FlatVar(primal), FlatVar(dual))
    assert x_curr_var.first.data[0] is imgs
    assert x_curr_var.first.data[1] is imgs_d1
    assert x_curr_var.first.data[2] is imgs_d2
    assert x_curr_var.second.data[0] is edgs
    assert x_curr_var.second.data[1] is edgs_d1
    assert x_curr_var.second.data[2] is edgs_d2

    x_curr = PairVar(primal, dual)
    assert x_curr[0].data is primal
    assert x_curr[1].data is dual


def test_prox_grad_tensor_tuples():
    def grad_map(xs):
        return xs[0] ** 2, xs[1] + xs[2], xs[1] - xs[2]

    def prox_map(xs):
        return xs[0], 2 * xs[1], xs[2] / 2

    ss = torch.tensor(0.01)

    xs = (torch.randn(10), torch.rand(5), torch.randn(5))
    gd_xs = grad_map(xs)
    pgd_xs = prox_map(tuple(x - ss * gd_x for x, gd_x in zip(xs, gd_xs)))

    xs_var = FlatVar(xs)
    gd_xvs = FlatVar(grad_map(xs_var.data))
    pgd_xvs = FlatVar(prox_map((xs_var - ss * gd_xvs).data))

    assert torch.allclose(pgd_xvs.data[0], pgd_xs[0])
    assert torch.allclose(pgd_xvs.data[1], pgd_xs[1])
    assert torch.allclose(pgd_xvs.data[2], pgd_xs[2])
