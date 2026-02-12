import torch
import pytest

from paramancer.variable import Variable
from paramancer.operators.imaging import conv_op


def test_variable_data():
    x = torch.randn(10)
    x_curr_var = Variable(x)
    assert torch.allclose(x_curr_var.data, x)       # Single tensor.
    
    a = torch.randn(100)
    b = torch.rand(20)
    c = torch.randn(5)
    x_curr_var = Variable((a, b, c))                # Tuple of tensors
    assert x_curr_var.data[0] is a
    assert x_curr_var.data[1] is b
    assert x_curr_var.data[2] is c
    
    ker_d1 = torch.rand(2, 1, 3, 3)             # kernel
    imgs = torch.rand(10, 3, 64, 64)            # primal variable
    imgs_d1 = conv_op(imgs, ker_d1)             # dual variable
    # vvvvv primal and dual tensors vvvvv
    x_curr_var = Variable.from_pdhg(Variable(imgs), Variable(imgs_d1))
    assert x_curr_var.primal.data is imgs
    assert x_curr_var.dual.data is imgs_d1
    
    ker_d1 = torch.rand(2, 1, 3, 3)             # kernel "1st derivative"
    ker_d2 = torch.rand(4, 1, 3, 3)             # kernel "2nd derivative"
    ker_d3 = torch.rand(8, 1, 3, 3)             # kernel "2nd derivative"
    
    imgs = torch.rand(10, 3, 64, 64)            # primal variable 1
    imgs_d1 = conv_op(imgs, ker_d1)             # primal variable 2
    imgs_d2 = conv_op(imgs_d1, ker_d2)             # primal variable 3
    
    edgs = torch.rand(10, 6, 64, 64)            # dual variable 1
    edgs_d1 = conv_op(edgs, ker_d2)             # dual variable 2
    edgs_d2 = conv_op(edgs_d1, ker_d3)          # dual variable 2
    
    # when `Variable` objects are passed to Variable.from_pdhg.
    primal = (imgs, imgs_d1, imgs_d2)
    dual = (edgs, edgs_d1, edgs_d2)
    x_curr_var = Variable.from_pdhg(Variable(primal), Variable(dual))
    assert x_curr_var.primal.data[0] is imgs
    assert x_curr_var.primal.data[1] is imgs_d1
    assert x_curr_var.primal.data[2] is imgs_d2
    assert x_curr_var.dual.data[0] is edgs
    assert x_curr_var.dual.data[1] is edgs_d1
    assert x_curr_var.dual.data[2] is edgs_d2
    
    # when `VariableType` objects are passed to Variable.from_pdhg.
    x_curr = Variable.from_pdhg(primal, dual)
    assert x_curr[0] is primal
    assert x_curr[1] is dual
    

def test_prox_grad_tensor_tuples():
    def grad_map(xs): return xs[0] ** 2, xs[1] + xs[2], xs[1] - xs[2]
    def prox_map(xs): return xs[0], 2 * xs[1], xs[2] / 2
    ss = torch.tensor(0.01)
    
    xs = (torch.randn(10), torch.rand(5), torch.randn(5))
    gd_xs = grad_map(xs)
    pgd_xs = prox_map(tuple(x - ss*gd_x for x, gd_x in zip(xs, gd_xs)))
    
    xs_var = Variable(xs)
    grad_map_var = Variable.wrap(grad_map)
    prox_map_var = Variable.wrap(prox_map)
    gd_xvs = grad_map_var(xs_var)
    pgd_xvs = prox_map_var(xs_var - ss * gd_xvs)
    
    assert torch.allclose(pgd_xvs.data[0], pgd_xs[0])
    assert torch.allclose(pgd_xvs.data[1], pgd_xs[1])
    assert torch.allclose(pgd_xvs.data[2], pgd_xs[2])



