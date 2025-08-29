import torch
import pytest
from paramancer.bloptim.step import NesterovMarkovParamStep
from paramancer.optim.step import NesterovStep


def test_nesterov_markov_param_step():
    def grad_map(vars):
        x, y = vars
        return x - a, b
    
    def grad_map_prm(vars, prms):
        x, y = vars
        a, b = prms
        return x - a, b
    
    Nx, Ny = 10, 5
    a, b = torch.randn(Nx), torch.rand(Ny)
    
    ss = torch.Tensor([0.1]).squeeze()
    
    step = NesterovStep(ss, grad_map)
    mp_step = NesterovMarkovParamStep(ss, grad_map_prm) # mp -> MarkovParam
    
    x_curr = torch.randn(Nx)
    y_curr = torch.randn(Ny)
    
    x_prev = x_curr.clone()
    y_prev = y_curr.clone()
    
    x_new, y_new = step(step(step((x_curr, y_curr))))
    
    def m_step(z):
        return mp_step(z, (a, b))
    (x_new_mp, y_new_mp), _ = m_step(m_step(m_step(
        ((x_curr, y_curr), (x_prev, y_prev))
    )))
    
    assert torch.allclose(x_new_mp, x_new)
    assert torch.allclose(y_new_mp, y_new)
    

