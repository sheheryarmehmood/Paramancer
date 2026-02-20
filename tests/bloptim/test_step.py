import torch
import pytest

from paramancer.bloptim.step import NesterovParamMarkovStep
from paramancer.optim.step import NesterovStep


def test_nesterov_param_markov_step():
    def grad_map(x):
        x1, x2 = x
        return x1 - u1, u2 * x2
    
    def grad_map_prm(vars, u):
        x1, x2 = vars
        u1, u2 = u
        return x1 - u1, u2 * x2
    
    N1, N2 = 10, 5
    u1, u2 = torch.randn(N1), torch.rand(N2)
    iters = 5
    ss = torch.Tensor([0.1]).squeeze()
    
    step = NesterovStep(ss, grad_map=grad_map)
    pm_step = NesterovParamMarkovStep(ss, grad_map_prm=grad_map_prm)
    
    x1, x2 = torch.randn(N1), torch.randn(N2)
    
    x_curr = x1, x2
    for _ in range(iters):
        x_curr = step(x_curr)
    x1_new, x2_new = x_curr
    
    x_prev = x1.clone(), x2.clone() 
    z_curr = x_curr, x_prev
    u = u1, u2
    for _ in range(iters):
        z_curr = pm_step(z_curr, u)
    x1_new_pm, x2_new_pm = z_curr[0]
    
    assert torch.allclose(x1_new_pm, x1_new)
    assert torch.allclose(x2_new_pm, x2_new)
    

