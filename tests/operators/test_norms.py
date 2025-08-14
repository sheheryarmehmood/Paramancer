import torch
import torch.linalg as la
import pytest
from paramancer.operators import norms

def test_vector_norms():
    p = torch.randn(5, 4, 3)
    
    norms_l1 = norms.l1(p, (0, 2))
    la_l1 = la.vector_norm(p, dim=1, ord=1)
    
    norms_l2 = norms.l2(p, 1)
    la_l2 = la.vector_norm(p, dim=(0, 2))
    
    norms_inf = norms.inf(p, -1)
    la_inf = la.vector_norm(p, ord=torch.inf)
    
    assert torch.allclose(norms_l1, la_l1, atol=1e-5)
    assert torch.allclose(norms_l2, la_l2, atol=1e-5)
    assert torch.allclose(norms_inf, la_inf, atol=1e-5)
    

    
    