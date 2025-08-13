import torch
import torch.linalg as la
import pytest
from paramancer.operators import norms

def test_vector_norms():
    p = torch.randn(5, 4, 3)
    
    assert torch.allclose(
        norms.l1(p, (0, 2)), la.vector_norm(p, dim=1, ord=1), atol=1e-5
    )
    assert torch.allclose(
        norms.l2(p, 1), la.vector_norm(p, dim=(0, 2)), atol=1e-5
    )
    assert torch.allclose(
        norms.inf(p, -1), la.vector_norm(p, ord=torch.inf), atol=1e-5
    )
    

    
    