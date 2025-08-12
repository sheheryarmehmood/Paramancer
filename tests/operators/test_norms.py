import torch
import torch.linalg as la
import pytest
from paramancer.operators import norms

def test_vector_norms():
    p = torch.randn(5, 4, 3)
    
    assert all([
        torch.allclose(norms.l1(p, (0, 2)), la.vector_norm(p, dim=1, ord=1)),
        torch.allclose(norms.l2(p, 1), la.vector_norm(p, dim=(0, 2))),
        torch.allclose(norms.inf(p, -1), la.vector_norm(p, ord=torch.inf))
    ])

def test_matrix_norms():
    p = torch.randn(10, 12)

    
    