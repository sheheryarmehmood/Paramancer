import torch
import torch.linalg as la
from paramancer.bloptim import neumann_series


def test_neumann_nxn():
    M, N, P, K = 100, 5, 3, 10000
    A = torch.rand(M, N)
    Q = A.T @ A
    matrix = 0.9 * Q / la.matrix_norm(Q, ord=2)
    vector = torch.randn(N, P).squeeze()
    
    true_sol = la.solve(torch.eye(N) - matrix, vector)
    neu_sol = neumann_series(lambda x: matrix @ x, vector, iters=K)
    
    assert torch.allclose(true_sol, neu_sol, atol=1e-5)


