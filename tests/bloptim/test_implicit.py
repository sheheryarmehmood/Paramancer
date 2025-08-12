import torch
import torch.autograd.functional as agF
import torch.linalg as la

from paramancer.bloptim.param_step import GDParamStep
from paramancer.bloptim import ImplicitDifferentiation


def test_implicit():
    def grad_fn_prm(x, u):
        A, b = u
        return A.T @ (A @ x - b)

    def minimizer(u):
        A, b = u
        return la.solve(A.T @ A, A.T @ b)

    M, N, K = 1000, 10, 1000
    A = torch.rand(M, N)
    b = torch.randn(M)
    u_given = A, b
    xmin = la.solve(A.T @ A, A.T @ b)
    xmin_grad = torch.randn(xmin.shape)

    u_grad_an = agF.vjp(minimizer, u_given, xmin_grad)[1]

    lip = la.matrix_norm(A.T @ A, ord=2)
    gd_step = GDParamStep(stepsize=1/lip, grad_map_prm=grad_fn_prm)

    imp_diff = ImplicitDifferentiation(gd_step, iters=K)
    u_grad_id = imp_diff(xmin, u_given, xmin_grad)


    assert torch.allclose(u_grad_id, u_grad_an)

