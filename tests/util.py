import torch
import torch.linalg as la


def quadratic(N: int, M: int=1000):
    A, b = torch.rand(M, N), torch.randn(M)
    Q, q = A.T @ A, A.T @ b
    lip = torch.linalg.norm(A.T @ A, ord=2)
    mu = torch.linalg.norm(Q, ord=-2)
    xm = la.solve(Q, q)
    return Q, q, xm, lip, mu


def quadratic_grad_sol(Q, q, xm_grad):
    Q = Q.detach().clone().requires_grad_()
    q = q.detach().clone().requires_grad_()
    xm = la.solve(Q, q)
    xm.backward(xm_grad)
    return Q.grad, q.grad


def lin_reg(M: int, N: int):
    A, b = torch.rand(M, N), torch.randn(M)
    Q, q = A.T @ A, A.T @ b
    lip = torch.linalg.norm(A.T @ A, ord=2)
    mu = torch.linalg.norm(Q, ord=-2)
    xm = la.solve(Q, q)
    return A, b, xm, lip, mu

def lin_reg_grad_sol(A, b, xm_grad):
    A = A.detach().clone().requires_grad_()
    b = b.detach().clone().requires_grad_()
    Q, q = A.T @ A, A.T @ b
    xm = la.solve(Q, q)
    xm.backward(xm_grad)
    return A.grad, b.grad


def grad_quad(Q: torch.Tensor, q: torch.Tensor):
    return lambda x: Q @ x - q


def grad_lin_reg(A: torch.Tensor, b: torch.Tensor):
    return grad_quad(A.T @ A, A.T @ b)


def prox_l1(t: torch.Tensor):
    return lambda x: x.sign() * torch.maximum(torch.zeros(1), x.abs() - t)

def prox_sq_l2(t: torch.Tensor):
    return lambda x: x / (1 + t)