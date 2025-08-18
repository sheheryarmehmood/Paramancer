import torch
import torch.linalg as la
import torch.autograd.functional as agF
from paramancer.bloptim.lower import OptimizerID
from paramancer.bloptim.param_step import GDParamStep, PolyakParamStep

def test_gd_with_imp_diff():
    M, N = 10, 8
    A, b = torch.rand(M, N), torch.randn(M)
    xm_grad = torch.randn(N)
    
    lip = la.matrix_norm(A.T @ A, ord=2)
    mu = la.matrix_norm(A.T @ A, ord=-2)
    
    def minimizer(A, b): return la.solve(A.T @ A, A.T @ b)
    xm_la, (A_grad, b_grad) = agF.vjp(minimizer, (A, b), xm_grad)
    
    ss = 2 / (lip + mu)
    
    def grad_map_prm(x, u):
        A, b = u
        return A.T @ (A @ x - b)
    
    param_step = GDParamStep(ss, grad_map_prm, tracking=True)
    x_init = torch.randn(N)
    A_given = A.detach().clone().requires_grad_()
    b_given = b.detach().clone().requires_grad_()
    xm_id = OptimizerID.apply(
        A_given, b_given, x_init, param_step, "default", 1e-8, None, 50000, 
        None, 2
    )
    xm_id.backward(xm_grad)
    
    assert torch.allclose(xm_la, xm_id, atol=1e-4)
    assert torch.allclose(A_grad, A_given.grad, atol=1e-3)
    assert torch.allclose(b_grad, b_given.grad, atol=1e-3)


def test_hb_with_imp_diff():
    M, N = 10, 8
    A, b = torch.rand(M, N), torch.randn(M)
    xm_grad = torch.randn(N)
    
    lip = la.matrix_norm(A.T @ A, ord=2)
    mu = la.matrix_norm(A.T @ A, ord=-2)
    
    def minimizer(A, b): return la.solve(A.T @ A, A.T @ b)
    xm_la, (A_grad, b_grad) = agF.vjp(minimizer, (A, b), xm_grad)
    
    sql, sqm = lip.sqrt(), mu.sqrt()
    ss = (2 / (sql + sqm)) ** 2
    mm = ((sql - sqm) / (sql + sqm)) ** 2
    
    def grad_map_prm(x, u):
        A, b = u
        return A.T @ (A @ x - b)
    
    param_step = PolyakParamStep(ss, mm, grad_map_prm, tracking=True)
    x_init = torch.randn(N)
    A_given = A.detach().clone().requires_grad_()
    b_given = b.detach().clone().requires_grad_()
    xm_id = OptimizerID.apply(
        A_given, b_given, x_init, param_step, "default", 1e-8, None, 10000, 
        None, 2
    )
    xm_id.backward(xm_grad)
    
    assert torch.allclose(xm_la, xm_id, atol=1e-4)
    assert torch.allclose(A_grad, A_given.grad, atol=1e-4)
    assert torch.allclose(b_grad, b_given.grad, atol=1e-4)