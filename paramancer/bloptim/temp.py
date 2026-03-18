import torch


def linreg(x: torch.Tensor, A: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    err = A @ x - b
    return 0.5 * err @ err

def grad_linreg(
    x: torch.Tensor, A: torch.Tensor, b: torch.Tensor
) -> torch.Tensor:
    return A.T @ (A @ x - b)

def vjp_grad_linreg(
    x: torch.Tensor, A: torch.Tensor, b: torch.Tensor, grad_out: torch.Tensor
) -> torch.Tensor:
    err = A @ x - b
    adj_out = A @ grad_out
    grad_x = A.T @ adj_out
    grad_A = torch.outer(err, grad_out) + torch.outer(adj_out, x)
    grad_b = -adj_out
    return grad_x, grad_A, grad_b

def jvp_grad_linreg(
    x: torch.Tensor, A: torch.Tensor, b: torch.Tensor,
    x_tan: torch.Tensor, A_tan: torch.Tensor, b_tan: torch.Tensor
) -> torch.Tensor:
    err = A @ x - b
    err_tan = A_tan @ x + A @ x_tan - b_tan
    out_tan = A.T @ err_tan + A_tan.T @ err
    return out_tan

M, N = 10, 5
A, b = torch.rand(M, N), torch.randn(M)
x = torch.randn(N)
A_tan, b_tan = torch.randn_like(A), torch.randn_like(b)
x_tan = torch.randn_like(x)
grad_out = torch.randn(N)

grad_x, grad_A, grad_b = vjp_grad_linreg(x, A, b, grad_out)

Arg = A.clone().requires_grad_()
brg = b.clone().requires_grad_()
xrg = x.clone().requires_grad_()
out = grad_linreg(xrg, Arg, brg)
out.backward(grad_out)

print("Comparing with backward")
assert torch.allclose(Arg.grad, grad_A)
assert torch.allclose(brg.grad, grad_b)
assert torch.allclose(xrg.grad, grad_x)
print("All close!")


_, vjpfunc = torch.func.vjp(grad_linreg, x, A, b)
vjpx, vjpA, vjpb = vjpfunc(grad_out)
print("Comparing with `func.vjp`")
assert torch.allclose(vjpA, grad_A)
assert torch.allclose(vjpb, grad_b)
assert torch.allclose(vjpx, grad_x)
print("All close!")


_, vjpx = torch.func.vjp(lambda x: grad_linreg(x, A, b), x)
vjpx_partial, = vjpx(grad_out)
print("Examining `func.vjp` with single argument")
assert torch.allclose(vjpx_partial, grad_x)
print("All close!")


out_tan = jvp_grad_linreg(x, A, b, x_tan, A_tan, b_tan)
_, jvp_out = torch.func.jvp(grad_linreg, (x, A, b), (x_tan, A_tan, b_tan))
print("Comparing with `func.jvp`")
assert torch.allclose(out_tan, jvp_out)
print("All close!")


out_tanx = jvp_grad_linreg(x, A, b, x_tan, 0*A, 0*b)
_, jvpx_out = torch.func.jvp(lambda x: grad_linreg(x, A, b), (x,), (x_tan,))
print("Examining `func.vjp` with single argument")
assert torch.allclose(out_tanx, jvpx_out)
print("All close!")