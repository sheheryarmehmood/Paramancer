from __future__ import annotations

from collections.abc import Callable

import torch

from ..variable.flat import FlatVar
from ..variable.types import (
    FlatRawVarType,
    FlatVarLike,
    PGradMapType,
    PSmoothObjType,
    ParamListType,
    is_parameter_type,
)
from ..variable.util import flatten_flat_raw, is_flat_var, unflatten_flat_raw


def _set_req_grad(vars: tuple[torch.Tensor, ...], rgs: bool | tuple[bool, ...]):
    if isinstance(rgs, bool):
        rgs = (rgs,) * len(vars)
    for var, rg in zip(vars, rgs):
        var.requires_grad_(rg)


def gradient(smooth: PSmoothObjType) -> PGradMapType:
    def grad_s(x: FlatVarLike, *args, **kwargs):
        outer_grad_enabled = torch.is_grad_enabled()
        x_was_var = is_flat_var(x)
        x_data = x.data if x_was_var else x

        x_flat, x_spec = flatten_flat_raw(x_data)

        u_flat: tuple[torch.Tensor, ...] = ()
        if len(args) > 0 and is_parameter_type(args[0]):
            u_flat = (
                tuple(args[0]) if isinstance(args[0], (ParamListType, tuple)) else (args[0],)
            )

        rgs = tuple(x_f.requires_grad for x_f in x_flat)
        _set_req_grad(x_flat, True)

        create_graph = outer_grad_enabled and (
            any(rg for rg in rgs) or any(u_f.requires_grad for u_f in u_flat)
        )

        with torch.enable_grad():
            x_unflat = unflatten_flat_raw(x_flat, x_spec)
            x_in = FlatVar(x_unflat) if x_was_var else x_unflat
            out = smooth(x_in, *args, **kwargs).sum()

        gd = torch.autograd.grad(
            out, x_flat, create_graph=create_graph, allow_unused=True
        )
        gd = tuple(
            torch.zeros_like(x_f) if g is None else g
            for x_f, g in zip(x_flat, gd)
        )

        _set_req_grad(x_flat, rgs)

        out_grad = unflatten_flat_raw(gd, x_spec)
        return FlatVar(out_grad) if x_was_var else out_grad

    return grad_s


def _gradient(
    smooth: Callable[..., torch.Tensor], *dargs: int
) -> Callable[..., FlatRawVarType]:
    def grad_s(*args):
        inps = [args[i] for i in dargs] if dargs else [args[0]]
        rgs = [inp.requires_grad for inp in inps]
        create_graph = any(getattr(arg, "requires_grad", False) for arg in args)

        for inp, rg in zip(inps, rgs):
            if not rg:
                inp.requires_grad_(True)
        with torch.enable_grad():
            out = smooth(*args).sum()
        gd = torch.autograd.grad(
            out, inps, create_graph=create_graph, allow_unused=True
        )
        if len(gd) == 1:
            gd = gd[0]

        for inp, rg in zip(inps, rgs):
            if not rg:
                inp.requires_grad_(False)
        return gd

    return grad_s
