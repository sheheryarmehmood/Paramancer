from __future__ import annotations
from functools import wraps
from typing import Tuple, Any
import torch

from .typing import VariableType, VariableLike, ApplyType, SpecType


def _is_tensor(x: Any) -> bool:
    return isinstance(x, torch.Tensor)

def _is_tuple_of_tensors(x: Any) -> bool:
    return isinstance(x, tuple) and all(_is_tensor(t) for t in x)

def _is_nested_variable(x: Any) -> bool:
    # exactly: ( (tensors...), (tensors...) )
    return (
        isinstance(x, tuple)
        and len(x) == 2
        and _is_tuple_of_tensors(x[0])
        and _is_tuple_of_tensors(x[1])
    )

def _is_valid(x: Any) -> bool:
    return _is_tensor(x) or _is_tuple_of_tensors(x) or _is_nested_variable(x)

def flatten(data: VariableType) -> Tuple[ApplyType, SpecType]:
    """
    Flattens VariableType into a flat tuple of tensors plus a spec to reconstruct it.

    Spec encodings:
    - ("flat",)
    - ("tuple", n)
    - ("nested", n_left, n_right)
    """
    if _is_tensor(data):
        return (data,), ("flat",)
    
    if _is_tuple_of_tensors(data):
        return data, ("tuple", len(data))
    
    if _is_nested_variable(data):
        left, right = data
        flat = (*left, *right)
        return flat, ("nested", len(left), len(right))

def unflatten(flat: ApplyType, spec: SpecType) -> VariableType:
    """
    Reconstructs VariableType from a flat tuple of tensors and a spec produced by `flatten`.
    """
    if not _is_tuple_of_tensors(flat):
        raise TypeError("`flat` must be a tuple of torch.Tensor.")
    
    if not isinstance(spec, tuple) or len(spec) < 1:
        raise TypeError("`spec` must be a non-empty tuple.")
    
    tag = spec[0]
    if tag == "flat":
        if len(flat) != 1:
            raise ValueError(
                f"Spec {spec} expects exactly 1 tensor, got {len(flat)}."
            )
        return flat[0]
    
    if tag == "tuple":
        if len(spec) != 2 or not isinstance(spec[1], int):
            raise TypeError("Spec ('tuple', n) must have an int n.")
        n = spec[1]
        if len(flat) != n:
            raise ValueError(
                f"Spec {spec} expects {n} tensors, got {len(flat)}."
            )
        return tuple(flat)
    
    if tag == "nested":
        if (
            len(spec) != 3 or not isinstance(spec[1], int) or 
            not isinstance(spec[2], int)
        ):
            raise TypeError(
                "Spec ('nested', n_left, n_right) must have two ints."
            )
        nL, nR = spec[1], spec[2]
        if len(flat) != nL + nR:
            raise ValueError(
                f"Spec {spec} expects {nL+nR} tensors, got {len(flat)}."
            )
        left = tuple(flat[:nL])
        right = tuple(flat[nL:nL + nR])
        return (left, right)


class Variable:
    """
    Wrapper for optimization variables that can be:
      - torch.Tensor
      - tuple[torch.Tensor, ...]
      - (tuple[torch.Tensor, ...], tuple[torch.Tensor, ...])
    
    Provides arithmetic operations, clone, and norm 
    while preserving the input structure.
    """

    def __init__(self, data: VariableType, level: str = "lower"):
        if not _is_valid(data):
            raise TypeError(
                "Unsupported VariableType. Expected Tensor, Tuple[Tensor,...],"
                " or Tuple[Tuple[Tensor,...], Tuple[Tensor,...]]."
            )
        self._data = data
        self._level = level
    
    def flatten(self):
        return flatten(self.data)
    
    @staticmethod
    def unflatten(flat: ApplyType, spec: SpecType) -> VariableType:
        return unflatten(flat, spec)
    
    
    
    # ------------------------
    # Utility: recursive apply
    # ------------------------
    @staticmethod
    def _apply(fn, var1, var2=None):
        """Apply fn recursively on a (and b if provided)."""
        if _is_tensor(var1):
            return fn(var1, var2) if var2 is not None else fn(var1)
        elif isinstance(var1, tuple):
            if var2 is None:
                return tuple(Variable._apply(fn, x1) for x1 in var1)
            else:
                return tuple(
                    Variable._apply(fn, x1, x2) for x1, x2 in zip(var1, var2)
                )
        else:
            raise TypeError(f"Unsupported type: {type(var1)}")

    # ------------------------
    # Arithmetic
    # ------------------------
    def __add__(self, other: Variable) -> Variable:
        return Variable(
            self._apply(lambda x1, x2: x1 + x2, self._data, other._data)
        )

    def __sub__(self, other: Variable) -> Variable:
        return Variable(
            self._apply(lambda x1, x2: x1 - x2, self._data, other._data)
        )
    
    def __neg__(self):
        return Variable(self._apply(lambda x: -x, self._data))

    def __mul__(self, scalar: float) -> Variable:
        return Variable(self._apply(lambda x: x * scalar, self._data))

    __rmul__ = __mul__

    # ------------------------
    # Clone
    # ------------------------
    def clone(self) -> Variable:
        return Variable(self._apply(lambda x: x.clone(), self._data))
    
    def requires_grad_(self, requires_grad) -> Variable:
        return Variable(
            self._apply(lambda x: x.requires_grad_(requires_grad), self._data)
        )

    # ------------------------
    # Norm (flatten recursively)
    # ------------------------
    def norm(self) -> torch.Tensor:
        norms = []

        def collect(x):
            if isinstance(x, torch.Tensor):
                norms.append(torch.norm(x))

        self._apply(collect, self._data)
        return torch.norm(torch.stack(norms))

    # ------------------------
    # Structure-preserving return
    # ------------------------
    @property
    def data(self) -> VariableType:
        """Return raw data in original structure."""
        return self._data
    
    @property
    def primal(self) -> Variable:
        if isinstance(self._data, tuple) and len(self._data) == 2:
            return Variable(self._data[0])
        raise AttributeError("This Variable has no `primal`.")
    
    @property
    def dual(self) -> Variable:
        if isinstance(self._data, tuple) and len(self._data) == 2:
            return Variable(self._data[1])
        raise AttributeError("This Variable has no `dual`.")
    
    @property
    def current(self) -> Variable:
        if isinstance(self._data, tuple) and len(self._data) == 2:
            return Variable(self._data[0])
        raise AttributeError("This Variable has no `current`.")
    
    @property
    def previous(self) -> Variable:
        if isinstance(self._data, tuple) and len(self._data) == 2:
            return Variable(self._data[1])
        raise AttributeError("This Variable has no `previous`.")

    def __repr__(self):
        return f"Variable({self._data})"
    
    @staticmethod
    def wrap(fn):
        def wrapped_fn(x: Variable) -> Variable:
            return Variable(fn(x.data))
        return wrapped_fn
    
    @staticmethod
    def ensure_var_input(fn):
        @wraps(fn)
        def wrapper(self, x_curr, *args, **kwargs):
            self._input_is_variable = isinstance(x_curr, Variable)
            if not self._input_is_variable:
                x_curr = Variable(x_curr)
            out = fn(self, x_curr, *args, **kwargs)
            return out if self._input_is_variable else out.data
        return wrapper
    
    @staticmethod
    def _from_pair(
        var1: VariableLike,
        var2: VariableLike
    ) -> Variable:
        v1 = var1.data if isinstance(var1, Variable) else var1
        v2 = var2.data if isinstance(var2, Variable) else var2
        return Variable((v1, v2))

    @staticmethod
    def from_momentum(
        curr: VariableLike,
        prev: VariableLike
    ) -> Variable:
        return Variable._from_pair(curr, prev)

    @staticmethod
    def from_pdhg(
        primal: VariableLike,
        dual: VariableLike
    ) -> Variable:
        return Variable._from_pair(primal, dual)


