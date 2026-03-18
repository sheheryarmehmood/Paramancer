from __future__ import annotations
from functools import wraps
import torch

from .types import (
    VariableType, VariableLike, BaseVariableLike,
    ScalarLike, FlattendType, VSpecType
)
from .util import (
    vlatten, unvlatten, is_tensor, is_valid_variable
)


class Variable:
    """
    Wrapper for optimization variables that can be:
      - torch.Tensor
      - tuple[torch.Tensor, ...]
      - (tuple[torch.Tensor, ...], tuple[torch.Tensor, ...])
    
    Provides arithmetic operations, clone, and norm 
    while preserving the input structure.
    """

    def __init__(self, data: VariableType):
        if not is_valid_variable(data):
            raise TypeError(
                "Unsupported VariableType. Expected Tensor, Tuple[Tensor,...],"
                " or Tuple[Tuple[Tensor,...], Tuple[Tensor,...]]."
            )
        self._data = data
    
    def flatten(self) -> tuple[FlattendType, VSpecType]:
        return vlatten(self.data)
    
    @staticmethod
    def unflatten(flat: FlattendType, spec: VSpecType) -> VariableType:
        return unvlatten(flat, spec)
    
    # ------------------------
    # Utility: recursive apply
    # ------------------------
    @staticmethod
    def _apply(fn, var1, var2=None):
        """Apply fn recursively on a (and b if provided)."""
        if is_tensor(var1):
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

    def __mul__(self, scalar: ScalarLike) -> Variable:
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
        def wrapped_fn(x: Variable, *args, **kwargs) -> Variable:
            return Variable(fn(x.data, *args, **kwargs))
        return wrapped_fn
    
    @staticmethod
    def _from_pair(
        var1: BaseVariableLike,
        var2: BaseVariableLike
    ) -> VariableLike:
        if type(var1) is not type(var2):
            raise TypeError("The two elements should be of the same type")
        if isinstance(var1, Variable):
            return Variable((var1.data, var2.data))
        return var1, var2
    
    @staticmethod
    def from_momentum(
        curr: BaseVariableLike,
        prev: BaseVariableLike
    ) -> VariableLike:
        return Variable._from_pair(curr, prev)

    @staticmethod
    def from_pdhg(
        primal: BaseVariableLike,
        dual: BaseVariableLike
    ) -> VariableLike:
        return Variable._from_pair(primal, dual)
