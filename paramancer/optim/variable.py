from __future__ import annotations
from typing import Union, Tuple
import torch
from paramancer.operators.norms import l2

# Types allowed for OptVar construction
TensorLike = torch.Tensor
FlatVariable = TensorLike
TupleVariable = Tuple[TensorLike, ...]
NestedVariable = Tuple[TupleVariable, TupleVariable]
VariableType = Union[FlatVariable, TupleVariable, NestedVariable]


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
        self._data = data

    # ------------------------
    # Utility: recursive apply
    # ------------------------
    @staticmethod
    def _apply(fn, var1, var2=None):
        """Apply fn recursively on a (and b if provided)."""
        if isinstance(var1, FlatVariable):
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

    def __mul__(self, scalar: float) -> Variable:
        return Variable(self._apply(lambda x: x * scalar, self._data))

    __rmul__ = __mul__

    # ------------------------
    # Clone
    # ------------------------
    def clone(self) -> Variable:
        return Variable(self._apply(lambda x: x.clone(), self._data))

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
    def primal(self) -> VariableType:
        if isinstance(self._data, FlatVariable) or len(self._data) > 2:
            raise RuntimeError(f"Does work with variables of this type")
        return self._data[0]
    
    @property
    def dual(self) -> VariableType:
        if isinstance(self._data, FlatVariable) or len(self._data) > 2:
            raise RuntimeError(f"Does work with variables of this type")
        return self._data[1]
    
    @property
    def current(self) -> VariableType:
        if isinstance(self._data, FlatVariable) or len(self._data) > 2:
            raise RuntimeError(f"Does work with variables of this type")
        return self._data[0]
    
    @property
    def previous(self) -> VariableType:
        if isinstance(self._data, FlatVariable) or len(self._data) > 2:
            raise RuntimeError(f"Does work with variables of this type")
        return self._data[1]

    def __repr__(self):
        return f"Variable({self._data})"
