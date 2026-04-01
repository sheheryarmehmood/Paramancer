from __future__ import annotations
from functools import wraps
import torch

from .types import (
    InnerVarType, InnerVarLike, FlatInnerVarLike, ScalarLike
)
from .util import (
    vlatten, unvlatten, is_tensor, is_valid_variable
)

# TODO: I should create a new base class named `BaseVar` or `BaseOptVar`, `AbstractVar`, `AbstractOptVar` or something like that which implements all the arithmetic and utility methods and does not differentiate between if it is a inner or an outer variable. Then I can inherit both `InnerVar` and `ParameterBundle` from it. This will also allow me to implement the `flatten` and `unflatten` methods in the base class and use them in both `InnerVar` and `ParameterBundle`. This will also make the code cleaner and more modular. This base class should also have member methods like `norm`, `clone`, `requires_grad_`, `zeros_like`, `flatten` (will replace `vlatten` and `platten`) etc. and static methods like `wrap`, and `unflatten` (will replace `unvlatten` and `unplatten`) etc. which can be used by both `InnerVar` and `ParameterBundle`. The methods like `norm`, `clone`, `zeros_like`, `flatten` and `flatten` can be put `util.py` as well which will act on objects which are not of type of the base variable class.
class InnerVar:
    """
    Wrapper for optimization variables that can be:
      - torch.Tensor
      - tuple[torch.Tensor, ...]
      - (tuple[torch.Tensor, ...], tuple[torch.Tensor, ...])
    
    Provides arithmetic operations, clone, and norm 
    while preserving the input structure.
    """

    def __init__(self, data: InnerVarType):
        if not is_valid_variable(data):
            raise TypeError(
                "Unsupported InnerVarType. Expected Tensor, Tuple[Tensor,...],"
                " or Tuple[Tuple[Tensor,...], Tuple[Tensor,...]]."
            )
        self._data = data
    
    def zeros_like(self):
        flat, spec = vlatten(self.data)
        zero_flat = tuple(torch.zeros_like(x) for x in flat)
        zero = unvlatten(zero_flat, spec)
        return InnerVar(zero)
    
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
                return tuple(InnerVar._apply(fn, x1) for x1 in var1)
            else:
                return tuple(
                    InnerVar._apply(fn, x1, x2) for x1, x2 in zip(var1, var2)
                )
        else:
            raise TypeError(f"Unsupported type: {type(var1)}")

    # ------------------------
    # Arithmetic
    # ------------------------
    def __add__(self, other: InnerVar) -> InnerVar:
        return InnerVar(
            self._apply(lambda x1, x2: x1 + x2, self._data, other._data)
        )

    def __sub__(self, other: InnerVar) -> InnerVar:
        return InnerVar(
            self._apply(lambda x1, x2: x1 - x2, self._data, other._data)
        )
    
    def __neg__(self):
        return InnerVar(self._apply(lambda x: -x, self._data))

    def __mul__(self, scalar: ScalarLike) -> InnerVar:
        return InnerVar(self._apply(lambda x: x * scalar, self._data))

    __rmul__ = __mul__

    # ------------------------
    # Clone
    # ------------------------
    def clone(self) -> InnerVar:
        return InnerVar(self._apply(lambda x: x.clone(), self._data))
    
    def requires_grad_(self, requires_grad) -> InnerVar:
        return InnerVar(
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
    def data(self) -> InnerVarType:
        """Return raw data in original structure."""
        return self._data
    
    @property
    def primal(self) -> InnerVar:
        if isinstance(self._data, tuple) and len(self._data) == 2:
            return InnerVar(self._data[0])
        raise AttributeError("This InnerVar has no `primal`.")
    
    @property
    def dual(self) -> InnerVar:
        if isinstance(self._data, tuple) and len(self._data) == 2:
            return InnerVar(self._data[1])
        raise AttributeError("This InnerVar has no `dual`.")
    
    @property
    def current(self) -> InnerVar:
        if isinstance(self._data, tuple) and len(self._data) == 2:
            return InnerVar(self._data[0])
        raise AttributeError("This InnerVar has no `current`.")
    
    @property
    def previous(self) -> InnerVar:
        if isinstance(self._data, tuple) and len(self._data) == 2:
            return InnerVar(self._data[1])
        raise AttributeError("This InnerVar has no `previous`.")

    def __repr__(self):
        return f"InnerVar({self._data})"
    
    @staticmethod
    def wrap(fn):
        @wraps(fn)
        def wrapped_fn(x: InnerVar, *args, **kwargs) -> InnerVar:
            return InnerVar(fn(x.data, *args, **kwargs))
        return wrapped_fn
    
    @staticmethod
    def _from_pair(
        var1: FlatInnerVarLike,
        var2: FlatInnerVarLike
    ) -> InnerVarLike:
        if type(var1) is not type(var2):
            raise TypeError("The two elements should be of the same type")
        if isinstance(var1, InnerVar):
            return InnerVar((var1.data, var2.data))
        return var1, var2
    
    @staticmethod
    def from_momentum(
        curr: FlatInnerVarLike,
        prev: FlatInnerVarLike
    ) -> InnerVarLike:
        return InnerVar._from_pair(curr, prev)

    @staticmethod
    def from_pdhg(
        primal: FlatInnerVarLike,
        dual: FlatInnerVarLike
    ) -> InnerVarLike:
        return InnerVar._from_pair(primal, dual)


AlgoVar = InnerVar
Variable = InnerVar
