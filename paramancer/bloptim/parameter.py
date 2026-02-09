from functools import wraps
import torch
from typing import Union, Tuple, Callable, Any

from ..optim.variable import TensorLike, Variable


FlatParameter = TensorLike
TupleParameter = Tuple[TensorLike, ...]
ParameterType = Union[FlatParameter, TupleParameter]

class Parameter(Variable):
    def __init__(self, data: ParameterType):
        super().__init__(data, level="upper")
    
    @staticmethod
    def wrap(fn):
        def wrapped_fn(x: Variable, u: Parameter) -> Variable:
            return Variable(fn(x.data, u.data))
        return wrapped_fn
    
    @staticmethod
    def ensure_var_param_inputs(fn: Callable) -> Callable:
        @wraps(fn)
        def wrapper(self, x_curr, u_given=None, *args, **kwargs) -> Any:
            # Remember input types
            x_was_raw = not isinstance(x_curr, Variable)
            u_was_raw = (
                u_given is not None and not isinstance(u_given, Parameter)
            )

            # Wrap if needed
            if x_was_raw:
                x_curr = Variable(x_curr)
            if u_was_raw:
                u_given = Parameter(u_given)

            # Call the actual step method
            result = fn(self, x_curr, u_given, *args, **kwargs)

            # Unwrap result if the input was raw
            if x_was_raw or u_was_raw:
                return result.data
            return result
        return wrapper
