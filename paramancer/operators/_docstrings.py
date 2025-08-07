# %% Docstrings for prox.py

def extra_params_l2(projection: bool) -> str:
    sor = "rad" if projection else "scal"
    l2 = "Euclidean norm"
    return (
        f"    eps (float, optional): small value to ensure that {l2} of "
        f"output does not exceed `{sor}`. Defaults to 1e-8.\n"
    )

def extra_params_group_l2(projection: bool) -> str:
    l2 = "Euclidean norm"
    return (
        f"    dim (int | tuple[int], optional): dims along which {l2} is "
        f"computed. Defaults to -1.\n"
        f"    keepdim (bool, optional): whether to keep the reduced "
        f"dimension or not. Defaults to False.\n"
        f"{extra_params_l2(projection)}"
    )

def which_matrix_norm(norm_type: str) -> str:
    if "spectral" in norm_type:
        return "spectral"
    
    if "nuclear" in norm_type:
        return "nuclear"
    
    return ""

def is_Euclidean_norm(norm_type: str) -> bool:
    return "Euclidean" == norm_type

def is_matrix_norm(norm_type: str) -> bool:
    return "spectral" == norm_type or "nuclear" == norm_type

def has_Euclidean_norm(norm_type: str) -> bool:
    return "Euclidean" in norm_type
    
def has_matrix_norm(norm_type: str) -> bool:
    return "spectral" in norm_type or "nuclear" in norm_type

def get_input_desc(norm_type: str, input_name: str) -> str:
    if not is_matrix_norm(norm_type):
        return ""
    
    return f"{input_name}.ndim must be 2."

def get_extra_params(norm_type: str, projection: bool) -> str:
    if not has_Euclidean_norm(norm_type):
        return ""
    
    if is_Euclidean_norm(norm_type):
        return extra_params_l2(projection)
    
    return extra_params_group_l2(projection)

def get_sor_shape(norm_type: str) -> str:
    if not has_Euclidean_norm(norm_type):
        return ""
    
    if not has_matrix_norm(norm_type):
        return "x.shape"
    
    return "(min(X.shape[-1], X.shape[-2]),)"

def get_more_method_desc(norm_type: str) -> str:
    if not is_matrix_norm(norm_type) and has_matrix_norm(norm_type):
        return (
            f"The {which_matrix_norm(norm_type)} norm is computed along the "
            f"last two dimensions."
        )
    return ""


def prox_doc(
    norm_type: str,
    projection: bool=False
) -> str:
    """
    Generates a standard docstring for `prox` or `proj` functions.

    Args:
        norm_type (str): Name of the norm used in the method.
        input_name (str): Name of the input tensor (e.g., "x", "X").
        projection (bool): Whether the method computes projection or not.

    Returns:
        str: A formatted docstring with argument and return descriptions.
    """
    input_name = "p"
    if projection:
        sor = "rad"
        method_desc = (
            f"Project a tensor onto {norm_type} norm ball.\n\n"
            f"This computes the projection of `{input_name}` onto the norm "
            f"ball defined by the {norm_type} norm and radius `{sor}`."
        )
        sor_desc = "Radius of the norm ball."
        return_desc = "Projected tensor"
    else:
        sor = "scal"
        method_desc = (
            f"Computes the proximal mapping of {norm_type} norm.\n\n"
            f"This evaluates the prox operator of {norm_type} scaled by "
            f"factor `{sor}` at {input_name}."
        )
        sor_desc = "Scale factor of the norm."
        return_desc = "The computed prox"
    sor_note = f"`{sor}` must be a scalar"
    input_desc = get_input_desc(norm_type, input_name)
    more_method_desc = get_more_method_desc(norm_type)
    sor_shape = get_sor_shape(norm_type)
    extra_params = get_extra_params(norm_type, projection)
    if sor_shape:
        sor_note = f"Either {sor_note} or it must match shape: {sor_shape}"
    return (
        f"{method_desc} {more_method_desc}\n\n"
        f"Args:\n"
        f"    {input_name} (torch.Tensor): The given point. {input_desc}\n"
        f"    {sor} (torch.Tensor): {sor_desc} {sor_note}.\n"
        f"{extra_params}\n"
        f"Returns:\n"
        f"    torch.Tensor: {return_desc} with same shape as `{input_name}`."
    )


# %% Docstrings for norms.py

def dim_param():
    return (
        "    dim (int | tuple[int]): dimension along which the norm "
        "should be computed.\n"
    )

def keep_dim_param():
    return (
        "    keepdim (bool, optional): After the reduction, the p.ndim "
        "matches the number of dimensions in the output tensor, if "
        "keepdim is true. The default is False. Defaults to False."
    )

def batch_param(quant_type):
    return (
        f"    batch (int | tuple[int], optional): Dimension along which the "
        f"{quant_type} should not be computed. When -1, the "
        f"operation is performed along all dimensions. Defaults to -1.\n"
    )

def method_desc_for_inner_norms(norm_type):
    more_method_desc = (
        f"Useful when computing the group norms where the inner norm is "
        f"the {norm_type} norm"
    )
    if norm_type != "Euclidean":
        more_method_desc = more_method_desc + (
            "which is always computed along the last two dimensions."
        )
    return more_method_desc

def norm_doc(
    quant_name: str,
    squared: bool=False,
    group_l2: bool=False,
    inner: bool=False
) -> str:
    more_method_desc = ""
    if "product" not in quant_name:
        quant_type = "squared norm" if squared else "norm"
        quant_name = quant_name + " norm"
        input_tensor_with_desc = "    p (torch.Tensor) The given input.\n"
    else:
        quant_type = "inner product"
        input_tensor_with_desc = (
            "    p (torch.Tensor) The given input # 1.\n"
            "    q (torch.Tensor) The given input # 2. "
            "Its shape must match that of `p`.\n"
        )
    extra_params = ""
    if group_l2:
        extra_params = extra_params + dim_param()
    if inner:
        extra_params = extra_params + keep_dim_param()
        more_method_desc = method_desc_for_inner_norms(quant_name)
    else:
        extra_params = extra_params + batch_param(quant_type)
    return (
        f"Computes the {quant_name} of a given input. {more_method_desc}\n\n"
        f"Args:\n"
        f"{input_tensor_with_desc}"
        f"{extra_params}"
        f"Returns:\n"
        f"torch.Tensor: The computed {quant_type}."
    )

