# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
import torch
import typing as t


def nanmax(tensor, dim=None, keepdim=False):
    """
    This function takes a tensor and along with a dimension 'dim', and a boolean flag 'keepdim'.
    It returns another tensor where for each 'dim' the values are replaced with maximum value in that dimension.
    If 'tensor' has any NaNs, it will return infinity instead of NaN.

    Parameters:
        tensor (Tensor): Input tensor from which to compute max
        dim (int or None): Dimension along which the maximum is computed
        keepdim (bool): Determines whether the output tensors have 'dim' retained or not

    Returns:
        Tensor: The resultant tensor after applying nanmax
    """
    min_value = torch.finfo(tensor.dtype).min
    output = tensor.nan_to_num(min_value).max(dim=dim, keepdim=keepdim)
    return output.values


class TorchPoolingOP(torch.nn.Module):
    """
    A module that applies a pooling operation on input tensor along given dimension.

    Parameters:
        op_name (str): Name of the pooling function to be used, from BASE_POOLING_FUNCTIONS.
        dim (int): Dimension along which the operation is performed.

    Attributes:
        name (str): The name of the pooling function being applied.
        dim (int): The dimension along which the operation is performed.
        op (function): The actual pooling function to be used.

    """

    TORCH_POOLING_FUNCTIONS = {
        "max": nanmax,
        "all": lambda x, *args, **kwargs: x,
    }

    def __init__(self, op_name: str, dim: int):
        super().__init__()
        self.name = op_name
        self.dim = dim
        self.op = self.TORCH_POOLING_FUNCTIONS[self.name]

    def forward(
        self, tensor: torch.Tensor, attention_mask: torch.Tensor = None, **kwargs
    ) -> torch.Tensor:
        """
        Applies the pooling operation on input tensor along given dimension.

        Parameters:
            tensor (torch.Tensor): The input tensor to which the operation is applied.

        Returns:
            torch.Tensor: Result of applying the pooling function on the input tensor,
                          along specified dimension.

        """
        if attention_mask is not None:
            attention_mask = attention_mask.bool()
            assert (
                attention_mask[:, -1].all() == True
            ), "Attention mask contains 0s at the end while assuming left padding."
            # nans will be ignored (used w/ attention mask)
            tensor[~attention_mask.bool()] = torch.nan
        return self.op(tensor, dim=self.dim)


POOLING_FUNCTIONS_REGISTRY = {
    "max": TorchPoolingOP,
    "all": TorchPoolingOP,
}


def get_pooling_op(pooling_op_name: str, dim: int):
    """
    Returns a pooling operation based on the provided name and dimension.

    Parameters:
        pooling_op_name (str): The name of the pooling operation to be returned.
        dim (int): The dimension along which the pooling will be performed.

    Returns:
        A callable object representing the desired pooling function.

    Raises:
        KeyError: If an invalid `pooling_op_name` is provided.

    Note:
        This function relies on a global registry of available pooling functions (POOLING_FUNCTIONS_REGISTRY).
        The specifics of this dictionary are not included in the docstring for brevity, but should be consulted if you want to use this function.
    """

    return POOLING_FUNCTIONS_REGISTRY[pooling_op_name](op_name=pooling_op_name, dim=dim)
