# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.

import pathlib
import re
import typing as t
from pathlib import Path

import torch
from torch.utils.hooks import RemovableHandle
from transformers import (
    AutoConfig,
    AutoModel,
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
)

TASK_MAPPING = {
    "text-generation": AutoModelForCausalLM,
    "sequence-classification": AutoModelForSequenceClassification,
}


def is_module_name_in_regex(module_name: str, regex_list: t.List[str]) -> bool:
    """Checks whether module_name matches any of the regular expressions in a list

    Args:
        module_name (str): name of the module to find
        regex_list (t.List[str]): list with regex expressions

    Returns:
        bool: whether any of the expressions in the list match the module name
    """
    all_match = [re.fullmatch(regex, module_name) is not None for regex in regex_list]
    return any(all_match)


def load_huggingface_model(
    model_path: t.Union[str, pathlib.Path],
    cache_dir: t.Optional[t.Union[str, pathlib.Path]],
    dtype: t.Any,
    device: str,
    seq_len: t.Optional[int] = None,
    rand_weights: bool = False,
    task: t.Optional[str] = "text-generation",
) -> t.Tuple[PreTrainedModel, PreTrainedTokenizer]:

    # Defaults for device and dtype
    if device is None:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
    if dtype is None:
        dtype = torch.get_default_dtype()
    else:
        dtype = dtype

    model_class = TASK_MAPPING.get(task, AutoModel)

    cache_dir = Path(cache_dir).expanduser().absolute()
    full_model_path = cache_dir / model_path
    if full_model_path.exists():
        model_path = full_model_path

    if rand_weights:
        config = AutoConfig.from_pretrained(
            model_path, cache_dir=cache_dir, device_map=device
        )
        model = model_class.from_config(config)
        tokenizer = AutoTokenizer.from_pretrained(model_path)
    else:
        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            cache_dir=cache_dir,
            device_map=device,
            dtype=dtype,
        )
        model = model_class.from_pretrained(
            model_path,
            cache_dir=cache_dir,
            device_map=device,
            force_download=False,
            torch_dtype=getattr(torch, dtype),
        )
    if seq_len:
        tokenizer.model_max_length = seq_len
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        if tokenizer.bos_token is not None:
            tokenizer.pad_token = tokenizer.bos_token
        else:
            tokenizer.add_special_tokens({"pad_token": "<pad>"})
            model.resize_token_embeddings(len(tokenizer))
            tokenizer.pad_token = "<pad>"
    return model, tokenizer


class ModelWithHooks:
    """
    Class wrapping a Pytorch model so that we can apply forward hooks on its responses.
    """

    def __init__(
        self,
        module: torch.nn.Module,
        hooks: t.Optional[t.List[torch.nn.Module]] = None,
    ) -> None:
        """
        Initialize the ModelWithHooks instance with a PyTorch module and optional hooks.

        Args:
            module (torch.nn.Module): A pytorch nn.module holding the model to be wrapped.
            hooks (Optional[List[torch.nn.Module]]): List of hooks that will be applied to the module's forward method. Default is None.
        """
        self.hooks: t.Dict[str, torch.nn.Module] = (
            {h.module_name: h for h in hooks} if hooks is not None else {}
        )
        self.module = module
        self._forward_hook_handles: t.List[RemovableHandle] = []

    @staticmethod
    def find_module_names(
        model: torch.nn.Module,
        regex_list: t.List[str],
        skip_first: int = 0,  # first layer tends to be the model root
    ) -> t.List[str]:
        """
        Find the names of modules in a PyTorch model that match certain regular expressions.

        Args:
            model (torch.nn.Module): The PyTorch model to search through.
            regex_list (List[str]): A list of regular expression patterns for matching module names.
            skip_first (int): Number of modules at the start of the model that should be skipped (e.g., root module). Defaults to 0.

        Returns:
            List[str]: A list of strings, where each string is a name of a module in `model` that matches one of the regular expressions from `regex_list`.
        """
        module_names = [name[0] for name in model.named_modules()][skip_first:]

        def is_module_name_in_regex_fn(module_name):
            return is_module_name_in_regex(module_name, regex_list)

        module_names = list(filter(is_module_name_in_regex_fn, module_names))
        return module_names

    def get_module(self) -> torch.nn.Module:
        return self.module

    def register_hooks(
        self,
        hooks=None,
    ):
        """
        Register forward hooks on the wrapped PyTorch model.

        Args:
            hooks (Optional[List[torch.nn.Module]]): List of hooks that will be applied to the module's forward method. Default is None. If provided, these are registered as new hooks and any existing hooks are replaced.

        The `hooks` argument should correspond to a dictionary mapping from module names (as returned by named_modules) to hook functions (or other callables that take three arguments: input, output, and module). These hook functions will be registered as forward hooks for the corresponding modules.
        """
        # register forward hook for all modules in the network with the exception of the root
        # module and container modules.
        if hooks is not None:
            assert len(self.hooks) == 0, "Hooks already registered"
            self.hooks: t.Dict[str, torch.nn.Module] = (
                {h.module_name: h for h in hooks} if hooks is not None else {}
            )
        self._forward_hook_handles = []
        for module_name, module in self.module.named_modules():
            if module_name in self.hooks:
                hook_fn = self.hooks[module_name]
                self._forward_hook_handles.append(module.register_forward_hook(hook_fn))

    def remove_hooks(self):
        """
        Remove all registered forward hooks from the wrapped PyTorch model and reset attributes.

        This method unregisters all hooks previously registered using `register_hooks`, effectively removing any functionality they were providing.
        """
        for h in self._forward_hook_handles:
            h.remove()
        self._forward_hook_handles = []
        self.hooks = {}

    def get_hook_outputs(self):
        """
        Retrieve the outputs of all hooks registered on modules in the wrapped PyTorch model.

        Returns:
            dict: A dictionary mapping from module names to their respective output, as stored by any hook that was registered for these modules and executed after a forward pass through the network.

        This method allows one to access the outputs of all hooks in an easy-to-use format, even if they were registered on different modules or layers. Note that the `outputs` attribute of each hook should be updated with the output of its respective module after a forward pass through the network.
        """
        outputs = {}
        for module_name, hook in self.hooks.items():
            outputs.update(hook.outputs)
        return outputs

    def update_hooks(self, *args, **kwargs):
        """
        Update the parameters of all hooks registered on modules in the wrapped PyTorch model.

        This method allows one to change the behaviour or configuration of any or all hooks registered for modules after initialization (e.g., due to some external condition).
        """
        assert len(self.hooks) > 0
        for module_name, hook in self.hooks.items():
            hook.update(*args, **kwargs)

    def get_target_module_names(self) -> t.List[str]:
        return list(self.hooks.keys())
