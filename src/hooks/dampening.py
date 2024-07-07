# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.

import typing as t
from pathlib import Path

import torch


class _DampeningHook(torch.nn.Module):
    def __init__(
        self,
        module_name: str,
        state_path: t.Optional[t.Union[Path, str]] = None,
        alpha: t.Optional[torch.Tensor] = None,
        device: str = "cuda",
    ):
        super().__init__()
        assert (state_path is None and alpha is not None) or (
            state_path is not None and alpha is None
        ), "Set either state_path or alpha."
        self.module_name = module_name
        self.device = device
        if state_path:
            state_dict = torch.load(state_path, map_location=device)
            self.load_state_dict(state_dict)
        else:
            self.register_buffer("alpha", alpha)
        if len(self.alpha.shape) == 1:
            self.alpha = self.alpha.unsqueeze(0).unsqueeze(0)

    def load_state_dict(
        self,
        state_dict: t.Mapping[str, torch.Any],
        strict: bool = True,
        assign: bool = False,
    ):
        alpha = state_dict["alpha"].to(self.device)
        self.register_buffer("alpha", alpha)

    def __call__(self, module, input, output) -> t.Any:
        return output * self.alpha


class Det0Hook(_DampeningHook): ...


class DampHook(_DampeningHook): ...


class AURAHook(_DampeningHook): ...
