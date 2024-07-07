# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
import typing as t
import torch


class DummyHook(torch.nn.Module):
    """A dummy hook to test the behavior of hooks."""

    def __init__(
        self,
        module_name: str,
        device: str = "cuda",
        **kwargs,
    ):
        super().__init__()
        self.module_name = module_name
        self.device = device
        self.counter = 0
        self.units_to_intervene = torch.arange(10).to(self.device)

    def update(self, *args, **kwargs):
        self.counter += 1
        return None

    def __call__(self, module, input, output) -> t.Any:
        # output[:, :, self.units_to_intervene] = 1000
        output[:, :, :] = 1000
        return output


def get_dummy_hook(module_name: str, device: str = "cuda") -> DummyHook:
    return DummyHook(module_name, device)
