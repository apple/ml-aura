# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
from .dampening import DampHook, Det0Hook, AURAHook
from .dummy import DummyHook
from .postprocess_and_save_hook import PostprocessAndSaveHook
import torch

HOOK_REGISTRY = {
    "dummy": DummyHook,
    "postprocess_and_save": PostprocessAndSaveHook,
    "det0": Det0Hook,
    "damp": DampHook,
    "aura": AURAHook,
}


def get_hook(name: str, *args, **kwargs) -> torch.nn.Module:
    return HOOK_REGISTRY[name](*args, **kwargs)
