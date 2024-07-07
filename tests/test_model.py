# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
from src.models.model_with_hooks import ModelWithHooks, load_huggingface_model
import torch


# Define a dummy hook
class DummyHook:
    def __init__(self, module_name=""):
        self.module_name = module_name

    def update(self): ...

    def __call__(self, module, input, output):
        self.outputs = {"": "dummy"}


def test_init():
    model = torch.nn.Linear(5, 2)
    mwh = ModelWithHooks(model)
    assert mwh.get_module() == model
    assert len(mwh._forward_hook_handles) == 0


def test_register_hooks():
    model = torch.nn.Linear(5, 2)
    mwh = ModelWithHooks(model)

    hook = DummyHook()
    mwh.register_hooks([hook])

    # Test if the hook is registered correctly and outputs are as expected
    assert len(mwh._forward_hook_handles) == 1
    x = torch.randn((2, 5))
    _ = model(x)
    mwh.update_hooks()
    assert "dummy" in mwh.get_hook_outputs()[""]


def test_remove_hooks():
    model = torch.nn.Linear(5, 2)
    mwh = ModelWithHooks(model)

    hook = DummyHook()
    mwh.register_hooks([hook])

    # Remove the hook and check if it's empty
    mwh.remove_hooks()
    assert len(mwh._forward_hook_handles) == 0
    assert len(mwh.get_hook_outputs()) == 0


def test_find_module_names():
    model, tokenizer = load_huggingface_model(
        "sshleifer/tiny-gpt2",
        rand_weights=True,
        cache_dir="/tmp/cache",
        device="cpu",
        dtype=torch.float32,
    )
    mwh = ModelWithHooks(model)

    # Test if the method can find modules correctly
    module_names = mwh.find_module_names(mwh.get_module(), [".*0.mlp.*"])
    assert len(module_names) == 5


def test_get_target_module_names():
    model, tokenizer = load_huggingface_model(
        "sshleifer/tiny-gpt2",
        rand_weights=True,
        cache_dir="/tmp/cache",
        device="cpu",
        dtype=torch.float32,
    )
    mwh = ModelWithHooks(model)

    hook = DummyHook("transformer.h.0.mlp.c_fc")
    mwh.register_hooks([hook])

    # Test if the method can get target modules correctly
    target_module_names = mwh.get_target_module_names()
    assert "transformer.h.0.mlp.c_fc" in target_module_names
