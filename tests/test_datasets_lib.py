# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
import pytest
from pathlib import Path
from transformers import AutoTokenizer
from src.datasets_lib import get_dataset, get_dataloader


@pytest.fixture(scope="session")
def dummy_data():
    tokenizer = AutoTokenizer.from_pretrained("sshleifer/tiny-gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    dataset, collator = get_dataset(
        "jigsaw", Path("tests/data/"), split="train", tokenizer=tokenizer
    )
    return {"dataset": dataset, "collator": collator}


def test_get_dataset(dummy_data):
    assert (
        dummy_data["dataset"] is not None
    )  # assuming non-empty datasets for simplicity


def test_get_dataloader(dummy_data):
    dataloader = get_dataloader(
        dummy_data["dataset"],
        batch_size=2,
        num_workers=0,
        collate_fn=dummy_data["collator"],
        drop_last=True,
        shuffle=False,
    )

    # check if the dataloader is iterable and returns correct batches
    for i, batch in enumerate(dataloader):
        assert len(batch["input_ids"]) == 2  # assuming a batch size of 2
