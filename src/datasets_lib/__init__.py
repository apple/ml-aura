# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
import typing as t
from pathlib import Path

import torch
import transformers
from .jigsaw_dataset import get_jigsaw_dataset


DATASET_LOADERS_REGISTRY = {
    "jigsaw": get_jigsaw_dataset,
}


def get_dataset(
    name: str,
    datasets_folder: Path,
    split: str,
    tokenizer: t.Optional[transformers.PreTrainedTokenizer] = None,
) -> t.Tuple[torch.utils.data.Dataset, t.Callable]:
    """Loads and returns a dataset split given its name. It also returns a collator function for the dataloader

    Args:
        name (str): dataset name
        datasets_folder (Path): path where dataset is located
        split (bool): train, val, test
        tokenizer (t.Optional[transformers.PreTrainedTokenizer], optional): a huggingface tokenizer in case it is a text dataset. Defaults to None.

    Returns:
        t.Tuple[torch.utils.data.Dataset, t.Callable]: pytorch Dataset instance and collator function
    """
    assert name in DATASET_LOADERS_REGISTRY
    data_loader = DATASET_LOADERS_REGISTRY[name]
    return data_loader(datasets_folder / name, split=split, tokenizer=tokenizer)


def get_dataloader(
    dataset: torch.utils.data.Dataset,
    batch_size: int,
    num_workers: int,
    collate_fn: int,
    drop_last: bool,
    shuffle: bool,
    **kwargs: dict,
) -> torch.utils.data.DataLoader:
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        collate_fn=collate_fn,
        drop_last=drop_last,
        shuffle=shuffle,
        **kwargs,
    )


if __name__ == "__main__":
    pass
