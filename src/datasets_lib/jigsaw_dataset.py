# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
from pathlib import Path
import transformers
from .collators import DictCollatorWithPadding
import torch
import pandas as pd
from collections import OrderedDict


class JigsawDataset(torch.utils.data.Dataset):
    """
    Implements a loader for the Jigsaw toxicity dataset.
    To get the files download from the following URL into `path`:
    https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge/data
    """

    LABEL_MAP = OrderedDict([("non-toxic", 0), ("toxic", 1)])
    LABEL_NAMES = ["non-toxic", "toxic"]

    def __init__(
        self, path: Path, split: str, tokenizer: transformers.PreTrainedTokenizer
    ) -> torch.utils.data.Dataset:
        self.split = split
        self.path = path
        self.tokenizer = tokenizer

        if self.split == "train":
            train_data = pd.read_csv(path / "train.csv", index_col="id")
            self.data = self._preprocess(train_data)
        elif self.split == "test":
            test_data = pd.read_csv(path / "test.csv", index_col="id")
            test_labels = pd.read_csv(path / "test_labels.csv", index_col="id")
            test_dataset = pd.concat(
                [test_data, test_labels], axis=1, ignore_index=False
            )
            # test dataset comes with unannotated data (label=-1)
            test_dataset = test_dataset.loc[
                (test_dataset[test_dataset.columns[1:]] > -1).all(axis=1)
            ]
            self.data = self._preprocess(test_dataset)
        _ = self.data[0]  # small test
        self.index = torch.arange(len(self.data))

    def set_label(self, label: str) -> None:
        index = torch.arange(len(self.data))
        if label is None:
            self.index = index
        else:
            labels = torch.asarray([d["toxic"] for d in self.data])
            int_label = self.LABEL_MAP[label]
            self.index = index[labels == int_label]

    def _preprocess(self, df):
        # Remove samples that are marked as non-toxic but e.g. insult or identity_hate is true
        remove_mask = (df["toxic"] == 0) & (
            df[["severe_toxic", "obscene", "threat", "insult", "identity_hate"]].max(1)
            != 0
        )
        df = df[~remove_mask]
        return df.reset_index().to_dict("records")

    def __getitem__(self, item):
        datum = self.data[int(self.index[item])]
        tokens = self.tokenizer(datum["comment_text"], truncation=True, padding=False)
        datum.update(tokens)
        return datum

    def __len__(self):
        return len(self.index)


def get_jigsaw_dataset(
    path: Path, split: str, tokenizer: transformers.PreTrainedTokenizer
) -> torch.utils.data.Dataset:
    return JigsawDataset(path, split, tokenizer), DictCollatorWithPadding(tokenizer)
