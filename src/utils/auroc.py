# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.

import numpy as np
from sklearn.metrics import roc_auc_score
from itertools import repeat
import multiprocessing


def _compute_auroc_chunk(responses: np.ndarray, labels: np.ndarray):
    """
    Function to compute Area Under the Receiver Operating Characteristic Curve (AUROC) for a chunk of data.

    Parameters:
    responses (np.ndarray): Array of model responses.
    labels (np.ndarray): Array of true labels.
    start (int): Starting index for the chunk of data.
    chunk_size (int): Desired size of the chunk.

    Returns:
    np.ndarray: Array of AUROC scores for the chunk of data.
    """

    # Compute and return AUROC scores for the chunk of data
    return roc_auc_score(
        labels[:, None].repeat(responses.shape[1], 1),
        responses,
        average=None,
    )


def compute_auroc(
    responses: np.ndarray,
    labels: np.ndarray,
    num_threads: int = 10,
    chunk_size: int = 10,
    pool=None,
) -> np.ndarray:
    """
    This function computes the Area Under the Receiver Operating Characteristic (AUROC) scores.

    Parameters:
    responses (np.ndarray): The array of predicted responses.
    labels (np.ndarray): The array of actual labels.
    num_threads (int, optional): The number of threads to use in parallel processing. Defaults to 10.
    chunk_size (int, optional): The size of each chunk of data to process at a time. Defaults to 10.

    Returns:
    np.ndarray: The array of computed AUROC scores.
    """
    responses_map = [
        responses[:, start : (start + chunk_size)]
        for start in np.arange(0, responses.shape[1], chunk_size)
    ]
    args = zip(responses_map, repeat(labels))
    if pool is None:
        with multiprocessing.Pool(num_threads) as pool:
            ret = pool.starmap(_compute_auroc_chunk, args)
    else:
        ret = pool.starmap(_compute_auroc_chunk, args)

    return np.concatenate(ret, 0)
