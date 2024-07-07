# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

from scripts.learn_aura import arguments_parser, main


@pytest.fixture
def args():
    parser = arguments_parser()
    args = parser.parse_args(
        [
            "--intervention",
            "aura",
            # tiny-gpt has 2 neurons and we analyze 2 layers (total 4 neurons).
            # Choosing to intervene on 3 of them for det0 and damp.
            "--num-experts",
            "3",
            "--interventions-cache-dir",
            "tests/data",
            "--config-path",
            "tests/configs/aura_test.yaml",
            "--responses-cache-dir",
            "tests/data/",
            "--intervention-tag",
            "test",
            "--num-workers",
            "1",
        ]
    )
    return args


@pytest.mark.parametrize("intervention", ["aura", "det0", "damp"])
def test_main(args, intervention):
    # Assuming that the main function doesn't have any side effects and returns None when successful
    with tempfile.TemporaryDirectory(dir="/tmp/") as tempfolder:
        cache_dir = args.interventions_cache_dir
        args.intervention = intervention
        args.interventions_cache_dir = Path(tempfolder)
        main(args)
        statedict_in_tests = torch.load(
            f"tests/data/{intervention}-toxicity-max/tiny-gpt2/transformer.h.0.mlp.c_proj.statedict"
        )
        statedict_created = torch.load(
            Path(tempfolder)
            / f"{intervention}-test-max/tiny-gpt2/transformer.h.0.mlp.c_proj.statedict"
        )
        assert np.allclose(
            statedict_in_tests["alpha"].numpy(), statedict_created["alpha"].numpy()
        )
        args.interventions_cache_dir = cache_dir


# Case 1: Responses paths do not exist
@pytest.mark.xfail()
def test_main_non_existent_responses(args):
    args.responses_paths = ["tests/data/nonexistent_responses"]
    main(args)
