# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
from pathlib import Path
from scripts import compute_responses
import os
import pandas as pd
import tempfile
import logging
from src.parsers import parsers

logger = logging.getLogger("TEST(compute responses)")
logger.setLevel(logging.DEBUG)


def test_compute_responses_e2e():
    with tempfile.TemporaryDirectory(dir="/tmp") as tmpdirname:
        logger.info("Setting up parser.")
        parser = compute_responses.get_parser()
        args = parser.parse_args(
            args=[
                "--config-path",
                "tests/configs/responses_test.yaml",
                "--data-dir",
                "./tests/data",
                "--device",
                "cpu",
                "--responses-cache-dir",
                tmpdirname,
            ]
        )
        logger.info("Calling main()")
        compute_responses.main(args)
        logger.info("Checking if responses have been saved...")
        [dataset, ] = parsers.get_single_args(args, ["dataset"])
        model_name = compute_responses.get_model_name_from_path(args.model_path)
        output_path = Path(args.responses_cache_dir)
        data = pd.read_csv(Path(args.data_dir) / dataset / "train.csv")
        base_path = output_path / args.tag / model_name / dataset
        for subset in args.subset:
            subset_path = base_path / subset
            module_names = os.listdir(subset_path)
            for module_name in module_names:
                if not os.path.isdir(subset_path / module_name):
                    continue
                pooling_op_path = subset_path / module_name / args.pooling_op[0]
                for id in data[data["toxic"] == int(subset == "toxic")]["id"]:
                    filename = pooling_op_path / f"{id}.pt"
                    logger.info(f"Checking if {str(filename)} exists")
                    assert (pooling_op_path / f"{id}.pt").exists()
        logger.info("Done.")
