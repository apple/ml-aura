# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.

import argparse
import itertools
import pathlib
import typing as t
import os

from ..utils import utils


INTERVENTIONS_CACHE_DIR = os.environ.get(
    "INTERVENTIONS_CACHE_DIR", pathlib.Path("/tmp/cache/model-interventions")
)
RESPONSES_CACHE_DIR = os.environ.get(
    "RESPONSES_CACHE_DIR", pathlib.Path("/tmp/cache/model-responses")
)
HF_HUB_CACHE = os.environ.get(
    "HF_HUB_CACHE", pathlib.Path("~/.cache/huggingface/hub").expanduser()
)


def get_single_args(args: argparse.Namespace, arg_names: t.List[str]) -> t.List[t.Any]:
    # Sanity check, only allowing multiplicity for args.module_names and args.subset
    selected = [getattr(args, a) for a in arg_names]
    assert all([len(a) == 1 for a in selected])
    return list(map(lambda x: x[0], selected))


def add_config_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument(
        "--config-path",
        type=pathlib.Path,
        help="Path to a yaml file that contains all that config that will overwrite CLI args.",
    )
    return parser


def add_wandb_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument(
        "--wandb-group",
        type=str,
        default=None,
        help="WandB group, if any.",
    )
    parser.add_argument(
        "--wandb-project",
        type=str,
        default="model-interventions",
        help="WandB project name.",
    )
    parser.add_argument(
        "--wandb-uid",
        type=str,
        default=None,
        help="WandB job name.",
    )
    parser.add_argument(
        "--wandb-logs-dir",
        type=str,
        default="model-interventions",
        help="WandB logging dir.",
    )
    parser.add_argument(
        "--wandb-team", type=str, default="my-team", help="The wandb team."
    )
    parser.add_argument(
        "--wandb-enabled",
        type=int,
        default=1,
        help="Set to 0 to disable WandB, otherwise enabled (default).",
    )

    return parser


def add_responses_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument(
        "--responses-paths",
        nargs="+",
        default=[],
        type=str,
        help="Paths to pkl files with responses to load.",
    )
    parser.add_argument(
        "--responses-cache-dir",
        default=RESPONSES_CACHE_DIR,
        type=pathlib.Path,
        help="Paths to where responses are saved before upload.",
    )
    parser.add_argument(
        "--tag",
        default="*",
        type=str,
        help="Tag in the hub remote dir.",
    )
    parser.add_argument(
        "--model-path", type=str, default=None, help="Path or url to model weights."
    )
    parser.add_argument(
        "--dataset",
        nargs="*",
        default=["*"],
        type=str,
        help="Dataset name(s) in the hub remote dir.",
    )
    parser.add_argument(
        "--subset",
        nargs="*",
        default=["*"],
        type=str,
        help="Dataset subset name(s) in the hub remote dir.",
    )
    parser.add_argument(
        "--module-names",
        nargs="*",
        default=["*"],
        type=str,
        help="Module name(s) in the hub remote dir.",
    )
    parser.add_argument(
        "--pooling-op",
        nargs="*",
        type=str,
        default=["min", "max", "mean", "median", "std", "last"],
        help="Pooling function from many tokens to one.",
    )
    return parser


def add_job_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    ####### Job Arguments ########
    parser.add_argument(
        "--data-dir",
        type=str,
        default="./data/",
        help="Path to datasets",
    )
    parser.add_argument(
        "--output-dir",
        type=pathlib.Path,
        default=None,
        help="Path were experiment outputs are saved",
    )
    parser.add_argument(
        "--verbose",
        type=int,
        default=0,
        help="Whether to set logging to info.",
    )
    return parser


def add_responses_paths(args: argparse.Namespace) -> argparse.Namespace:
    args.model_name = pathlib.Path(args.model_path).name
    list_args = [
        "tag",
        "model_name",
        "dataset",
        "subset",
        "module_names",
        "pooling_op",
    ]
    to_combine = [getattr(args, a) for a in list_args]
    # Make all lists so product accounts for all
    to_combine = [a if isinstance(a, list) else [a] for a in to_combine]
    combinations = list(itertools.product(*to_combine))
    args.responses_paths += ["/".join(c) for c in combinations]
    del args.model_name
    return args


def merge_config_into_args(args: argparse.Namespace) -> argparse.Namespace:
    if args.config_path:
        config = utils.load_yaml(args.config_path)  # TODO test this function
        args = utils.update_args_with_config(
            args, config["parameters"]
        )  # TODO test this function
    return args
