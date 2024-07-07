# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.

import argparse
import os
import re
import sys
import typing as t
from pathlib import Path

import yaml


def load_yaml(path: Path) -> t.Union[t.List, t.Dict]:
    # Adding float resolver that includes "1e-3" like floats. Otherwise they are loaded as strings.
    # https://stackoverflow.com/questions/30458977/yaml-loads-5e-6-as-string-and-not-a-number
    loader = yaml.SafeLoader
    loader.add_implicit_resolver(
        "tag:yaml.org,2002:float",
        re.compile(
            """^(?:
         [-+]?(?:[0-9][0-9_]*)\\.[0-9_]*(?:[eE][-+]?[0-9]+)?
        |[-+]?(?:[0-9][0-9_]*)(?:[eE][-+]?[0-9]+)
        |\\.[0-9_]+(?:[eE][-+][0-9]+)?
        |[-+]?[0-9][0-9_]*(?::[0-5]?[0-9])+\\.[0-9_]*
        |[-+]?\\.(?:inf|Inf|INF)
        |\\.(?:nan|NaN|NAN))$""",
            re.X,
        ),
        list("-+0123456789."),
    )

    with open(path, "r") as infile:
        return yaml.load(infile, Loader=loader)


def make_safe_dict(input):
    def _make_safe(value):
        if isinstance(value, (int, float, str, type(None))):
            return value
        elif isinstance(value, list):
            return [_make_safe(item) for item in value]
        elif isinstance(value, dict):
            return {k: _make_safe(v) for k, v in value.items()}
        else:
            return str(value)

    return _make_safe(input)


def dump_yaml(data: t.Union[t.Dict, t.List], path: Path) -> t.Union[t.List, t.Dict]:
    with open(path, "w") as outfile:
        return yaml.safe_dump(make_safe_dict(data), outfile)


def update_args_with_config(
    args: argparse.Namespace,
    config: t.Dict,
    ignore_mismatches: bool = False,
) -> argparse.Namespace:
    args_dict = vars(args)
    for k, v in config.items():
        _argv = f"--{k.replace('_', '-')}"
        if k in args_dict:
            if _argv not in sys.argv:
                args_dict[k] = v  # only update if not manually set by CLI
        elif not (ignore_mismatches):
            raise ValueError(f"{k} argument found in config file but not in Argparse.")
        else:
            args_dict[k] = v
    return args


def setup_wandb(args: argparse.Namespace) -> t.Any:
    import wandb

    wandb_config = load_yaml(".wandb.yaml")
    os.environ["WANDB_API_KEY"] = wandb_config["WANDB_API_KEY"]
    os.environ["WANDB_BASE_URL"] = wandb_config["WANDB_BASE_URL"]
    run = wandb.init(
        project=args.wandb_project,
        entity=args.wandb_team or None,
        config=vars(args),
        group=args.wandb_group or None,
        name=args.wandb_uid or None,
        dir=args.wandb_logs_dir,
        mode="online" if args.wandb_enabled == 1 else "disabled",
    )
    return run
