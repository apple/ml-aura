# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.

import argparse
import json
import logging
import multiprocessing
from pathlib import Path

import torch

from src.datasets_lib.responses_io import ResponsesLoader
from src.hooks import HOOK_REGISTRY
from src.parsers import parsers
from src.utils.auroc import compute_auroc


def main(args):
    args = parsers.merge_config_into_args(args)
    args = parsers.add_responses_paths(args)
    assert (
        args.num_experts is None or args.num_experts > 0
    ), "--num-experts must be > 0 or None."

    model_name = Path(args.model_path).name

    if args.verbose == 1:
        logging.getLogger().setLevel(logging.INFO)

    logging.info(
        json.dumps(
            {k: str(v) for k, v in vars(args).items()},
            indent=2,
            sort_keys=True,
        )
    )

    CPU_COUNT = (
        multiprocessing.cpu_count() if args.num_workers is None else args.num_workers
    )

    label_map = {label: 1 for label in args.positive_subset}
    label_map.update({label: 0 for label in args.negative_subset})

    responses_loader = ResponsesLoader(
        root=args.responses_cache_dir,
        from_folders=[
            Path(args.tag) / model_name / "*/*/*/*",
        ],
        columns=["responses", "id", "label", "subset"],
        label_map=label_map,
    )

    # Get only the module names requested via args
    module_names = responses_loader.get_attribute_values(
        "module_names", filter=args.module_names
    )

    logging.info(module_names)

    for pooling_op_name in args.pooling_op:
        tag = f"{args.intervention}-{args.intervention_tag}-{pooling_op_name}"
        auroc_per_module = {}
        pool = multiprocessing.Pool(CPU_COUNT)
        for module_name in module_names:
            # Load responses for a given module
            data_subset = responses_loader.load_data_subset(
                {"module_names": module_name, "pooling_op": pooling_op_name},
                num_workers=CPU_COUNT,
            )

            logging.info(module_name)
            logging.info(f"Responses: {data_subset['responses'].shape}")
            logging.info(f"Labels: {data_subset['label'].shape}")
            logging.info(
                f"Computing AUROC on {data_subset['responses'].shape} responses ..."
            )
            auroc = compute_auroc(
                responses=data_subset["responses"],
                labels=data_subset["label"],
                chunk_size=10,
                pool=pool,
            )
            auroc_per_module[module_name] = torch.from_numpy(auroc.astype("float32"))

        pool.close()

        # Compute AUROC threshold depending on the method.
        # Note that Det0 and Damp require *all* neurons in the model to do that, not just per layer.
        if args.intervention == "aura":
            auroc_threshold = 0.5
        else:
            auroc_full = torch.cat(list(auroc_per_module.values()))
            if args.num_experts is not None:
                assert args.num_experts <= len(
                    auroc_full
                ), f"Choosing {args.num_experts} experts but only have {len(auroc_full)} neurons."
            logging.info(f"Finding threshold on {len(auroc_full)} neurons ...")
            auroc_threshold = (
                float(torch.sort(auroc_full, descending=True).values[args.num_experts])
                if args.num_experts is not None
                else 0.5
            )

        # Now, create a hook per layer based on the AUROC threshold found.
        def aura_fn(auroc: torch.Tensor) -> torch.Tensor:
            alpha = torch.ones_like(auroc, dtype=torch.float32)
            mask = auroc > auroc_threshold
            alpha[mask] = 1 - 2 * (auroc[mask] - 0.5)
            return alpha

        def damp_fn(auroc: torch.Tensor) -> torch.Tensor:
            alpha = torch.ones_like(auroc, dtype=torch.float32)
            mask = auroc > auroc_threshold
            alpha[mask] = args.damp_alpha
            return alpha

        def det0_fn(auroc: torch.Tensor) -> torch.Tensor:
            alpha = torch.ones_like(auroc, dtype=torch.float32)
            mask = auroc > auroc_threshold
            alpha[mask] = 0
            return alpha

        alpha_fn_map = {
            "aura": aura_fn,
            "damp": damp_fn,
            "det0": det0_fn,
        }

        logging.info("=" * 40)
        intervention_dir: Path = Path(args.interventions_cache_dir) / tag / model_name
        intervention_dir.mkdir(exist_ok=True, parents=True)
        for module_name in module_names:
            logging.info(f"Saving Hook {module_name} ...")
            hook = HOOK_REGISTRY[args.intervention](
                module_name=module_name,
                alpha=alpha_fn_map[args.intervention](auroc_per_module[module_name]),
            )
            torch.save(
                hook.state_dict(), intervention_dir / (module_name + ".statedict")
            )
    logging.warning(f"Hooks saved in {intervention_dir}")


def arguments_parser():
    parser = argparse.ArgumentParser()

    ####### Adds config specific args #######
    parser = parsers.add_config_args(parser)

    ####### Adds job specific args #######
    parser = parsers.add_job_args(parser)

    ####### Adds response specific args #######
    parser = parsers.add_responses_args(parser)

    ####### Script Arguments #########
    parser.add_argument(
        "--intervention",
        type=str,
        default="aura",
        choices=["det0", "damp", "aura"],
        help="Intervention type. ",
    )
    parser.add_argument(
        "--positive-subset",
        type=str,
        nargs="+",
        default=[],
        help="Data subsets to serve as positive examples.",
    )
    parser.add_argument(
        "--negative-subset",
        type=str,
        nargs="+",
        default=[],
        help="Data subsets to serve as negative examples.",
    )
    parser.add_argument(
        "--interventions-cache-dir",
        type=Path,
        default=parsers.INTERVENTIONS_CACHE_DIR,
        help="Temporary path where interventions are saved locally.",
    )
    parser.add_argument(
        "--intervention-tag",
        type=str,
        default="toxicity",
        help="The tag (folder) in which interventions are saved. "
        "Do not add the intervention name, it will be prepended. "
        "Eg. if --intervention-tag=toxicity, the final tag will be aura-toxicity.",
    )
    parser.add_argument(
        "--num-experts",
        type=int,
        default=None,
        help="If None, all experts with AUROC>0.5 are selected. "
        "Otherwise, experts are sorted by AUROC and the "
        "top --num-experts are selected. "
        "Applies to interventions `det0` and `damp`.",
    )
    parser.add_argument(
        "--damp-alpha",
        type=float,
        default=0.5,
        help="The fixed dampening factor for intervention `damp`. "
        "This factor will be multiplied with the neuron output.",
    )
    parser.add_argument(
        "--num-workers", type=int, default=None, help="Number of workers in dataloader."
    )
    parser.add_argument("--seed", type=int, default=42, help="RNG seed")
    return parser


if __name__ == "__main__":
    args = arguments_parser().parse_args()
    args = parsers.merge_config_into_args(args)
    args = parsers.add_responses_paths(args)
    logging.getLogger().setLevel(logging.INFO)
    logging.info(args)
    main(args)
