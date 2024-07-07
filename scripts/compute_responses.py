# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.

# Loads a model and a dataset and extracts intermediate responses
import argparse
import logging
import os
import typing as t
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from src import hooks
from src.datasets_lib import get_dataloader, get_dataset
from src.models.model_with_hooks import ModelWithHooks, load_huggingface_model
from src.parsers import parsers
from src.utils import utils


# Already run in parallel inside DataLoader
os.environ["TOKENIZERS_PARALLELISM"] = "False"


def get_model_name_from_path(model_path: t.Union[Path, str]) -> str:
    """Extracts the model name from a given model path.

    Args:
        model_path (t.Union[Path, str]): A string or Path object representing
            the file system path to the trained model.

    Returns:
        str: The last part of the provided path as a string representing
            the name of the trained model.
    """
    return str(Path(model_path).name)


def compute_responses(args: argparse.Namespace) -> None:
    # Sanity check, only allowing multiplicity for args.module_names and args.subset
    [dataset, ] = parsers.get_single_args(args, ["dataset"])
    model_name = get_model_name_from_path(args.model_path)

    # Setting paths
    output_path = Path(args.responses_cache_dir)
    base_path = output_path / args.tag / model_name / dataset

    if args.verbose == 1:
        logging.getLogger().setLevel(logging.INFO)

    # Logging arguments
    for arg, value in sorted(vars(args).items()):
        logging.info("Argument %s: %r", arg, value)

    # Set random seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Setup device and distributed learning
    if args.device in ["cuda", None] and torch.cuda.is_available():
        args.device = "cuda"
    elif args.device == "cuda":
        raise (RuntimeError("Cuda not available"))
    elif args.device is None:
        args.device = "cpu"

    # Models and Tokenizers
    module, tokenizer = load_huggingface_model(
        model_path=args.model_path,
        cache_dir=parsers.HF_HUB_CACHE,
        device=args.device,
        dtype=args.dtype,
        rand_weights=(args.rand_weights == 1),
        seq_len=args.seq_len,
    )
    model = ModelWithHooks(
        module=module,
    )

    # Datasets
    train_dataset, collate_fn = get_dataset(
        name=dataset,
        datasets_folder=Path(args.data_dir),
        split="train",
        tokenizer=tokenizer,
    )

    module_names = model.find_module_names(module, args.module_names)

    assert isinstance(args.subset, list)
    if len(args.subset) == 0:
        subsets = train_dataset.LABEL_NAMES
    else:
        subsets = args.subset

    # NOTE: batchnorm will not work properly since we split data by subset (concept / label)
    if subsets == ["*"]:
        subsets = train_dataset.LABEL_NAMES
    for subset in subsets:
        logging.info(f"Current subset: {subset}")
        train_dataset.set_label(subset)
        label_output_path = base_path / subset
        for module_name in module_names:
            module_path = label_output_path / module_name
            os.makedirs(module_path, exist_ok=True)
            utils.dump_yaml(vars(args), label_output_path / "config.yaml")
        hook_fns = [
            hooks.get_hook(
                "postprocess_and_save",
                module_name=module_name,
                pooling_op_names=args.pooling_op,
                output_path=label_output_path,
                save_fields=["id"],
                threaded=False,
            )
            for module_name in module_names
        ]
        model.remove_hooks()
        model.register_hooks(hook_fns)
        checkpoint = {"current_batch": 0}
        checkpoint_path = label_output_path / "checkpoint.pt"
        logging.info(f"Checkpointing to {str(checkpoint_path)}")
        if args.resume == 1:
            if checkpoint_path.exists():
                checkpoint = torch.load(checkpoint_path)
                logging.info(f"Loaded existing checkpoint.")
        current_batch = checkpoint["current_batch"]

        # Sampling and dataloader
        loader = get_dataloader(
            train_dataset,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            collate_fn=collate_fn,
            drop_last=False,
            shuffle=True,
        )
        max_batches = (
            len(loader)
            if args.max_batches is None
            else min(len(loader), args.max_batches)
        )
        if current_batch == max_batches:
            logging.warning(f"All batches found in [{output_path / args.tag}], nothing to compute.")
        else:
            if current_batch > 0:
                logging.info(f"Resuming from batch {current_batch}")
            else:
                logging.info("Computing batch responses")

            iloader = iter(loader)
            for idx in tqdm(range(max_batches)):
                batch = next(iloader)
                if idx >= current_batch:
                    with torch.inference_mode():
                        model.update_hooks(batch_idx=idx, batch=batch)
                        input_ids, attention_mask = (
                            batch["input_ids"],
                            batch["attention_mask"],
                        )
                        input_ids = input_ids.to(args.device)
                        attention_mask = attention_mask.to(args.device)
                        try:
                            module(input_ids=input_ids, attention_mask=attention_mask)
                        except hooks.custom_exceptions.TargetModuleReached:
                            pass
                checkpoint["current_batch"] = idx + 1
                torch.save(checkpoint, checkpoint_path)
            logging.info("Done")
    logging.warning(f"Responses saved in {output_path / args.tag}.")

def main(args: argparse.Namespace) -> None:
    args = parsers.merge_config_into_args(args)
    compute_responses(args)


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="Compute Responses",
        description="Extracts and saves responses from a model",
    )
    # Adds config specific args
    parser = parsers.add_config_args(parser)

    # Adds response specific args
    parser = parsers.add_responses_args(parser)

    # Adds job specific args
    parser = parsers.add_job_args(parser)

    # Script Arguments
    parser.add_argument(
        "--batch-size",
        type=int,
        default=2,
        help="Batch size to use in dataloader",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="cuda, cpu, mps.",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        help="E.g. float32, float32",
    )
    parser.add_argument(
        "--max-batches",
        type=int,
        default=None,
        help="Limit number of batches to process.",
    )
    parser.add_argument(
        "--num-workers", type=int, default=1, help="Number of workers in dataloader."
    )
    parser.add_argument("--seq-len", type=int, default=128, help="Max sequence length")
    parser.add_argument("--seed", type=int, default=42, help="RNG seed")
    parser.add_argument(
        "--resume",
        type=int,
        default=0,
        help="Whether to resume from the last batch",
    )
    parser.add_argument(
        "--rand-weights",
        type=int,
        default=0,
        help="Whether to initialize model with random weights",
    )
    return parser


if __name__ == "__main__":
    args = get_parser().parse_args()
    main(args)
