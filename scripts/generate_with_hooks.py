# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.

# Loads a model and a dataset and extracts intermediate responses
import argparse
import logging
import typing as t
import os

from transformers import pipeline, set_seed
from pathlib import Path

from src.models.model_with_hooks import load_huggingface_model, ModelWithHooks
from src.parsers import parsers
from src.hooks import get_hook


logging.getLogger().setLevel(logging.INFO)

# Already run in parallel inside DataLoader
os.environ["TOKENIZERS_PARALLELISM"] = "False"


def print_generated_sentences(output: t.List[t.Dict[str, str]]) -> None:
    for o in output:
        logging.info(o["generated_text"])


def generate(args: argparse.Namespace) -> None:
    model, tokenizer = load_huggingface_model(
        model_path=args.model_path,
        cache_dir=parsers.HF_HUB_CACHE,
        seq_len=args.seq_len,
        device=args.device,
        dtype=args.dtype,
    )
    logging.info(model)
    # Create hooks
    module_names_hooks = ModelWithHooks.find_module_names(model, args.module_names)
    logging.info("Creating hooks for:")
    for m in module_names_hooks:
        logging.info(f"\t{m}")

    hooks = []
    for module_name in module_names_hooks:
        # Load intervention state for this module
        if args.intervention_state_path is not None:
            state_path = (
                Path(args.interventions_cache_dir)
                / args.intervention_state_path
                / f"{module_name}.statedict"
            )
        else:
            state_path = None

        hook = get_hook(
            args.intervention_name,
            module_name=module_name,
            device=args.device,
            state_path=state_path,
        )
        hooks.append(hook)
        logging.info(hook)
    logging.info("Done!")

    # Create hooked model
    model_hooks = ModelWithHooks(
        module=model,
        hooks=hooks,
    )

    # Generate without hooks
    num_sequences = 5
    prompt = "Once upon a time"
    generator = pipeline(
        "text-generation",
        model=model_hooks.module,
        tokenizer=tokenizer,
    )

    # Generate without hooks
    set_seed(42)
    decoded_no_hook = generator(
        prompt, max_length=20, num_return_sequences=num_sequences, do_sample=True
    )
    logging.info("Without hook")
    print_generated_sentences(decoded_no_hook)

    # Register hooks
    model_hooks.register_hooks()

    # Generate with hooks
    set_seed(42)
    decoded_hook = generator(
        prompt, max_length=20, num_return_sequences=num_sequences, do_sample=True
    )
    logging.info("With hook")
    print_generated_sentences(decoded_hook)
    model_hooks.remove_hooks()

    # Generate without hooks again
    set_seed(42)
    decoded_no_hook_2 = generator(
        prompt, max_length=20, num_return_sequences=num_sequences, do_sample=True
    )
    logging.info("Without hook (2nd)")
    print_generated_sentences(decoded_no_hook_2)


def main(args: argparse.Namespace) -> None:
    args = parsers.merge_config_into_args(args)
    generate(args)


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="Generate with hooks",
        description="Extracts and saves responses from a model",
    )
    ####### Adds config specific args #######
    parser = parsers.add_config_args(parser)

    ####### Script Arguments #########
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
        default="float32",
        help="E.g. float32, float32",
    )
    parser.add_argument(
        "--model-path", type=str, required=False, help="Path or url to model weights."
    )
    parser.add_argument(
        "--module-names",
        nargs="*",
        type=str,
        default=[".*"],
        help="Module names to intervene upon.",
    )
    parser.add_argument(
        "--intervention-name",
        type=str,
        default="dummy",
        help="Name of intervention to be applied",
    )
    parser.add_argument(
        "--intervention-state-path",
        type=str,
        default=None,
        help="Path to intervention state file, if any.",
    )
    parser.add_argument(
        "--interventions-cache-dir",
        type=str,
        default=parsers.INTERVENTIONS_CACHE_DIR,
        help="Path to intervention state file, if any.",
    )
    parser.add_argument("--seq-len", type=int, default=128, help="Max sequence length")
    parser.add_argument("--seed", type=int, default=42, help="RNG seed")
    parser.add_argument("--verbose", type=int, default=0)
    return parser


if __name__ == "__main__":
    args = get_parser().parse_args()
    if args.verbose == 1:
        logging.basicConfig(level=logging.INFO)
    elif args.verbose >= 2:
        logging.basicConfig(level=logging.DEBUG)
    main(args)
