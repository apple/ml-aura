# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
import typing as t
from .pooling_ops import get_pooling_op
from pathlib import Path
import torch
import logging
from collections import defaultdict
import threading
import os


class PostprocessAndSaveHook(torch.nn.Module):
    """
    A PyTorch module to post-process and save outputs of other modules.

    This hook takes as input the output of another module in a model, applies some
    operations on it (like pooling), optionally saves it somewhere, and if required,
    returns the processed output back for further use. Since it is a pytorch
    module, it supports loading and saving state dicts

    Parameters:
        module_name : str
            Name of the parent module whose outputs are being hooked to.

        pooling_op_names : list[str]
            List of pooling operation names to be applied on the input data, e.g., ['max', 'mean'].

        output_path : Path
            The location where the processed outputs (if any) should be saved or not (None).

        save_fields : list[str]
            Fields of interest in the inputs and/or outputs to be stored for later use, e.g., ['features', 'labels'].

        return_outputs : bool, optional
            If True, returns the processed output back from this hooked module for further use. Default is False.

        threaded : bool, optional
            If True, the hook will be launched in a different thread.

    """

    def __init__(
        self,
        module_name: str,
        pooling_op_names: t.List[str],
        output_path: t.Optional[Path],
        save_fields: t.List[str],
        return_outputs: bool = False,
        threaded: bool = True,
    ):
        super().__init__()
        self.module_name = module_name
        # Storing as modules to allow stateful ops
        # these are applied independently, not one after the other
        self.pooling_ops = torch.nn.ModuleList(
            [get_pooling_op(name, dim=1) for name in pooling_op_names]
        )
        self.output_path = output_path
        self.save_fields = save_fields
        self.return_outputs = return_outputs
        self.batch_idx = None
        self.attention_mask = None
        self.threaded = threaded
        self.thread_handle = None

    def update(
        self,
        batch_idx: int,
        batch: dict,
    ) -> None:
        """
        Updates the state of this hook with new input data.

        This includes setting the current batch index and updating the inputs,
        which are then processed by the pooling operations defined in __init__().

        Parameters:
            batch_idx : int
                The index of the current mini-batch in a full epoch or dataset.

            batch : dict
                A dictionary containing the input data for this hooked module, e.g., features and labels.

        Returns:
            None
        """
        assert "id" in batch
        self.batch_idx = batch_idx
        self.batch = batch
        self.outputs = defaultdict(list)

    def save(self, module_name: str, output: t.List[dict]) -> None:
        """
        Applies pooling operations on input data and saves them to disk or optionally returns them.

        The processed outputs are saved in torch pickle format at the specified location, with each file named after a specific
        combination of module_name and pooling operation name. These files can later be loaded back into memory for further use.

        Parameters:
            module_name : str
                Name of the parent module whose outputs are being hooked to.

            output : list[dict]
                The processed output from the parent module after applying pooling operations.

        Returns:
            None
        """
        for pooling_op in self.pooling_ops:
            attention_mask = self.batch["attention_mask"]
            pooled_output = pooling_op(output.detach().clone(), attention_mask=attention_mask)

            for sample_index in range(len(pooled_output)):
                datum = {}
                sample_id = self.batch["id"][sample_index]
                sample_outputs = pooled_output[sample_index].cpu()
                for field in self.save_fields:
                    datum[field] = self.batch[field][sample_index]
                if pooling_op.name == "all":
                    sample_outputs = sample_outputs[attention_mask[sample_index].bool()]
                datum.update(
                    {
                        "responses": sample_outputs.cpu(),
                    }
                )
                if self.output_path is not None:
                    output_path = (
                        self.output_path
                        / module_name
                        / pooling_op.name
                        / f"{sample_id}.pt"
                    )
                    os.makedirs(output_path.parent, exist_ok=True)
                    torch.save(datum, output_path)
                if self.return_outputs:
                    self.outputs[module_name].append(datum)

    def __call__(self, module, input, output) -> None:
        """
        Called when this hooked module's output changes.

        This method applies pooling operations to the inputs and saves them either on disk or optionally returns them.

        Parameters:
            module : torch.nn.Module
                The module whose output has changed.

            input : tuple or torch.Tensor or dict of them
                Input to this module.

            output : torch.Tensor
                Output from this module.

        Returns:
            None
        """
        assert (
            self.batch_idx is not None
        ), "update() must be called before executing the hook"

        def _hook(module_name: str, output: t.Any):
            if isinstance(output, torch.Tensor):
                self.save(module_name, output.detach())
            elif isinstance(output, (list, tuple)):
                for idx in range(len(output)):
                    _hook(f"{module_name}:{idx}", output[idx])
            else:
                logging.warn(f"Found {type(output)} in {self.module_name}")

        if self.threaded:
            self.thread_handle = threading.Thread(
                target=_hook, args=(self.module_name, output)
            )
            self.thread_handle.start()
        else:
            _hook(self.module_name, output)
