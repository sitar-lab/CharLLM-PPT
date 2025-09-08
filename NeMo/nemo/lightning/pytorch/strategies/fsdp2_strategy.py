# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import shutil
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Literal, Optional, Union

import lightning.pytorch as pl
import torch
from lightning.fabric.plugins import CheckpointIO
from lightning.pytorch.strategies.model_parallel import ModelParallelStrategy as PLModelParallelStrategy
from lightning.pytorch.trainer.states import TrainerFn
from lightning.pytorch.utilities.types import STEP_OUTPUT
from torch.distributed._composable.fsdp import MixedPrecisionPolicy
from torch.utils.data import DataLoader
from typing_extensions import override

from nemo.lightning import io
from nemo.lightning.pytorch.strategies.utils import (
    ckpt_to_dir,
    create_checkpoint_io,
    fsdp2_strategy_parallelize,
    setup_data_sampler,
)
from nemo.utils import logging

try:
    from torch.distributed.tensor._api import distribute_tensor
    from torch.distributed.tensor.placement_types import Shard
except ImportError:
    from torch.distributed._tensor.api import distribute_tensor
    from torch.distributed._tensor.placement_types import Shard


class FSDP2Strategy(PLModelParallelStrategy, io.IOMixin):
    """FSDP2Strategy implementing FSDP via FSDP 2.

    Notes:
    - TP + FSDP2 is currently not supported.
    """

    def __init__(
        self,
        data_parallel_size: Union[Literal["auto"], int] = "auto",
        tensor_parallel_size: Union[Literal["auto"], int] = "auto",
        data_sampler=None,
        checkpoint_io=None,
        mp_policy=None,
        parallelize_fn=None,
        **kwargs,
    ):
        """Initializes the FSDP2Strategy with specified parallelization settings.

        Args:
            data_parallel_size (Union[Literal["auto"], int]): Number of data-parallel replicas.
            tensor_parallel_size (Union[Literal["auto"], int]): Number of tensor-parallel groups.
            data_sampler (optional): Custom data sampler to process dataloaders.
            mp_policy (optional): Mixed precision policy for parameter and operation casting.
                Defaults to:
                ```python
                MixedPrecisionPolicy(
                    param_dtype=torch.bfloat16,
                    reduce_dtype=torch.float32,
                    output_dtype=None,
                    cast_forward_inputs=True,
                )
                ```
            parallelize_fn (callable, optional): Function for parallelizing the model. Defaults to None.
            **kwargs: Additional arguments for base class initialization.
        """
        super().__init__(data_parallel_size=data_parallel_size, tensor_parallel_size=tensor_parallel_size, **kwargs)
        self._checkpoint_io = checkpoint_io
        self.data_sampler = data_sampler
        self.checkpoint = None
        self.mp_policy = mp_policy
        if self.mp_policy is None:
            self.mp_policy = MixedPrecisionPolicy(
                param_dtype=torch.bfloat16,
                reduce_dtype=torch.float32,
                output_dtype=None,
                cast_forward_inputs=True,
            )
        self.parallelize_fn = parallelize_fn or fsdp2_strategy_parallelize

    @property
    @override
    def lightning_restore_optimizer(self) -> bool:
        """Optim state restoration is enabled"""
        return True

    def load_optimizer_state_dict(self, checkpoint) -> None:
        """Stores a reference to the optimizer state-dict for later restoration.

        Instead of immediately restoring the optimizer's state, this method saves the checkpoint
        reference and defers the restoration until the first training step. This is necessary
        because, in NeMo 2.0, PeFT adapters are added dynamically just before the first training
        step. Attempting to restore the optimizer state-dict before the adapters are initialized
        would result in an error.

        Args:
            checkpoint (dict): A dictionary containing the trainer's checkpoint,
                            including the optimizer state-dict.
        """
        # TODO(@akoumparouli): refactor.
        self.checkpoint = checkpoint

    def _load_optimizer_state_dict(self) -> None:
        """Restores the optimizer state-dict from the stored checkpoint.

        This method applies the optimizer state stored in the checkpoint to the corresponding
        optimizers. It ensures that the optimizer states are correctly restored after the
        PeFT adapters have been added in the first training step.

        If no checkpoint is stored, the method exits without performing any restoration.

        Note: This operation runs only once, as the checkpoint reference is cleared after execution.
        """
        from torch.distributed.checkpoint.state_dict import set_optimizer_state_dict

        if self.checkpoint is None:
            for optimizer, opt_state in zip(self.optimizers, self.checkpoint["optimizer_states"]):
                set_optimizer_state_dict(
                    self.lightning_module,
                    optimizer,
                    optim_state_dict=opt_state,
                    options={},
                )
            # run this only once
            self.checkpoint = None

    @override
    def setup_environment(self) -> None:
        """setup distributed environment and device mesh"""
        from torch.distributed.device_mesh import init_device_mesh

        self._setup_distributed()
        if self._data_parallel_size == "auto":
            self._data_parallel_size = self.num_nodes
        if self._tensor_parallel_size == "auto":
            self._tensor_parallel_size = self.num_processes
        # No TP currently
        self._device_mesh = init_device_mesh(
            device_type=self.root_device.type,
            mesh_shape=(self._data_parallel_size,),
            mesh_dim_names=("data_parallel",),
        )
        self.lightning_module._device_mesh = self._device_mesh

    @override
    def setup(self, trainer: pl.Trainer) -> None:
        """Configures the strategy within the PyTorch Lightning trainer.

        Args:
            trainer (pl.Trainer): The PyTorch Lightning trainer instance.
        """
        self.trainer = trainer
        setup_data_sampler(self.trainer)
        # connect trainer to accelerator.
        self.accelerator.setup(trainer)
        # Parallelize model
        if getattr(self, '_init_model_parallel', True):
            self.parallelize()
        # setup optim
        if getattr(self, '_setup_optimizers', True) and trainer.state.fn == TrainerFn.FITTING:
            super().setup_optimizers(trainer)

    def parallelize(self):
        """Applies fully_shard on model"""
        if self.parallelize_fn is not None:
            # TODO(@akoumparouli): self.lightning_module is an nn.Module child, use it directly?
            # Apply FSDP2 and TP to the model
            self.parallelize_fn(self.lightning_module.model, device_mesh=self._device_mesh, mp_policy=self.mp_policy)
            # Apply this only once
            self.parallelize_fn = None
        else:
            logging.warning("Called parallelize more than once.")

    def _get_loss_reduction(self, step_type: str):
        """Retrieves the loss reduction method for a given step type.

        Args:
            step_type (str): The type of step (e.g., "training", "validation").

        Returns:
            Callable: The loss reduction function, if defined; otherwise, None.
        """
        for fn_name in [f"{step_type}_loss_reduction", "loss_reduction"]:
            if hasattr(self.lightning_module, fn_name):
                return getattr(self.lightning_module, fn_name)
        return None

    def _step_proxy(self, step_type, batch, batch_idx=None):
        """Executes a training, validation, or test step and applies loss reduction if available.

        Args:
            step_type (str): The step type ("training", "validation", "test", "predict").
            batch: The input batch.
            batch_idx (optional): Index of the batch.

        Returns:
            Tuple: The computed loss and a dictionary with reduced loss metrics.
        """
        method_name = f"{step_type}_step"
        if self.model != self.lightning_module:
            loss = self._forward_redirection(self.model, self.lightning_module, method_name, batch, batch_idx)
        else:
            loss = getattr(self.lightning_module, method_name)(batch, batch_idx)

        _loss_reduction = self._get_loss_reduction(step_type)
        if _loss_reduction:
            return _loss_reduction.forward(batch, loss)
        return loss, {'avg': loss}

    @override
    def optimizer_state(self, optimizer):
        """returns the sharded optim state"""
        return optimizer.state_dict()

    @override
    def training_step(self, batch, batch_idx=None) -> STEP_OUTPUT:
        """Defines the training step, logging relevant metrics.

        Args:
            batch: The input batch.
            batch_idx (optional): The index of the batch.

        Returns:
            STEP_OUTPUT: The loss for backpropagation.
        """

        # See load_optimizer_state_dict to understand why we call this here.
        if self.checkpoint is not None:
            self._load_optimizer_state_dict()

        assert self.lightning_module is not None
        assert self.model is not None

        loss = self.lightning_module.training_step(batch, batch_idx)
        self.lightning_module.log(
            'global_step',
            self.trainer.global_step,
            prog_bar=True,
            rank_zero_only=True,
            batch_size=1,
        )

        self.lightning_module.log(
            'step',
            self.trainer.global_step,
        )

        return loss

    @override
    def validation_step(self, batch, batch_idx=None) -> Any:
        """Defines the validation step, logging validation loss.

        Args:
            batch: The input batch.
            batch_idx (optional): The index of the batch.

        Returns:
            Any: The validation loss.
        """
        assert self.lightning_module is not None
        assert self.model is not None
        with self.precision_plugin.val_step_context():
            loss, reduced = self._step_proxy("validation", batch, batch_idx)
            self.lightning_module.log('val_loss', reduced['avg'], rank_zero_only=True, batch_size=1)
            return loss

    @override
    def test_step(self, batch, batch_idx=None) -> STEP_OUTPUT:
        """Defines the test step, logging test loss.

        Args:
            batch: The input batch.
            batch_idx (optional): The index of the batch.

        Returns:
            Any: The test loss.
        """
        assert self.lightning_module is not None
        assert self.model is not None
        with self.precision_plugin.test_step_context():
            loss, reduced = self._step_proxy("test", batch, batch_idx)
            self.lightning_module.log('test_loss', reduced['avg'], rank_zero_only=True, batch_size=1)

            return loss

    @override
    def predict_step(self, batch, batch_idx=None) -> STEP_OUTPUT:
        """Runs one predict step.

        Args:
            batch (dict): the batch to use for pred.
            batch_idx (int, optional): the batch index. Defaults to None.

        Returns:
            STEP_OUTPUT: the reduced loss.
        """
        assert self.lightning_module is not None
        assert self.model is not None
        with self.precision_plugin.predict_step_context():
            loss, reduced = self._step_proxy("predict", batch, batch_idx)
            return reduced

    @override
    def process_dataloader(self, dataloader: DataLoader) -> DataLoader:
        """Applies data-samples to dataloader"""
        if self.data_sampler:
            return self.data_sampler.transform_dataloader(dataloader)

        return dataloader

    @contextmanager
    @override
    def tensor_init_context(self, empty_init: Optional[bool] = None):
        """Context manager used for initialization"""
        # Materializaton happens in `setup()`
        # @akoumparouli: using Parent's tensor_init_context causes mcore
        # parameters to be initialized on GPU instead of (assumed) CPU.
        yield

    @property
    @override
    def checkpoint_io(self) -> CheckpointIO:
        """CheckpointIO getter

        Returns:
            CheckpointIO: _description_
        """
        if self._checkpoint_io is None:
            self._checkpoint_io = create_checkpoint_io()

        return self._checkpoint_io

    @checkpoint_io.setter
    def checkpoint_io(self, io: CheckpointIO) -> None:
        """CheckpointIO setter

        Args:
            io (CheckpointIO): the checkpointio to use.
        """
        self._checkpoint_io = io

    @property
    def current_epoch_step(self) -> int:
        """Gets the current step within an epoch.

        Returns:
            int: The step index within the epoch.
        """
        return max(
            self.trainer.fit_loop.epoch_loop.automatic_optimization.optim_progress.optimizer.step.current.completed,
            self.trainer.fit_loop.epoch_loop.manual_optimization.optim_step_progress.current.completed,
        )

    @override
    def remove_checkpoint(self, filepath: Union[str, Path]) -> None:
        """Removes a checkpoint from the filesystem.

        Args:
            filepath (Union[str, Path]): Path to the checkpoint to be removed.
        """
        ckpt = ckpt_to_dir(filepath)
        if self.is_global_zero:
            if os.path.islink(ckpt):
                os.unlink(ckpt)
            elif os.path.exists(ckpt):
                shutil.rmtree(ckpt)

    @override
    def lightning_module_state_dict(self) -> Dict[str, Any]:
        """Collects the state dict of the model.

        Only returns a non-empty state dict on rank 0 if ``save_distributed_checkpoint=False``.

        """
        from nemo.lightning.pytorch.strategies.utils import to_cpu

        assert self.lightning_module is not None
        state_dict = self.lightning_module.state_dict()

        module_names = list(state_dict.keys())
        for name in module_names:
            param = state_dict.pop(name)
            state_dict[name] = to_cpu(param)
        return state_dict

    @override
    def save_checkpoint(
        self, checkpoint: Dict[str, Any], filepath: Union[str, Path], storage_options: Optional[Any] = None
    ) -> None:
        """
        Unshards FSDP2 checkpoint and passes it to checkpoint_io for saving to a file.
        """

        if self.is_global_zero:
            self.checkpoint_io.save_checkpoint(checkpoint, filepath, storage_options=storage_options)

    @override
    def load_checkpoint(self, checkpoint_path: str | Path) -> Dict[str, Any]:
        """Loads checkpoint with checkpoint_io"""
        return self.checkpoint_io.load_checkpoint(checkpoint_path)

    @override
    @torch.no_grad
    def load_model_state_dict(self, ckpt, strict=False):
        """Shards a full state dict"""
        # TODO(@akoumparouli): update `placements` value once TP is enabled.
        sharded_state = {
            k: distribute_tensor(v, self.device_mesh, placements=(Shard(dim=0),))
            for k, v in ckpt['state_dict'].items()
        }
        self.lightning_module.load_state_dict(sharded_state, strict=strict)
