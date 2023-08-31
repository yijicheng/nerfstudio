# Copyright 2022 the Regents of the University of California, Nerfstudio Team and contributors. All rights reserved.
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

"""
Code to train model.
"""
from __future__ import annotations

import dataclasses
import functools
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from threading import Lock
from typing import Dict, List, Literal, Optional, Tuple, Type, cast

import torch
from rich import box, style
from rich.panel import Panel
from rich.table import Table
from torch.cuda.amp.grad_scaler import GradScaler

from nerfstudio.configs.experiment_config import ExperimentConfig
from nerfstudio.data.datamanagers.base_datamanager import VanillaDataManager
from nerfstudio.engine.callbacks import TrainingCallback, TrainingCallbackAttributes, TrainingCallbackLocation
from nerfstudio.engine.optimizers import Optimizers
from nerfstudio.pipelines.base_pipeline import VanillaPipeline
from nerfstudio.utils import profiler, writer
from nerfstudio.utils.decorators import check_eval_enabled, check_main_thread, check_viewer_enabled
from nerfstudio.utils.misc import step_check
from nerfstudio.utils.rich_utils import CONSOLE
from nerfstudio.utils.writer import EventName, TimeWriter
from nerfstudio.viewer.server.viewer_state import ViewerState
from nerfstudio.viewer_beta.viewer import Viewer as ViewerBetaState

from nerfstudio.engine.trainer import TrainerConfig
from nerfstudio.engine.trainer import Trainer
from nerfstudio.utils.comms import is_main_process, get_rank
import time

TRAIN_INTERATION_OUTPUT = Tuple[torch.Tensor, Dict[str, torch.Tensor], Dict[str, torch.Tensor]]
TORCH_DEVICE = str

@dataclass
class MultipleFittingTrainerConfig(TrainerConfig):
    """Template Trainer Configuration.

    Add your custom trainer config parameters here.
    
    """

    _target: Type = field(default_factory=lambda: MultipleFittingTrainer)
    """target class to instantiate"""
    max_num_global_iterations: Optional[int] = None
    """Maximum number of gloabl iterations to run."""
    max_num_outer_epochs: int = 1
    """Maximum number of outer epochs to run."""
    epoch: int = 0
    """Current epoch."""
    decoder_optimizer_checkpoint: Optional[Path] = None
    """Decoder optimizer, scheduler, gradient scalar checkpoint"""
    num_train_subjects: int = 10
    """Number of training subjects"""


    stage1_load_dir: Optional[str] = None
    stage1_max_num_iterations: Optional[int] = None
    stage1_max_num_outer_epochs: Optional[int] = None
    stage1_decoder_subject: Optional[str] = None



class MultipleFittingTrainer(Trainer):
    """Trainer class

    Args:
        config: The configuration object.
        local_rank: Local rank of the process.
        world_size: World size of the process.

    Attributes:
        config: The configuration object.
        local_rank: Local rank of the process.
        world_size: World size of the process.
        device: The device to run the training on.
        pipeline: The pipeline object.
        optimizers: The optimizers object.
        callbacks: The callbacks object.
        training_state: Current model training state.
    """

    pipeline: VanillaPipeline
    optimizers: Optimizers
    callbacks: List[TrainingCallback]

    def __init__(self, config: MultipleFittingTrainerConfig, local_rank: int = 0, world_size: int = 1) -> None:
        super().__init__(config, local_rank, world_size)
        self.global_step = 0

    def setup(self, test_mode: Literal["test", "val", "inference"] = "val") -> None:
        """Setup the Trainer by calling other setup functions.

        Args:
            test_mode:
                'val': loads train/val datasets into memory
                'test': loads train/test datasets into memory
                'inference': does not load any dataset into memory
        """
        self.pipeline = self.config.pipeline.setup(
            device=self.device,
            test_mode=test_mode,
            world_size=self.world_size,
            local_rank=self.local_rank,
            grad_scaler=self.grad_scaler,
        )
        self.optimizers = self.setup_optimizers()

        assert not (self.config.is_viewer_enabled() or self.config.is_viewer_beta_enabled())
        # # set up viewer if enabled
        # viewer_log_path = self.base_dir / self.config.viewer.relative_log_filename
        # self.viewer_state, banner_messages = None, None
        # if self.config.is_viewer_enabled() and self.local_rank == 0:
        #     datapath = self.config.data
        #     if datapath is None:
        #         datapath = self.base_dir
        #     self.viewer_state = ViewerState(
        #         self.config.viewer,
        #         log_filename=viewer_log_path,
        #         datapath=datapath,
        #         pipeline=self.pipeline,
        #         trainer=self,
        #         train_lock=self.train_lock,
        #     )
        #     banner_messages = [f"Viewer at: {self.viewer_state.viewer_url}"]
        # if self.config.is_viewer_beta_enabled() and self.local_rank == 0:
        #     self.viewer_state = ViewerBetaState(
        #         self.config.viewer,
        #         log_filename=viewer_log_path,
        #         datapath=self.base_dir,
        #         pipeline=self.pipeline,
        #         trainer=self,
        #         train_lock=self.train_lock,
        #     )
        #     banner_messages = [f"Viewer Beta at: {self.viewer_state.viewer_url}"]
        # self._check_viewer_warnings()

        # self._load_checkpoint()

        # self.callbacks = self.pipeline.get_training_callbacks(
        #     TrainingCallbackAttributes(
        #         optimizers=self.optimizers,
        #         grad_scaler=self.grad_scaler,
        #         pipeline=self.pipeline,
        #     )
        # )

        # # set up writers/profilers if enabled
        # writer_log_path = self.base_dir / self.config.logging.relative_log_dir
        # writer.setup_event_writer(
        #     self.config.is_wandb_enabled(),
        #     self.config.is_tensorboard_enabled(),
        #     log_dir=writer_log_path,
        #     experiment_name=self.config.experiment_name,
        #     project_name=self.config.project_name,
        # )
        # writer.setup_local_writer(
        #     self.config.logging, max_iter=self.config.max_num_iterations, banner_messages=banner_messages
        # )
        # writer.put_config(name="config", config_dict=dataclasses.asdict(self.config), step=0)
        # profiler.setup_profiler(self.config.logging, writer_log_path)

    def reinit(self, config) -> None:
        self.config = config
        self._start_step: int = 0
        
        self.base_dir: Path = config.get_base_dir()
        # directory to save checkpoints
        self.checkpoint_dir: Path = config.get_checkpoint_dir()
        CONSOLE.log(f"Saving checkpoints to: {self.checkpoint_dir}")

    def resetup(self) -> None:
        s = time.time()
        self.pipeline.config = self.config.pipeline
        self.pipeline.datamanager = self.config.pipeline.datamanager.setup(
            device=self.device, test_mode=self.pipeline.test_mode, world_size=self.world_size, local_rank=self.local_rank
        )
        self.pipeline.datamanager.to(self.device)
        CONSOLE.print(f"datamanager: {time.time() - s}")

        s = time.time()
        if self.config.pipeline.model.decoder_checkpoint is not None or self.config.pipeline.model.subject_checkpoint is not None: # type: ignore
            assert self.config.load_dir is None and self.config.load_dir is None
        self.pipeline.model.scene_box = self.pipeline.datamanager.train_dataset.scene_box
        self.pipeline.model.num_train_data = len(self.pipeline.datamanager.train_dataset)
        self.pipeline.model.metadata = self.pipeline.datamanager.train_dataset.metadata
        self.pipeline.model._reload_checkpoint() # type: ignore

        # self._reload_optimizer()
        if self.config.pipeline.model.freeze_decoder: # type: ignore
            self.optimizers.optimizers.pop('field.sigma_net')
            self.optimizers.schedulers.pop('field.sigma_net')
            self.optimizers.optimizers.pop('field.color_net')
            self.optimizers.schedulers.pop('field.color_net')

        self.callbacks = self.pipeline.get_training_callbacks(
            TrainingCallbackAttributes(
                optimizers=self.optimizers,
                grad_scaler=self.grad_scaler,
                pipeline=self.pipeline,
            )
        )
        CONSOLE.print(f"model: {time.time() - s}")

        s = time.time()
        # set up writers/profilers if enabled
        banner_messages = None
        writer_log_path = self.base_dir / self.config.logging.relative_log_dir
        writer.setup_event_writer(
            self.config.is_wandb_enabled(),
            self.config.is_tensorboard_enabled(),
            log_dir=writer_log_path,
            experiment_name=self.config.experiment_name,
            project_name=self.config.project_name,
        )
        writer.setup_local_writer(
            self.config.logging, max_iter=self.config.max_num_iterations, banner_messages=banner_messages
        )
        writer.put_config(name="config", config_dict=dataclasses.asdict(self.config), step=0)
        profiler.setup_profiler(self.config.logging, writer_log_path)
        CONSOLE.print(f"writer: {time.time() - s}")

    def _reload_optimizer(self) -> None:
        load_decoder_optimizer = self.config.decoder_optimizer_checkpoint
        if load_decoder_optimizer is not None:
            assert self.config.load_dir is None and self.config.load_dir is None
            assert load_decoder_optimizer.exists(), f"Checkpoint {load_decoder_optimizer} does not exist"
            loaded_state = torch.load(load_decoder_optimizer, map_location="cpu")
            # self._start_step = loaded_state["step"] + 1 # BUG
            # load the checkpoints for optimizers, and gradient scalar
            self.optimizers.load_optimizers(loaded_state["optimizers"])
            if "schedulers" in loaded_state and self.config.load_scheduler:
                self.optimizers.load_schedulers(loaded_state["schedulers"])
            self.grad_scaler.load_state_dict(loaded_state["scalers"])
            del loaded_state
            CONSOLE.print(f"Done loading optimizer checkpoint from {load_decoder_optimizer}")
        else:
            CONSOLE.print("No optimizer checkpoint to load, so training from scratch.")

    def setup_optimizers(self) -> Optimizers:
        """Helper to set up the optimizers

        Returns:
            The optimizers object given the trainer config.
        """
        optimizer_config = self.config.optimizers.copy()
        param_groups = self.pipeline.get_param_groups()
        camera_optimizer_config = self.config.pipeline.datamanager.camera_optimizer
        if camera_optimizer_config is not None and camera_optimizer_config.mode != "off":
            assert camera_optimizer_config.param_group not in optimizer_config
            optimizer_config[camera_optimizer_config.param_group] = {
                "optimizer": camera_optimizer_config.optimizer,
                "scheduler": camera_optimizer_config.scheduler,
            }
        return Optimizers(optimizer_config, param_groups)

    def train(self) -> None:
        """Train the model."""
        assert self.pipeline.datamanager.train_dataset is not None, "Missing DatsetInputs"

        # don't want to call save_dataparser_transform if pipeline's datamanager does not have a dataparser
        if isinstance(self.pipeline.datamanager, VanillaDataManager):
            self.pipeline.datamanager.train_dataparser_outputs.save_dataparser_transform(
                self.base_dir / "dataparser_transforms.json"
            )

        self._init_viewer_state()
        with TimeWriter(writer, EventName.TOTAL_TRAIN_TIME):
            num_iterations = self.config.max_num_iterations
            step = 0
            for step in range(self._start_step, self._start_step + num_iterations):
                while self.training_state == "paused":
                    time.sleep(0.01)
                with self.train_lock:
                    with TimeWriter(writer, EventName.ITER_TRAIN_TIME, step=step) as train_t:
                        self.pipeline.train()

                        # training callbacks before the training iteration
                        for callback in self.callbacks:
                            callback.run_callback_at_location(
                                step, location=TrainingCallbackLocation.BEFORE_TRAIN_ITERATION
                            )

                        # time the forward pass
                        loss, loss_dict, metrics_dict = self.train_iteration(step)

                        # training callbacks after the training iteration
                        for callback in self.callbacks:
                            callback.run_callback_at_location(
                                step, location=TrainingCallbackLocation.AFTER_TRAIN_ITERATION
                            )

                # Skip the first two steps to avoid skewed timings that break the viewer rendering speed estimate.
                if step > 1:
                    writer.put_time(
                        name=EventName.TRAIN_RAYS_PER_SEC, # type: ignore
                        duration=self.world_size
                        * self.pipeline.datamanager.get_train_rays_per_batch()
                        / max(0.001, train_t.duration),
                        step=step,
                        avg_over_steps=True,
                    )

                self._update_viewer_state(step)

                # a batch of train rays
                if step_check(step, self.config.logging.steps_per_log, run_at_zero=True):
                    writer.put_scalar(name="Train Loss", scalar=loss, step=self.global_step)
                    writer.put_dict(name="Train Loss Dict", scalar_dict=loss_dict, step=self.global_step)
                    writer.put_dict(name="Train Metrics Dict", scalar_dict=metrics_dict, step=self.global_step)
                    # The actual memory allocated by Pytorch. This is likely less than the amount
                    # shown in nvidia-smi since some unused memory can be held by the caching
                    # allocator and some context needs to be created on GPU. See Memory management
                    # (https://pytorch.org/docs/stable/notes/cuda.html#cuda-memory-management)
                    # for more details about GPU memory management.
                    writer.put_scalar(
                        name="GPU Memory (MB)", scalar=torch.cuda.max_memory_allocated(get_rank()) / (1024**2), step=self.global_step
                    )

                # Do not perform evaluation if there are no validation images
                if self.pipeline.datamanager.eval_dataset:
                    self.eval_iteration(step)

                if step_check(step, self.config.steps_per_save):
                    self.save_checkpoint(step)

                writer.write_out_storage()
                
                self.global_step += 1

        # 
        self.global_step -= 1

        # save checkpoint at the end of training
        self.save_checkpoint(step)

        # write out any remaining events (e.g., total train time)
        writer.write_out_storage()
        writer.close_event_writer()

        table = Table(
            title=None,
            show_header=False,
            box=box.MINIMAL,
            title_style=style.Style(bold=True),
        )
        table.add_row("Config File", str(self.config.get_base_dir() / "config.yml"))
        table.add_row("Checkpoint Directory", str(self.checkpoint_dir))
        CONSOLE.print(Panel(table, title="[bold][green]:tada: Training Finished :tada:[/bold]", expand=False))

        # after train end callbacks
        for callback in self.callbacks:
            callback.run_callback_at_location(step=step, location=TrainingCallbackLocation.AFTER_TRAIN)

        if not self.config.viewer.quit_on_train_completion:
            self._train_complete_viewer()

    def _load_checkpoint(self) -> None:
        """Helper function to load pipeline and optimizer from prespecified checkpoint"""
        load_dir = self.config.load_dir
        load_checkpoint = self.config.load_checkpoint
        if load_dir is not None:
            load_step = self.config.load_step
            if load_step is None:
                print("Loading latest Nerfstudio checkpoint from load_dir...")
                # NOTE: this is specific to the checkpoint name format
                load_step = sorted(int(x[x.find("-") + 1 : x.find(".")]) for x in os.listdir(load_dir))[-1]
            load_path: Path = load_dir / f"step-{load_step:09d}.ckpt"
            assert load_path.exists(), f"Checkpoint {load_path} does not exist"
            loaded_state = torch.load(load_path, map_location="cpu")
            self._start_step = loaded_state["step"] + 1
            # load the checkpoints for pipeline, optimizers, and gradient scalar
            self.pipeline.load_pipeline(loaded_state["pipeline"], loaded_state["step"])
            self.optimizers.load_optimizers(loaded_state["optimizers"])
            if "schedulers" in loaded_state and self.config.load_scheduler:
                self.optimizers.load_schedulers(loaded_state["schedulers"])
            self.grad_scaler.load_state_dict(loaded_state["scalers"])
            CONSOLE.print(f"Done loading Nerfstudio checkpoint from {load_path}")
        elif load_checkpoint is not None:
            assert load_checkpoint.exists(), f"Checkpoint {load_checkpoint} does not exist"
            loaded_state = torch.load(load_checkpoint, map_location="cpu")
            self._start_step = loaded_state["step"] + 1
            # load the checkpoints for pipeline, optimizers, and gradient scalar
            self.pipeline.load_pipeline(loaded_state["pipeline"], loaded_state["step"])
            self.optimizers.load_optimizers(loaded_state["optimizers"])
            if "schedulers" in loaded_state and self.config.load_scheduler:
                self.optimizers.load_schedulers(loaded_state["schedulers"])
            self.grad_scaler.load_state_dict(loaded_state["scalers"])
            CONSOLE.print(f"Done loading Nerfstudio checkpoint from {load_checkpoint}")
        else:
            CONSOLE.print("No Nerfstudio checkpoint to load, so training from scratch.")

    # @check_main_thread
    def save_checkpoint(self, step: int) -> None:
        """Save the model and optimizers

        Args:
            step: number of steps in training for given checkpoint
        """
        # possibly make the checkpoint directory
        if not self.checkpoint_dir.exists():
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        # save the checkpoint
        ckpt_path: Path = self.checkpoint_dir / f"step-{step:09d}.ckpt"
        torch.save(
            {
                "step": step,
                "global_step": self.global_step,
                "pipeline": self.pipeline.module.state_dict()  # type: ignore
                if hasattr(self.pipeline, "module")
                else self.pipeline.state_dict(),
                "optimizers": {k: v.state_dict() for (k, v) in self.optimizers.optimizers.items()},
                "schedulers": {k: v.state_dict() for (k, v) in self.optimizers.schedulers.items()},
                "scalers": self.grad_scaler.state_dict(),
            },
            ckpt_path,
        )
        # possibly delete old checkpoints
        if self.config.save_only_latest_checkpoint:
            # delete everything else in the checkpoint folder
            for f in self.checkpoint_dir.glob("*"):
                if f != ckpt_path:
                    f.unlink()

    @profiler.time_function
    def train_iteration(self, step: int) -> TRAIN_INTERATION_OUTPUT:
        """Run one iteration with a batch of inputs. Returns dictionary of model losses.

        Args:
            step: Current training step.
        """

        self.optimizers.zero_grad_all()
        cpu_or_cuda_str: str = self.device.split(":")[0]
        assert (
            self.gradient_accumulation_steps > 0
        ), f"gradient_accumulation_steps must be > 0, not {self.gradient_accumulation_steps}"
        for _ in range(self.gradient_accumulation_steps):
            with torch.autocast(device_type=cpu_or_cuda_str, enabled=self.mixed_precision):
                _, loss_dict, metrics_dict = self.pipeline.get_train_loss_dict(step=step)
                loss = functools.reduce(torch.add, loss_dict.values())
                loss /= self.gradient_accumulation_steps
            self.grad_scaler.scale(loss).backward()  # type: ignore
        self.optimizers.optimizer_scaler_step_all(self.grad_scaler)

        if self.config.log_gradients:
            total_grad = 0
            for tag, value in self.pipeline.model.named_parameters():
                assert tag != "Total"
                if value.grad is not None:
                    grad = value.grad.norm()
                    weight = value.norm()
                    metrics_dict[f"Gradients/{tag}"] = grad  # type: ignore
                    metrics_dict[f"Weights/{tag}"]  = weight  # type: ignore
                    total_grad += grad

                    if grad.isinf():
                        CONSOLE.print(f"[red]Gradients/{tag}: inf[/red]")
                    if grad.isnan():
                        CONSOLE.print(f"[red]Gradients/{tag}: nan[/red]")

                    if weight.isinf():
                        CONSOLE.print(f"[red]Weights/{tag}: inf[/red]")
                    if weight.isnan():
                        CONSOLE.print(f"[red]Weights/{tag}: nan[/red]")

            metrics_dict["Gradients/Total"] = cast(torch.Tensor, total_grad)  # type: ignore

        scale = self.grad_scaler.get_scale()
        self.grad_scaler.update()
        metrics_dict["grad_scaler"] = scale # type: ignore
        # If the gradient scaler is decreased, no optimization step is performed so we should not step the scheduler.
        if scale <= self.grad_scaler.get_scale():
            self.optimizers.scheduler_step_all(self.global_step)

        # Merging loss and metrics dict into a single output.
        return loss, loss_dict, metrics_dict  # type: ignore

    @check_eval_enabled
    @profiler.time_function
    def eval_iteration(self, step: int) -> None:
        """Run one iteration with different batch/image/all image evaluations depending on step size.

        Args:
            step: Current training step.
        """
        # a batch of eval rays
        if step_check(step, self.config.steps_per_eval_batch):
            _, eval_loss_dict, eval_metrics_dict = self.pipeline.get_eval_loss_dict(step=step)
            eval_loss = functools.reduce(torch.add, eval_loss_dict.values())
            writer.put_scalar(name="Eval Loss", scalar=eval_loss, step=self.global_step)
            writer.put_dict(name="Eval Loss Dict", scalar_dict=eval_loss_dict, step=self.global_step)
            writer.put_dict(name="Eval Metrics Dict", scalar_dict=eval_metrics_dict, step=self.global_step)

        # one eval image
        if step_check(step, self.config.steps_per_eval_image):
            with TimeWriter(writer, EventName.TEST_RAYS_PER_SEC, write=False) as test_t:
                metrics_dict, images_dict = self.pipeline.get_eval_image_metrics_and_images(step=step)
            writer.put_time(
                name=EventName.TEST_RAYS_PER_SEC, # type: ignore
                duration=metrics_dict["num_rays"] / test_t.duration,
                step=self.global_step,
                avg_over_steps=True,
            )
            writer.put_dict(name="Eval Images Metrics", scalar_dict=metrics_dict, step=self.global_step)
            group = "Eval Images"
            for image_name, image in images_dict.items():
                writer.put_image(name=group + "/" + image_name, image=image, step=self.global_step)

        # all eval images
        if step_check(step, self.config.steps_per_eval_all_images):
            metrics_dict = self.pipeline.get_average_eval_image_metrics(step=step)
            writer.put_dict(name="Eval Images Metrics Dict (all images)", scalar_dict=metrics_dict, step=self.global_step)
