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

#!/usr/bin/env python
"""Train a radiance field with nerfstudio.
For real captures, we recommend using the [bright_yellow]nerfacto[/bright_yellow] model.

Nerfstudio allows for customizing your training and eval configs from the CLI in a powerful way, but there are some
things to understand.

The most demonstrative and helpful example of the CLI structure is the difference in output between the following
commands:

    ns-train -h
    ns-train nerfacto -h nerfstudio-data
    ns-train nerfacto nerfstudio-data -h

In each of these examples, the -h applies to the previous subcommand (ns-train, nerfacto, and nerfstudio-data).

In the first example, we get the help menu for the ns-train script.
In the second example, we get the help menu for the nerfacto model.
In the third example, we get the help menu for the nerfstudio-data dataparser.

With our scripts, your arguments will apply to the preceding subcommand in your command, and thus where you put your
arguments matters! Any optional arguments you discover from running

    ns-train nerfacto -h nerfstudio-data

need to come directly after the nerfacto subcommand, since these optional arguments only belong to the nerfacto
subcommand:

    ns-train nerfacto {nerfacto optional args} nerfstudio-data
"""

from __future__ import annotations

import random
import socket
import traceback
from datetime import timedelta
from typing import Any, Callable, Literal, Optional

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import tyro
import yaml

from nerfstudio.configs.config_utils import convert_markup_to_ansi
from nerfstudio.configs.method_configs import AnnotatedBaseConfigUnion
from nerfstudio.engine.trainer import TrainerConfig
from nerfstudio.utils import comms, profiler
from nerfstudio.utils.rich_utils import CONSOLE

import os
from pathlib import Path
from nerfstudio.engine.multiple_fitting_trainer import MultipleFittingTrainerConfig
from nerfstudio.data.datasets.base_dataset import InputDataset
from nerfstudio.utils.comms import get_rank, get_world_size, is_main_process

DEFAULT_TIMEOUT = timedelta(minutes=30)

# speedup for when input size to model doesn't change (much)
torch.backends.cudnn.benchmark = True  # type: ignore


def _find_free_port() -> str:
    """Finds a free port."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(("", 0))
    port = sock.getsockname()[1]
    sock.close()
    return port


def _set_random_seed(seed) -> None:
    """Set randomness seed in torch and numpy"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def _find_subject_ckpt(config, subject):
    # current/shared config
    if config.epoch == 0:
        return None
    output_dir = config.output_dir
    method_name = config.method_name
    max_num_iterations = config.max_num_iterations
    max_num_outer_epochs = config.max_num_outer_epochs

    # current subject of previous epoch
    experiment_name = subject + f"_{max_num_outer_epochs}-{config.epoch-1}"

    method_dir = Path(f"{output_dir}/{experiment_name}/{method_name}")
    try:
        timestamp_dirs = sorted(os.listdir(method_dir), key=lambda x: int(x.replace("-", "").replace("_", "")), reverse=True)
        for timestamp_dir in timestamp_dirs:
            checkpoint_dir = Path(timestamp_dir) / "nerfstudio_models"
            checkpoint_path = method_dir / checkpoint_dir / f"step-{max_num_iterations - 1:09d}.ckpt"
            if checkpoint_path.exists():
                return checkpoint_path
            else:
                continue
        return None
    except:
        return None
    
def _find_decoder_ckpt(config, subject, last_sharding_subjects):
    # current/shared config
    output_dir = config.output_dir
    method_name = config.method_name
    max_num_iterations = config.max_num_iterations
    max_num_outer_epochs = config.max_num_outer_epochs

    if subject in last_sharding_subjects:
        # turn to previous epoch        
        experiment_name = subject + f"_{max_num_outer_epochs}-{config.epoch-1}"
    else:
        # previous subject of current epoch
        experiment_name = subject + f"_{max_num_outer_epochs}-{config.epoch}"

    method_dir = Path(f"{output_dir}/{experiment_name}/{method_name}")
    try:
        timestamp_dirs = sorted(os.listdir(method_dir), key=lambda x: int(x.replace("-", "").replace("_", "")), reverse=True)
        for timestamp_dir in timestamp_dirs:
            checkpoint_dir = Path(timestamp_dir) / "nerfstudio_models"
            checkpoint_path = method_dir / checkpoint_dir / f"step-{max_num_iterations - 1:09d}.ckpt"
            if checkpoint_path.exists():
                return checkpoint_path
            else:
                continue
        return None
    except:
        return None

def train_loop(local_rank: int, world_size: int, config: MultipleFittingTrainerConfig, global_rank: int = 0):
    """Main training function that sets up and runs the trainer per process

    Args:
        local_rank: current rank of process
        world_size: total number of gpus available
        config: config file specifying training regimen
    """

    if config.num_train_subjects > 1:
        config.pipeline.model.multiple_fitting = True # type: ignore

    if config.num_train_subjects == 10:
        with open("/mnt/blob/data/rodin/data_10persons.txt", "r") as f:
            ALL_SUBJECT = f.read().splitlines()
    elif config.num_train_subjects == 15:
        with open("/mnt/blob/data/rodin/data_15persons.txt", "r") as f:
            ALL_SUBJECT = f.read().splitlines()
    else:
        with open("/mnt/blob2/avatar/person_list/2.18_hd.txt", 'r') as f:
            ALL_SUBJECT = f.read().splitlines()[:config.num_train_subjects]

    # subject sharding
    assert len(ALL_SUBJECT) % world_size == 0
    shard_subject = ALL_SUBJECT[local_rank:][::world_size]
    CONSOLE.print(f"{local_rank}/{world_size}: {shard_subject}")

    # global initialization
    global_step = 0
    config.max_num_global_iterations = int(config.max_num_outer_epochs * (len(ALL_SUBJECT) / world_size) * (config.max_num_iterations-1))
    config.optimizers["field.sigma_net"]["scheduler"].max_steps = config.max_num_global_iterations
    config.optimizers["field.color_net"]["scheduler"].max_steps = config.max_num_global_iterations
    config.optimizers["field.grids"]["scheduler"].max_steps = config.max_num_global_iterations
    
    # trainer
    _set_random_seed(config.machine.seed + global_rank)
    trainer = config.setup(local_rank=local_rank, world_size=world_size) # trainer.__init__
    trainer.setup()

    # previous_subject: Optional[str] = None
    for epoch in range(config.max_num_outer_epochs):
        for subject in shard_subject:
            # experiment initialization
            config.set_timestamp()
            config.epoch = epoch
            config.experiment_name = subject + f"_{config.max_num_outer_epochs}-{epoch}"

            # class varibale initialization
            InputDataset.exclude_batch_keys_from_device = ['image', 'mask']

            config.data = Path("/mnt/blob2/render_output_hd/")
            assert config.data is not None
            if config.data:
                CONSOLE.log("Using --data alias for [yellow]directory[/yellow] of --data.pipeline.datamanager.data")
                config.pipeline.datamanager.data = config.data / subject # type: ignore
                config.pipeline.datamanager.dataparser.data = config.pipeline.datamanager.data # type: ignore

            if config.prompt:
                CONSOLE.log("Using --prompt alias for --data.pipeline.model.prompt")
                config.pipeline.model.prompt = config.prompt

            if config.load_config:
                CONSOLE.log(f"Loading pre-set config from: {config.load_config}")
                config = yaml.load(config.load_config.read_text(), Loader=yaml.Loader)

            subject_ckpt = _find_subject_ckpt(config, subject)
            if subject_ckpt is not None:
                config.pipeline.model.subject_checkpoint = subject_ckpt # type: ignore

            # if previous_subject is not None:
            #     decoder_ckpt = _find_decoder_ckpt(config, previous_subject, ALL_SUBJECT[-get_world_size():])
            #     if decoder_ckpt is not None:
            #         config.pipeline.model.decoder_checkpoint = decoder_ckpt # type: ignore
            #         config.decoder_optimizer_checkpoint = decoder_ckpt
            # previous_subject = subject

            # print and save config
            if is_main_process():
                config.print_to_terminal()
            config.save_config()
            # BUG: configs of some modules are setup after config.setup / trainer.setup, 
            # so save_config will give fake module config (training is right).
            # However, due to these config are class variable, so they become right  
            # after the next loop.

            _set_random_seed(config.machine.seed + global_rank + global_step)

            trainer.reinit(config)
            trainer.resetup()
            trainer.train()


def _distributed_worker(
    local_rank: int,
    main_func: Callable,
    world_size: int,
    num_devices_per_machine: int,
    machine_rank: int,
    dist_url: str,
    config: TrainerConfig,
    timeout: timedelta = DEFAULT_TIMEOUT,
    device_type: Literal["cpu", "cuda", "mps"] = "cuda",
) -> Any:
    """Spawned distributed worker that handles the initialization of process group and handles the
       training process on multiple processes.

    Args:
        local_rank: Current rank of process.
        main_func: Function that will be called by the distributed workers.
        world_size: Total number of gpus available.
        num_devices_per_machine: Number of GPUs per machine.
        machine_rank: Rank of this machine.
        dist_url: URL to connect to for distributed jobs, including protocol
            E.g., "tcp://127.0.0.1:8686".
            It can be set to "auto" to automatically select a free port on localhost.
        config: TrainerConfig specifying training regimen.
        timeout: Timeout of the distributed workers.

    Raises:
        e: Exception in initializing the process group

    Returns:
        Any: TODO: determine the return type
    """
    assert torch.cuda.is_available(), "cuda is not available. Please check your installation."
    global_rank = machine_rank * num_devices_per_machine + local_rank

    dist.init_process_group(
        backend="nccl" if device_type == "cuda" else "gloo",
        init_method=dist_url,
        world_size=world_size,
        rank=global_rank,
        timeout=timeout,
    )
    assert comms.LOCAL_PROCESS_GROUP is None
    num_machines = world_size // num_devices_per_machine
    for i in range(num_machines):
        ranks_on_i = list(range(i * num_devices_per_machine, (i + 1) * num_devices_per_machine))
        pg = dist.new_group(ranks_on_i)
        if i == machine_rank:
            comms.LOCAL_PROCESS_GROUP = pg

    assert num_devices_per_machine <= torch.cuda.device_count()
    output = main_func(local_rank, world_size, config, global_rank)
    comms.synchronize()
    dist.destroy_process_group()
    return output


def launch(
    main_func: Callable,
    num_devices_per_machine: int,
    num_machines: int = 1,
    machine_rank: int = 0,
    dist_url: str = "auto",
    config: Optional[TrainerConfig] = None,
    timeout: timedelta = DEFAULT_TIMEOUT,
    device_type: Literal["cpu", "cuda", "mps"] = "cuda",
) -> None:
    """Function that spawns multiple processes to call on main_func

    Args:
        main_func (Callable): function that will be called by the distributed workers
        num_devices_per_machine (int): number of GPUs per machine
        num_machines (int, optional): total number of machines
        machine_rank (int, optional): rank of this machine.
        dist_url (str, optional): url to connect to for distributed jobs.
        config (TrainerConfig, optional): config file specifying training regimen.
        timeout (timedelta, optional): timeout of the distributed workers.
        device_type: type of device to use for training.
    """
    assert config is not None
    world_size = num_machines * num_devices_per_machine
    if world_size == 0:
        raise ValueError("world_size cannot be 0")
    elif world_size == 1:
        # uses one process
        try:
            main_func(local_rank=0, world_size=world_size, config=config)
        except KeyboardInterrupt:
            # print the stack trace
            CONSOLE.print(traceback.format_exc())
        finally:
            profiler.flush_profiler(config.logging)
    elif world_size > 1:
        # Using multiple gpus with multiple processes.
        if dist_url == "auto":
            assert num_machines == 1, "dist_url=auto is not supported for multi-machine jobs."
            port = _find_free_port()
            dist_url = f"tcp://127.0.0.1:{port}"
        if num_machines > 1 and dist_url.startswith("file://"):
            CONSOLE.log("file:// is not a reliable init_method in multi-machine jobs. Prefer tcp://")

        process_context = mp.spawn(
            _distributed_worker,
            nprocs=num_devices_per_machine,
            join=False,
            args=(main_func, world_size, num_devices_per_machine, machine_rank, dist_url, config, timeout, device_type),
        )
        # process_context won't be None because join=False, so it's okay to assert this
        # for Pylance reasons
        assert process_context is not None
        try:
            process_context.join()
        except KeyboardInterrupt:
            for i, process in enumerate(process_context.processes):
                if process.is_alive():
                    CONSOLE.log(f"Terminating process {i}...")
                    process.terminate()
                process.join()
                CONSOLE.log(f"Process {i} finished.")
        finally:
            profiler.flush_profiler(config.logging)


def main(config: TrainerConfig) -> None:
    """Main function."""

    # config.set_timestamp()
    # if config.data:
    #     CONSOLE.log("Using --data alias for --data.pipeline.datamanager.data")
    #     config.pipeline.datamanager.data = config.data

    # if config.prompt:
    #     CONSOLE.log("Using --prompt alias for --data.pipeline.model.prompt")
    #     config.pipeline.model.prompt = config.prompt

    # if config.load_config:
    #     CONSOLE.log(f"Loading pre-set config from: {config.load_config}")
    #     config = yaml.load(config.load_config.read_text(), Loader=yaml.Loader)

    # # print and save config
    # config.print_to_terminal()
    # config.save_config()

    launch(
        main_func=train_loop,
        num_devices_per_machine=config.machine.num_devices,
        device_type=config.machine.device_type,
        num_machines=config.machine.num_machines,
        machine_rank=config.machine.machine_rank,
        dist_url=config.machine.dist_url,
        config=config,
    )


def entrypoint():
    """Entrypoint for use with pyproject scripts."""
    # Choose a base configuration and override values.
    tyro.extras.set_accent_color("bright_yellow")
    main(
        tyro.cli(
            AnnotatedBaseConfigUnion,
            description=convert_markup_to_ansi(__doc__),
        )
    )


if __name__ == "__main__":
    entrypoint()
