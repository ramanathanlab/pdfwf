"""Parsl configuration for PDF conversion workflow."""

from __future__ import annotations

from typing import Any

from parsl.addresses import address_by_interface
from parsl.config import Config
from parsl.executors import HighThroughputExecutor
from parsl.launchers import MpiExecLauncher
from parsl.providers import PBSProProvider
from parsl.utils import get_all_checkpoints


def get_config(user_opts: dict[str, Any]) -> Config:
    """Get the Parsl config for Polaris@ALCF."""
    run_dir = user_opts.get('run_dir', './parsl')

    checkpoints = get_all_checkpoints(run_dir)
    print('Found the following checkpoints: ', checkpoints)

    config = Config(
        executors=[
            HighThroughputExecutor(
                label='htex',
                heartbeat_period=15,
                heartbeat_threshold=120,
                worker_debug=True,
                # available_accelerators will override settings for max_workers
                available_accelerators=user_opts['available_accelerators'],
                cores_per_worker=user_opts['cores_per_worker'],
                address=address_by_interface('bond0'),
                cpu_affinity='block-reverse',
                prefetch_capacity=0,
                provider=PBSProProvider(
                    launcher=MpiExecLauncher(
                        bind_cmd='--cpu-bind',
                        overrides='--depth=64 --ppn 1',
                    ),
                    account=user_opts['account'],
                    queue=user_opts['queue'],
                    select_options='ngpus=4',
                    # PBS directives: for array jobs pass '-J' option
                    scheduler_options=user_opts['scheduler_options'],
                    # Command to be run before starting a worker, such as:
                    worker_init=user_opts['worker_init'],
                    # number of compute nodes allocated for each block
                    nodes_per_block=user_opts['nodes_per_block'],
                    init_blocks=1,
                    min_blocks=0,
                    max_blocks=1,  # Increase to have more parallel jobs
                    cpus_per_node=user_opts['cpus_per_node'],
                    walltime=user_opts['walltime'],
                ),
            ),
        ],
        checkpoint_files=checkpoints,
        run_dir=run_dir,
        checkpoint_mode='task_exit',
        retries=2,
        app_cache=True,
    )

    return config
