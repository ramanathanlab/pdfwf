from parsl.config import Config

# PBSPro is the right provider for Polaris:
from parsl.providers import PBSProProvider
# The high throughput executor is for scaling to HPC systems:
from parsl.executors import HighThroughputExecutor
# You can use the MPI launcher, but may want the Gnu Parallel launcher, see below
from parsl.launchers import MpiExecLauncher, GnuParallelLauncher
# address_by_interface is needed for the HighThroughputExecutor:
from parsl.addresses import address_by_interface
# For checkpointing:
from parsl.utils import get_all_checkpoints


def get_config(user_opts: dict[str, any]) -> Config:
    run_dir = user_opts.get("run_dir", "./parsl")

    checkpoints = get_all_checkpoints(run_dir)
    print("Found the following checkpoints: ", checkpoints)

    config = Config(
            executors=[
                HighThroughputExecutor(
                    label="htex",
                    heartbeat_period=15,
                    heartbeat_threshold=120,
                    worker_debug=True,
                    available_accelerators=user_opts["available_accelerators"], # if this is set, it will override other settings for max_workers if set
                    cores_per_worker=user_opts["cores_per_worker"],
                    address=address_by_interface("bond0"),
                    cpu_affinity="block-reverse",
                    prefetch_capacity=0,
                    provider=PBSProProvider(
                        launcher=MpiExecLauncher(bind_cmd="--cpu-bind", overrides="--depth=64 --ppn 1"),
                        # Which launcher to use?  Check out the note below for some details.  Try MPI first!
                        # launcher=GnuParallelLauncher(),
                        account=user_opts["account"],
                        queue=user_opts["queue"],
                        select_options="ngpus=4",
                        # PBS directives (header lines): for array jobs pass '-J' option
                        scheduler_options=user_opts["scheduler_options"],
                        # Command to be run before starting a worker, such as:
                        worker_init=user_opts["worker_init"],
                        # number of compute nodes allocated for each block
                        nodes_per_block=user_opts["nodes_per_block"],
                        init_blocks=1,
                        min_blocks=0,
                        max_blocks=1, # Can increase more to have more parallel jobs
                        cpus_per_node=user_opts["cpus_per_node"],
                        walltime=user_opts["walltime"]
                    ),
                ),
            ],
            checkpoint_files = checkpoints,
            run_dir=run_dir,
            checkpoint_mode = 'task_exit',
            retries=2,
            app_cache=True,
    )

    return config 