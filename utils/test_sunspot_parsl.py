from argparse import ArgumentParser
from pathlib import Path

from parsl.concurrent import ParslPoolExecutor

from pdfwf.parsl import SunspotSettings


def sleep_fn(idx: int):
    import time
    time.sleep(5)
    return idx


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--run_dir", type=Path, required=True)
    parser.add_argument("--num_tasks", type=int, default=1000)
    args = parser.parse_args()

    args.run_dir.mkdir(exist_ok=True, parents=True)

    num_nodes = 32
    settings = SunspotSettings(
        num_nodes=num_nodes,
        account="candle_aesp_CNDA",
        queue="run_next",
        walltime="02:00:00",
    )

    parsl_config = settings.get_config(run_dir=args.run_dir)

    num_workers = num_nodes * 12

    with ParslPoolExecutor(config=parsl_config) as pool:
        for i in range(0, args.num_tasks, num_workers):
            inputs = list(range(i, i + num_workers))
            list(pool.map(sleep_fn, inputs))
