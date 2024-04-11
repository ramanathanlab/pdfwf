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

    settings = SunspotSettings(
        num_nodes=10,
        account="candle_aesp",
        queue="run_next",
        walltime="00",
    )

    parsl_config = settings.get_parsl_config(run_dir=args.run_dir)

    inputs = list(range(args.num_tasks))

    with ParslPoolExecutor(config=parsl_config) as pool:
        list(pool.map(sleep_fn, inputs))

