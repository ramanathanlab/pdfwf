"""PDF conversion workflow with marker."""
from __future__ import annotations

import logging
from argparse import ArgumentParser
from pathlib import Path

import parsl
from parsl import python_app

from pdfwf.config import get_config


def setup_logging(logger_name: str) -> logging.Logger:
    """Setup logging for the PDF conversion workflow."""
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    formatter = logging.Formatter(
        '[%(asctime)s] [%(levelname)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
    )
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    return logger


@python_app
def marker_single_app(pdf_path: str, out_dir: str) -> str:
    """Process a single PDF with marker."""
    import json
    import os
    from pathlib import Path

    from marker.convert import convert_single_pdf
    from marker.models import load_all_models

    pdf_name = Path(pdf_path).stem

    model_lst = load_all_models()
    full_text, out_meta = convert_single_pdf(pdf_path, model_lst)

    output_md = os.path.join(out_dir, pdf_name + '.md')
    with open(output_md, 'w+', encoding='utf-8') as f:
        f.write(full_text)

    out_meta_filename = os.path.join(out_dir, pdf_name + '.metadata.json')
    with open(out_meta_filename, 'w+', encoding='utf-8') as f:
        f.write(json.dumps(out_meta, indent=4))

    return output_md


if __name__ == '__main__':
    parser = ArgumentParser()
    # PDF conversion options
    parser.add_argument(
        '--pdf-dir',
        type=Path,
        help='Directory containing pdfs to convert',
    )
    parser.add_argument(
        '--out-dir',
        type=Path,
        help='Directory to place converted pdfs in',
    )

    # Parsl options
    parser.add_argument(
        '--run-dir',
        default='./parsl',
        type=Path,
        help='Directory to place parsl run files in',
    )
    parser.add_argument(
        '--hf-cache',
        default=None,
        type=Path,
        help='Directory to place marker huggingface cache in',
    )
    parser.add_argument(
        '--num-nodes',
        type=int,
        default=1,
        help='Number of nodes to use for conversion',
    )
    parser.add_argument(
        '--account',
        required=True,
        type=str,
        help='Account to charge for job',
    )
    parser.add_argument(
        '--queue',
        default='debug',
        type=str,
        help='Queue to use on polaris',
    )
    parser.add_argument(
        '--walltime',
        default='1:00:00',
        type=str,
        help='Max walltime for job in form HH:MM:SS',
    )
    parser.add_argument(
        '--worker-init',
        default='',
        type=str,
        help='Worker init string to pass to parsl',
    )

    # Debugging options
    parser.add_argument(
        '--num_conversions',
        type=float,
        default=float('inf'),
        help='Number of pdfs to convert (useful for debugging)',
    )

    args = parser.parse_args()

    # setup pdfwf logging
    logger = setup_logging('pdfwf')

    # Setup parsl
    run_dir = str(args.run_dir.resolve())
    args.worker_init += f';cd {run_dir}'
    if args.hf_cache is not None:
        args.worker_init += f';export HF_HOME={args.hf_cache.resolve()}'

    user_opts = {
        'run_dir': run_dir,
        # Load the environment where parsl is installed
        'worker_init': args.worker_init,
        # Specify any PBS options here, like filesystems
        'scheduler_options': '#PBS -l filesystems=home:eagle:grand',
        'account': args.account,
        'queue': args.queue,
        'walltime': args.walltime,
        # Number of nodes to allocate
        'nodes_per_block': args.num_nodes,
        # Up to 64 with multithreading
        'cpus_per_node': 32,
        # Each Polaris node has 4 GPUs, setting this ensures one worker per GPU
        'available_accelerators': 4,
        # This will set the number of cpu hardware threads per worker.
        'cores_per_worker': 8,
    }

    config = get_config(user_opts)
    parsl.load(config)

    # Setup convsersions
    out_path = args.out_dir.resolve()
    out_path.mkdir(exist_ok=True, parents=True)

    # Submit jobs
    futures = []
    for pdf_path in args.pdf_dir.glob('**/*.pdf'):
        text_outdir = (out_path / pdf_path.relative_to(args.pdf_dir)).parent
        text_outdir.mkdir(exist_ok=True, parents=True)
        futures.append(marker_single_app(str(pdf_path), str(text_outdir)))

        if len(futures) >= args.num_conversions:
            logger.info(
                f'Reached max number of conversions '
                f' ({int(args.num_conversions)})',
            )
            break

    logger.info(f'Submitted {len(futures)} jobs')

    with open(args.out_dir / 'result_log.txt', 'w+') as f:
        for future in futures:
            try:
                res = future.result()
            except Exception as e:
                res = f'TID: {future.TID}\tError: {e}'

            f.write(f'{res}\n')

    logger.info(f'Completed {len(futures)} jobs')
