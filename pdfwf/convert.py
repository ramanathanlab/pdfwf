"""PDF conversion workflow with marker."""
from __future__ import annotations

import sys
from argparse import ArgumentParser
from pathlib import Path

import parsl
from parsl import python_app

from pdfwf.parsl import ComputeSettingsTypes
from pdfwf.utils import BaseModel
from pdfwf.utils import batch_data
from pdfwf.utils import setup_logging


@python_app
def marker_single_app(pdf_paths: list[str], out_dirs: list[str]) -> list[str]:
    """Process a single PDF with marker.

    Parameters
    ----------
    pdf_path : list[str]
        Paths to a batch of PDF file to convert.
    out_dirs : list[str]
        Paths to a batch of output directories to place the parsed PDFs.

    Returns:
    -------
    list[str]
        The paths to the output prefix (i.e., a reference to the output files).
    """
    import json
    from pathlib import Path

    from pdfwf.parsers.marker import MarkerParser

    # Initialize the marker parser. This loads the models into memory
    # and registers them in a global registry unique to the current
    # parsl worker process. This ensures that the models are only
    # loaded once per worker process (i.e., we warmstart the models)
    parser = MarkerParser()

    output_prefixes = []
    # Process each PDF
    for pdf_path, out_dir in zip(pdf_paths, out_dirs):
        # Parse the PDF
        output = parser.parse(pdf_path)
        if output is None:
            output_prefixes.append(f'Error: Failed to parse {pdf_path}')
            continue

        # Unpack the output
        full_text, out_meta = output

        # Set the output path prefix as the name of the PDF file
        prefix = Path(out_dir) / Path(pdf_path).stem

        # Write the output markdown file /out_dir/pdf_name.md
        out_path = prefix.with_suffix('.md')
        with open(out_path, 'w+', encoding='utf-8') as f:
            f.write(full_text)

        # Write the output metadata file /out_dir/pdf_name.metadata.json
        out_path = prefix.with_suffix('.metadata.json')
        with open(out_path, 'w+', encoding='utf-8') as f:
            f.write(json.dumps(out_meta, indent=4))

        # Collect the output prefix
        output_prefixes.append(prefix.as_posix())

    return output_prefixes


class WorkflowConfig(BaseModel):
    """Configuration for the PDF conversion workflow."""

    pdf_dir: Path
    """Directory containing pdfs to convert."""

    out_dir: Path
    """Directory to place converted pdfs in."""

    num_conversions: int = sys.maxsize
    """Number of pdfs to convert (useful for debugging)."""

    chunk_size: int = 1
    """Number of pdfs to convert in a single batch."""

    compute_settings: ComputeSettingsTypes
    """Compute settings (HPC platform, number of GPUs, etc)."""


if __name__ == '__main__':
    parser = ArgumentParser(description='PDF conversion workflow')
    parser.add_argument(
        '--config',
        type=Path,
        required=True,
        help='Path to workflow configuration file',
    )
    args = parser.parse_args()

    # Load workflow configuration
    config = WorkflowConfig.from_yaml(args.config)

    # Setup output directory
    config.out_dir = config.out_dir.resolve()
    config.out_dir.mkdir(exist_ok=True, parents=True)

    # Setup logging
    logger = setup_logging('pdfwf', config.out_dir)

    # Setup parsl for distributed computing
    parsl_cfg = config.compute_settings.get_config(config.out_dir / 'parsl')
    parsl.load(parsl_cfg)

    # TODO: Once we decide on output format, we can probably
    # have a single output directory set via a partial function

    # Collect PDFs in batches for more efficient processing
    pdf_paths, out_dirs = [], []
    for pdf_path in config.pdf_dir.glob('**/*.pdf'):
        # Create output directory keeping the same directory structure
        text_outdir = (
            config.out_dir / pdf_path.relative_to(config.pdf_dir)
        ).parent
        text_outdir.mkdir(exist_ok=True, parents=True)
        # Collect the input args
        pdf_paths.append(pdf_path.as_posix())
        out_dirs.append(text_outdir.as_posix())

        if len(pdf_paths) >= config.num_conversions:
            break

    # Batch the input args
    batched_pdf_paths = batch_data(pdf_paths, config.chunk_size)
    batched_out_paths = batch_data(out_dirs, config.chunk_size)
    batched_args = zip(batched_pdf_paths, batched_out_paths)

    # Submit jobs
    futures = [marker_single_app(*args) for args in batched_args]

    logger.info(f'Submitted {len(futures)} jobs')

    # Wait for jobs to complete and log success/failure
    with open(config.out_dir / 'result_log.txt', 'w+') as f:
        for future in futures:
            try:
                res = future.result()
            except Exception as e:
                # Individual conversion errors are handled from the
                # exception_handler decorator, this error will occur from a
                # parsl worker level (likely import errors/package errors)
                res = [f'TID: {future.TID}\tError: {e}']

            f.write('\n'.join(res))

    logger.info(f'Completed {len(futures)} jobs')
