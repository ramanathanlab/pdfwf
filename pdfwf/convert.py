"""PDF conversion workflow with marker."""
from __future__ import annotations

import functools
import sys
from argparse import ArgumentParser
from pathlib import Path

from parsl.concurrent import ParslPoolExecutor

from pdfwf.parsl import ComputeSettingsTypes
from pdfwf.utils import BaseModel
from pdfwf.utils import batch_data
from pdfwf.utils import setup_logging


def parse_pdfs(pdf_paths: list[str], parser_id: str, output_dir: Path) -> None:
    """Process a single PDF with marker.

    Parameters
    ----------
    pdf_path : list[str]
        Paths to a batch of PDF file to convert.
    parser_id: str
        The parser to use.
    output_dir: Path
        Directory to write the output JSON lines file to.
    """
    # TODO: We should pass in generic kwargs to initialize the parser
    import json

    # Initialize the parser. This loads the models into memory and registers
    # them in a global registry unique to the current parsl worker process.
    # This ensures that the models are only loaded once per worker process
    # (i.e., we warmstart the models)
    if parser_id == 'marker':
        from pdfwf.parsers.marker import MarkerParser

        parser = MarkerParser()
    elif parser_id == 'oreo':
        from pdfwf.parsers.oreo.oreo_v2 import OreoParser

        parser = OreoParser()
    else:
        raise ValueError(f'Invalid parser_id: {parser_id}')

    # Process the PDF files in bulk
    documents = parser.parse(pdf_paths)

    # Convert the document into a JSON lines string
    lines = [json.dumps(doc) for doc in documents]

    # Store the JSON lines strings to a disk using a single write operation
    with open(output_dir / f'{parser.id}.jsonl', 'a+') as f:
        f.writelines(lines)


class WorkflowConfig(BaseModel):
    """Configuration for the PDF parsing workflow."""

    pdf_dir: Path
    """Directory containing pdfs to parse."""

    out_dir: Path
    """Directory to place parsed pdfs in."""

    parser_id: str
    """The parser to use."""

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
    parsl_config = config.compute_settings.get_config(config.out_dir / 'parsl')

    # Log the checkpoint files
    logger.info(
        f'Found the following checkpoints: {parsl_config.checkpoint_files}'
    )

    # Collect PDFs in batches for more efficient processing
    pdf_paths = [p.as_posix() for p in config.pdf_dir.glob('**/*.pdf')]

    # Limit the number of conversions for debugging
    if len(pdf_paths) >= config.num_conversions:
        pdf_paths = pdf_paths[: config.num_conversions]

    # Batch the input args
    batched_pdf_paths = batch_data(pdf_paths, config.chunk_size)

    # Setup the worker function with default arguments
    worker_fn = functools.partial(
        parse_pdfs, parser_id=config.parser_id, output_dir=config.out_dir
    )

    # Distribute the input files across processes
    with ParslPoolExecutor(parsl_config) as pool:
        pool.map(worker_fn, batched_pdf_paths)
