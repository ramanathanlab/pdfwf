"""PDF conversion workflow with marker."""
from __future__ import annotations

import functools
import sys
from argparse import ArgumentParser
from pathlib import Path
from typing import Any

from parsl.concurrent import ParslPoolExecutor

from pdfwf.parsers.marker import MarkerParserSettings
from pdfwf.parsers.oreo import OreoParserSettings
from pdfwf.parsl import ComputeSettingsTypes
from pdfwf.utils import BaseModel
from pdfwf.utils import batch_data
from pdfwf.utils import setup_logging


def parse_pdfs(
    pdf_paths: list[str],
    output_dir: Path,
    parser_kwargs: dict[str, Any],
) -> None:
    """Process a single PDF with marker.

    Parameters
    ----------
    pdf_path : list[str]
        Paths to a batch of PDF file to convert.
    output_dir: Path
        Directory to write the output JSON lines file to.
    parser_kwargs : dict[str, Any]
        Keyword arguments to pass to the parser. Contains an extra `name`
        argument to specify the parser to use.
    """
    import json

    # raise ValueError(f"Received parser kwargs: {parser_kwargs}, name type: {type(parser_kwargs['name'])}")
    print(parser_kwargs, flush=True)
    # Initialize the parser. This loads the models into memory and registers
    # them in a global registry unique to the current parsl worker process.
    # This ensures that the models are only loaded once per worker process
    # (i.e., we warmstart the models)
    # parser_name = parser_kwargs.pop('name', None)
    print(parser_kwargs['name'], flush=True)
    print(parser_kwargs, flush=True)
    if parser_kwargs['name'] == 'marker':
        from pdfwf.parsers.marker import MarkerParser

        parser = MarkerParser(**parser_kwargs)
    elif parser_kwargs['name'] == 'oreo':
        from pdfwf.parsers.oreo import OreoParser

        parser = OreoParser(**parser_kwargs)
    else:
        raise ValueError(f'Unknown parser name: {parser_kwargs["name"]}')
    print(f'Built {parser_kwargs["name"]} parser', flush=True)
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
    """The output directory of the workflow."""

    num_conversions: int = sys.maxsize
    """Number of pdfs to convert (useful for debugging)."""

    chunk_size: int = 1
    """Number of pdfs to convert in a single batch."""

    parser_settings: MarkerParserSettings | OreoParserSettings
    """Parser settings (e.g., model paths, etc)."""

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

    logger.info(f'Loaded configuration: {config}')

    # Collect PDFs in batches for more efficient processing
    pdf_paths = [p.as_posix() for p in config.pdf_dir.glob('**/*.pdf')]

    # Limit the number of conversions for debugging
    if len(pdf_paths) >= config.num_conversions:
        pdf_paths = pdf_paths[: config.num_conversions]

    logger.info(f'Found {len(pdf_paths)} PDFs to parse')

    # Batch the input args
    batched_pdf_paths = batch_data(pdf_paths, config.chunk_size)

    # Create a subdirectory to write the output to
    pdf_output_dir = config.out_dir / 'parsed_pdfs'
    pdf_output_dir.mkdir(exist_ok=True)

    logger.info(f'Writing output to {pdf_output_dir}')

    # Setup the worker function with default arguments
    worker_fn = functools.partial(
        parse_pdfs,
        output_dir=pdf_output_dir,
        parser_kwargs=config.parser_settings.model_dump(),
    )

    # Setup parsl for distributed computing
    parsl_config = config.compute_settings.get_config(config.out_dir / 'parsl')

    # Log the checkpoint files
    logger.info(
        f'Found the following checkpoints: {parsl_config.checkpoint_files}'
    )

    # Distribute the input files across processes
    with ParslPoolExecutor(parsl_config) as pool:
        pool.map(worker_fn, batched_pdf_paths)
