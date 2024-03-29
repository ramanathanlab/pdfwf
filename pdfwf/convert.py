"""PDF conversion workflow."""
from __future__ import annotations

import functools
import sys
from argparse import ArgumentParser
from pathlib import Path
from typing import Any

from parsl.concurrent import ParslPoolExecutor

from pdfwf.parsers import ParserConfigTypes
from pdfwf.parsl import ComputeSettingsTypes
from pdfwf.utils import BaseModel
from pdfwf.utils import batch_data
from pdfwf.utils import setup_logging


def parse_pdfs(
    pdf_paths: list[str], output_dir: Path, parser_kwargs: dict[str, Any]
) -> None:
    """Parse a batch of PDF files and write the output to a JSON lines file.

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
    import time

    from pdfwf.parsers import get_parser
    from pdfwf.utils import setup_logging

    # Setup logging
    logger = setup_logging('pdfwf')

    # Start the application timer
    start = time.time()

    # Initialize the parser. This loads the models into memory and registers
    # them in a global registry unique to the current parsl worker process.
    # This ensures that the models are only loaded once per worker process
    # (i.e., we warmstart the models)
    parser = get_parser(parser_kwargs, register=True)

    # Process the PDF files in bulk
    documents = parser.parse(pdf_paths)

    # Print the parsing time
    end = time.time()
    logger.info(f'Parsed {len(pdf_paths)} PDFs in {end - start:.2f} seconds')

    # If parsing failed, return early
    if documents is None:
        logger.info(f'Failed to parse {pdf_paths}')
        return

    # Merge parsed documents into a single string of JSON lines
    lines = ''.join(f'{json.dumps(doc)}\n' for doc in documents)

    # Store the JSON lines strings to a disk using a single write operation
    with open(output_dir / f'{parser.unique_id}.jsonl', 'a+') as f:
        f.write(lines)

    # Log the time taken to process the batch
    end = time.time()
    logger.info(
        f'Processed {len(pdf_paths)} PDFs in {end - start:.2f} seconds'
    )


def parse_zip(
    zip_file: str, output_dir: Path, parser_kwargs: dict[str, Any]
) -> None:
    """Parse the PDF files stored within a zip file.

    Parameters
    ----------
    zip_file : str
        Path to the zip file containing the PDFs to parse.
    output_dir : Path
        Directory to write the output JSON lines file to.
    parser_kwargs : dict[str, Any]
        Keyword arguments to pass to the parser. Contains an extra `name`
        argument to specify the parser to use.
    """
    import shutil
    import subprocess
    import uuid
    from pathlib import Path

    from pdfwf.convert import parse_pdfs

    # Make a temporary directory to unzip the file (use a UUID
    # to avoid name collisions)
    local_dir = Path('/local/scratch') / str(uuid.uuid4())
    temp_dir = local_dir / Path(zip_file).stem
    temp_dir.mkdir(parents=True)

    # Unzip the file (quietly--no verbose output)
    subprocess.run(['unzip', '-q', zip_file, '-d', temp_dir], check=False)

    # Glob the PDFs
    pdf_paths = [str(p) for p in temp_dir.glob('**/*.pdf')]

    # Call the parse_pdfs function
    parse_pdfs(pdf_paths, output_dir, parser_kwargs)

    # Clean up the temporary directory
    shutil.rmtree(local_dir)

    # Log the zip file that was processed
    print(f'Finished processing {zip_file}')


class WorkflowConfig(BaseModel):
    """Configuration for the PDF parsing workflow."""

    pdf_dir: Path
    """Directory containing pdfs to parse."""

    out_dir: Path
    """The output directory of the workflow."""

    iszip: bool = False
    """Whether the input files are zip files containing many PDFs."""

    num_conversions: int = sys.maxsize
    """Number of pdfs to convert (useful for debugging)."""

    chunk_size: int = 1
    """Number of pdfs to convert in a single batch."""

    parser_settings: ParserConfigTypes
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

    # Save the configuration to the output directory
    config.write_yaml(config.out_dir / 'config.yaml')

    # File extension for the input files
    file_ext = 'zip' if config.iszip else 'pdf'

    # Collect files in batches for more efficient processing
    files = [p.as_posix() for p in config.pdf_dir.glob(f'**/*.{file_ext}')]

    # Limit the number of conversions for debugging
    if len(files) >= config.num_conversions:
        files = files[: config.num_conversions]
        logger.info(
            f'len(files) exceeds {config.num_conversions}. '
            f'Only first {config.num_conversions} pdfs passed.'
        )

    # Log the input files
    logger.info(f'Found {len(files)} {file_ext} files to parse')

    # Batch the input args
    # Zip files have many PDFs, so we process them in a single batch,
    # while individual PDFs are batched in chunks to maintain higher throughput
    batched_files = (
        files if config.iszip else batch_data(files, config.chunk_size)
    )

    # Create a subdirectory to write the output to
    pdf_output_dir = config.out_dir / 'parsed_pdfs'
    pdf_output_dir.mkdir(exist_ok=True)

    # Log the output directory and number of batches
    logger.info(f'Writing output to {pdf_output_dir}')
    logger.info(f'Processing {len(batched_files)} batches')  # type: ignore[arg-type]

    # Setup the worker function with default arguments
    if config.iszip:
        worker_fn = functools.partial(
            parse_zip,
            output_dir=pdf_output_dir,
            parser_kwargs=config.parser_settings.model_dump(),
        )
    else:
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
        list(pool.map(worker_fn, batched_files))
