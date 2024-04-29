"""Balance output jsonl files from a workflow run."""
from __future__ import annotations

import functools
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from uuid import uuid4

from tqdm import tqdm

from pdfwf.utils import batch_data


def _write_jsonl(output_dir: Path, lines: str) -> None:
    """Write a list of documents to a JSONL file.

    Parameters
    ----------
    output_dir : Path
        The directory to write the output JSON lines file to.
    lines : str
        The JSON lines string.
    """
    # Write the JSON lines strings to disk
    with open(output_dir / f'{uuid4()}.jsonl', 'w') as f:
        f.write(lines)


def _balance_jsonl_files(
    jsonl_files: list[Path], output_dir: Path, lines_per_file: int
) -> None:
    # Create a list to store the parsed documents
    documents: list[str] = []

    for path in tqdm(jsonl_files, desc='JSONL files'):
        # Read the JSONL file
        with open(path) as f:
            lines = f.readlines()

        # Append the lines to the documents list
        documents.extend(lines)

        # Skip if we don't have enough documents
        if len(documents) < lines_per_file:
            continue

        # Write the balanced JSONL files
        for i in range(0, len(documents), lines_per_file):
            # Skip if we don't have enough documents
            if i + lines_per_file > len(documents):
                break

            # Write the JSONL file with the specified number of lines
            _write_jsonl(
                output_dir, ''.join(documents[i : i + lines_per_file])
            )

        # Reset the documents list
        if len(documents) % lines_per_file == 0:
            documents = []
        else:
            # Keep the remaining documents that were not written
            documents = documents[-(len(documents) % lines_per_file) :]

    # Write the remaining documents
    if documents:
        _write_jsonl(output_dir, ''.join(documents))


def balance_jsonl_files(
    jsonl_files: list[Path],
    output_dir: Path,
    lines_per_file: int = 1000,
    num_workers: int = 1,
) -> None:
    """Balance output jsonl files from a workflow run.

    Parameters
    ----------
    jsonl_files : list[Path]
        List of JSONL files to balance.
    output_dir : Path
        The directory to write the balanced JSONL files to.
    lines_per_file : int, optional
        Number of lines per balanced JSONL file, by default 1000
    num_workers : int, optional
        Number of worker processes to use for balancing, by default 1
    """
    # Create the output directory
    output_dir.mkdir(exist_ok=False, parents=True)

    if num_workers == 1:
        _balance_jsonl_files(jsonl_files, output_dir, lines_per_file)
        return

    # Determine chunk size
    chunk_size = len(jsonl_files) // num_workers

    # Split the JSONL files into chunks
    batches = batch_data(jsonl_files, chunk_size)

    worker_fn = functools.partial(
        _balance_jsonl_files,
        output_dir=output_dir,
        lines_per_file=lines_per_file,
    )

    # Make sure we don't have more workers than batches
    num_workers = min(num_workers, len(batches))

    # Balance the JSONL files in parallel
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        executor.map(worker_fn, batches)
