"""Converts a directory of markdown files to JSONL files."""
from __future__ import annotations

import json
from argparse import ArgumentParser
from pathlib import Path
from uuid import uuid4

import pandas as pd
import pyarrow.parquet as pq


def _write_jsonl(output_dir: Path, documents: list[dict[str, str]]) -> None:
    """Write a list of documents to a JSONL file.

    Parameters
    ----------
    output_dir : Path
        The directory to write the output JSON lines file to.
    documents : list[dict[str, str]]
        The list of parsed documents.
    """
    # Merge parsed documents into a single string of JSON lines
    lines = ''.join(f'{json.dumps(doc)}\n' for doc in documents)

    # Write the JSON lines strings to disk
    with open(output_dir / f'{uuid4()}.jsonl', 'w') as f:
        f.write(lines)


def parquet_to_jsonl(
    parquet_file: Path,
    output_dir: Path,
    lines_per_jsonl: int,
    text_field: str = 'text',
) -> None:
    """Convert a parquet file to a JSONL file.

    Parameters
    ----------
    parquet_file : Path
        The path to the parquet file.
    output_dir : Path
        The directory to write the output JSON lines file to.
    pdf_dir : Path
        The directory containing the source PDF files.
    lines_per_jsonl : int
        Number of lines files per JSONL file.
    text_field : str, default='text'
        The name of the field containing the text in the parquet file.
    """
    # Create the output directory
    output_dir.mkdir(exist_ok=False, parents=True)

    # Create a list to store the parsed documents
    documents: list[dict[str, str]] = []

    # Read parquet file
    table = pq.read_table(parquet_file, memory_map=True)

    # Convert to pandas
    df: pd.DataFrame = table.to_pandas()

    # Drop any columns that are not needed
    for colname in ['embedding', 'encoded_labels']:
        if colname in df.columns:
            df = df.drop(columns=[colname])

    # Rename the text field if necessary
    if text_field not in df.columns:
        df = df.rename(columns={text_field: 'text'})

    # Loop through the parquet file
    for idx, row in df.iterrows():
        # Append the parsed document to the list
        documents.append(row.to_dict())

        # Write the parsed documents to disk
        if idx % lines_per_jsonl == 0:
            _write_jsonl(output_dir, documents)
            # Clear the documents list
            documents = []

    # Write the remaining documents to disk
    if documents:
        _write_jsonl(output_dir, documents)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument(
        '--parquet_file',
        type=Path,
        help='The path to the parquet file.',
    )
    parser.add_argument(
        '--output_dir',
        type=Path,
        help='The directory to write the output JSON lines file to.',
    )
    parser.add_argument(
        '--lines_per_jsonl',
        type=int,
        help='Number of lines per JSONL file.',
    )
    parser.add_argument(
        '--text_field',
        type=str,
        default='text',
        help='The name of the field containing the text in the parquet file.',
    )
    args = parser.parse_args()

    parquet_to_jsonl(
        parquet_file=args.parquet_file,
        output_dir=args.output_dir,
        lines_per_jsonl=args.lines_per_jsonl,
        text_field=args.text_field,
    )
