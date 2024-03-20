"""Converts a directory of markdown files to JSONL files."""
from __future__ import annotations

import json
from argparse import ArgumentParser
from pathlib import Path
from uuid import uuid4

from tqdm import tqdm


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


def markdown_to_jsonl(
    markdown_dir: Path, output_dir: Path, pdf_dir: Path, md_per_jsonl: int
) -> None:
    """Convert a directory of markdown files to JSONL files.

    Parameters
    ----------
    markdown_dir : Path
        The path to the markdown directory.
    output_dir : Path
        The directory to write the output JSON lines file to.
    pdf_dir : Path
        The directory containing the source PDF files.
    md_per_jsonl : int
        Number of markdown files per JSONL file.
    """
    # Create the output directory
    output_dir.mkdir(exist_ok=False, parents=True)

    # Retrieve the markdown files
    markdown_files: list[Path] = list(markdown_dir.glob('*.md'))

    # Create a list to store the parsed documents
    documents: list[dict[str, str]] = []

    for idx, path in tqdm(enumerate(markdown_files)):
        # Read the markdown file
        text = path.read_text()

        # Retrieve the PDF source path
        pdf_path = pdf_dir / f'{path.stem}.pdf'

        # Check if the PDF file exists
        if not pdf_path.exists():
            print(f'Could not find {pdf_path} for {path}. Skipping.')
            continue

        # Append the parsed document to the list
        documents.append({'path': str(pdf_path), 'text': text})

        # Write the parsed documents to disk
        if idx % md_per_jsonl == 0:
            _write_jsonl(output_dir, documents)
            # Clear the documents list
            documents = []

    # Write the remaining documents to disk
    if documents:
        _write_jsonl(output_dir, documents)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument(
        '--markdown_dir', type=Path, help='The path to the markdown directory.'
    )
    parser.add_argument(
        '--pdf_dir',
        type=Path,
        help='The directory containing the source PDF files.',
    )
    parser.add_argument(
        '--output_dir',
        type=Path,
        help='The directory to write the output JSON lines file to.',
    )
    parser.add_argument(
        '--md_per_jsonl',
        type=int,
        help='Number of markdown files per JSONL file.',
    )
    args = parser.parse_args()

    markdown_to_jsonl(
        markdown_dir=args.markdown_dir,
        output_dir=args.output_dir,
        pdf_dir=args.pdf_dir,
        md_per_jsonl=args.md_per_jsonl,
    )
