"""CLI for the PDF workflow package."""
from __future__ import annotations

from pathlib import Path

import typer

app = typer.Typer()


@app.command()
def marker(
    pdf_dir: Path = typer.Option(  # noqa: B008
        ...,
        '--pdf_path',
        '-p',
        help='The directory containing the PDF files to convert'
        ' (recursive glob).',
    ),
    output_dir: Path = typer.Option(  # noqa: B008
        ...,
        '--output_dir',
        '-o',
        help='The directory to write the output JSON lines file to.',
    ),
) -> None:
    """Parse PDFs using the marker parser."""
    from pdfwf.convert import parse_pdfs

    # Make the output directory if it doesn't exist
    output_dir.mkdir(exist_ok=True)

    # Collect PDFs in batches for more efficient processing
    pdf_paths = [p.as_posix() for p in pdf_dir.glob('**/*.pdf')]

    # Print the number of PDFs to be parsed
    typer.echo(f'Converting {len(pdf_paths)} PDFs with marker...')
    typer.echo(f'Parsed PDFs written to output directory: {output_dir}')

    # Parse the PDFs
    parse_pdfs(pdf_paths, output_dir, {'name': 'marker'})


@app.command()
def oreo(  # noqa: PLR0913
    pdf_dir: Path = typer.Option(  # noqa: B008
        ...,
        '--pdf_path',
        '-p',
        help='The directory containing the PDF files to convert'
        ' (recursive glob).',
    ),
    output_dir: Path = typer.Option(  # noqa: B008
        ...,
        '--output_dir',
        '-o',
        help='The directory to write the output JSON lines file to.',
    ),
    detection_weights_path: Path = typer.Option(  # noqa: B008
        ...,
        '--detection_weights_path',
        '-d',
        help='Weights to layout detection model.',
    ),
    text_cls_weights_path: Path = typer.Option(  # noqa: B008
        ...,
        '--text_cls_weights_path',
        '-t',
        help='Model weights for (meta) text classifier.',
    ),
    spv05_category_file_path: Path = typer.Option(  # noqa: B008
        ...,
        '--spv05_category_file_path',
        '-s',
        help='Path to the SPV05 category file.',
    ),
    detect_only: bool = typer.Option(
        False,
        '--detect_only',
        '-d',
        help='File type to be parsed (ignores other files in the input_dir)',
    ),
    meta_only: bool = typer.Option(
        False,
        '--meta_only',
        '-m',
        help='Only parse PDFs for meta data',
    ),
    equation: bool = typer.Option(
        False,
        '--equation',
        '-e',
        help='Include equations into the text categories',
    ),
    table: bool = typer.Option(
        False,
        '--table',
        '-t',
        help='Include table visualizations (will be stored)',
    ),
    figure: bool = typer.Option(
        False,
        '--figure',
        '-f',
        help='Include figure  (will be stored)',
    ),
    secondary_meta: bool = typer.Option(
        False,
        '--secondary_meta',
        '-s',
        help='Include secondary meta data (footnote, headers)',
    ),
    accelerate: bool = typer.Option(
        False,
        '--accelerate',
        '-a',
        help='If true, accelerate inference by packing non-meta text patches',
    ),
    batch_yolo: int = typer.Option(
        128,
        '--batch_yolo',
        '-b',
        help='Main batch size for detection/# of images loaded per batch',
    ),
    batch_vit: int = typer.Option(
        512,
        '--batch_vit',
        '-v',
        help='Batch size of pre-processed patches for ViT pseudo-OCR inference',
    ),
    batch_cls: int = typer.Option(
        512,
        '--batch_cls',
        '-c',
        help='Batch size K for subsequent text processing',
    ),
    bbox_offset: int = typer.Option(
        2,
        '--bbox_offset',
        '-o',
        help='Number of pixels along which',
    ),
) -> None:
    """Parse PDFs using the oreo parser."""
    from pdfwf.convert import parse_pdfs

    # Make the output directory if it doesn't exist
    output_dir.mkdir(exist_ok=True)

    # Collect PDFs in batches for more efficient processing
    pdf_paths = [p.as_posix() for p in pdf_dir.glob('**/*.pdf')]

    # Print the number of PDFs to be parsed
    typer.echo(f'Converting {len(pdf_paths)} PDFs with oreo...')
    typer.echo(f'Parsed PDFs written to output directory: {output_dir}')

    # Setup parser kwargs
    parser_kwargs = {
        'name': 'oreo',
        'detection_weights_path': detection_weights_path,
        'text_cls_weights_path': text_cls_weights_path,
        'spv05_category_file_path': spv05_category_file_path,
        'detect_only': detect_only,
        'meta_only': meta_only,
        'equation': equation,
        'table': table,
        'figure': figure,
        'secondary_meta': secondary_meta,
        'accelerate': accelerate,
        'batch_yolo': batch_yolo,
        'batch_vit': batch_vit,
        'batch_cls': batch_cls,
        'bbox_offset': bbox_offset,
    }

    # Parse the PDFs
    parse_pdfs(pdf_paths, output_dir, parser_kwargs)


def main() -> None:
    """Entry point for CLI."""
    app()


if __name__ == '__main__':
    main()
