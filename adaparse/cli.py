"""CLI for the PDF workflow package."""

from __future__ import annotations

from pathlib import Path

import typer

app = typer.Typer()


@app.command()
def nougat(  # noqa: PLR0913
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
    batchsize: int = typer.Option(
        10,
        '--batchsize',
        '-bs',
        help='Number of pages per patch. Maximum 10 for A100 40GB.',
    ),
    checkpoint: Path = typer.Option(  # noqa: B008
        'nougat_ckpts/base',
        '--checkpoint',
        '-c',
        help='Path to existing or new Nougat model checkpoint '
        ' (to be downloaded)',
    ),
    mmd_out: Path = typer.Option(  # noqa: B008
        None,
        '--mmd_out',
        '-m',
        help='The directory to write optional mmd outputs along with jsonls.',
    ),
    recompute: bool = typer.Option(
        False,
        '--recompute',
        '-r',
        help='Override pre-existing parsed outputs.',
    ),
    full_precision: bool = typer.Option(
        False,
        '--full_precision',
        '-f',
        help='Use float32 instead of bfloat32.',
    ),
    markdown: bool = typer.Option(
        True,
        '--markdown',
        '-md',
        help='Output pdf content in markdown compatible format.',
    ),
    skipping: bool = typer.Option(
        True,
        '--skipping',
        '-s',
        help='Skip if the model falls in repetition.',
    ),
    nougat_logs_path: Path = typer.Option(  # noqa: B008
        'adaparse_nougat_logs',
        '--nougat_logs_path',
        '-n',
        help='The path to the Nougat-specific logs.',
    ),
    num_conversions: int = typer.Option(
        0,
        '--num_conversions',
        '-nc',
        help='Number of pdfs to convert (useful for debugging, by default '
        'convert every document).',
    ),
) -> None:
    """Parse PDFs using the nougat parser."""
    from adaparse.convert import parse_pdfs

    # Make the output directory if it doesn't exist
    output_dir.mkdir(exist_ok=True)

    # Collect PDFs in batches for more efficient processing
    pdf_paths = [p.as_posix() for p in pdf_dir.glob('**/*.pdf')]

    # Limit the number of conversions for debugging
    if num_conversions and len(pdf_paths) >= num_conversions:
        pdf_paths = pdf_paths[:num_conversions]
        typer.echo(
            f'len(pdf_paths) exceeds {num_conversions}. '
            f'Only first {num_conversions} pdfs passed.'
        )

    # Raise an error if no PDFs are found
    if not pdf_paths:
        raise ValueError(f'No PDFs found in the input directory {pdf_dir}.')

    # Print the number of PDFs to be parsed
    typer.echo(f'Converting {len(pdf_paths)} PDFs with nougat...')
    typer.echo(f'Parsed PDFs written to output directory: {output_dir}')

    if mmd_out:
        typer.echo(f'Optional mmd outputs written to: {mmd_out}')

    # Setup parser kwargs
    parser_kwargs = {
        'name': 'nougat',
        'batchsize': batchsize,
        'checkpoint': checkpoint,
        'mmd_out': mmd_out,
        'recompute': recompute,
        'full_precision': full_precision,
        'markdown': markdown,
        'skipping': skipping,
        'nougat_logs_path': nougat_logs_path,
    }

    # Parse the PDFs
    parse_pdfs(pdf_paths, output_dir, parser_kwargs=parser_kwargs)


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
    num_conversions: int = typer.Option(
        0,
        '--num_conversions',
        '-nc',
        help='Number of pdfs to convert (useful for debugging, by default '
        'convert every document).',
    ),
) -> None:
    """Parse PDFs using the marker parser."""
    from adaparse.convert import parse_pdfs

    # Make the output directory if it doesn't exist
    output_dir.mkdir(exist_ok=True)

    # Collect PDFs in batches for more efficient processing
    pdf_paths = [p.as_posix() for p in pdf_dir.glob('**/*.pdf')]

    # Limit the number of conversions for debugging
    if num_conversions and len(pdf_paths) >= num_conversions:
        pdf_paths = pdf_paths[:num_conversions]
        typer.echo(
            f'len(pdf_paths) exceeds {num_conversions}. '
            f'Only first {num_conversions} pdfs passed.'
        )

    # Raise an error if no PDFs are found
    if not pdf_paths:
        raise ValueError(f'No PDFs found in the input directory {pdf_dir}.')

    # Print the number of PDFs to be parsed
    typer.echo(f'Converting {len(pdf_paths)} PDFs with marker...')
    typer.echo(f'Parsed PDFs written to output directory: {output_dir}')

    # Parse the PDFs
    parse_pdfs(pdf_paths, output_dir, {'name': 'marker'})


# TODO: For now the Oreo paths have hard-coded defaults,
#       in future release we will provide a way to download
#       the models and provide the paths as arguments.
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
        '/lus/eagle/projects/argonne_tpc/siebenschuh/N-O-REO/model_weights/yolov5_detection_weights.pt',
        '--detection_weights_path',
        '-d',
        help='Weights to layout detection model.',
    ),
    text_cls_weights_path: Path = typer.Option(  # noqa: B008
        '/lus/eagle/projects/argonne_tpc/siebenschuh/N-O-REO/text_classifier/meta_text_classifier',
        '--text_cls_weights_path',
        '-t',
        help='Model weights for (meta) text classifier.',
    ),
    spv05_category_file_path: Path = typer.Option(  # noqa: B008
        '/lus/eagle/projects/argonne_tpc/siebenschuh/N-O-REO/meta/spv05_categories.yaml',
        '--spv05_category_file_path',
        '-s',
        help='Path to the SPV05 category file.',
    ),
    yolov5_path: Path = typer.Option(  # noqa: B008
        None,
        '--yolov5_path',
        '-y',
        help='Path to the YOLOv5 repository.',
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
        help='Batch size of pre-processed patches for ViT '
        'pseudo-OCR inference',
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
        '-x',
        help='Number of pixels along which',
    ),
    num_conversions: int = typer.Option(
        0,
        '--num_conversions',
        '-nc',
        help='Number of pdfs to convert (useful for debugging, by default '
        'convert every document).',
    ),
) -> None:
    """Parse PDFs using the oreo parser."""
    from adaparse.convert import parse_pdfs

    # Make the output directory if it doesn't exist
    output_dir.mkdir(exist_ok=True)

    # Collect PDFs in batches for more efficient processing
    pdf_paths = [p.as_posix() for p in pdf_dir.glob('**/*.pdf')]

    # Limit the number of conversions for debugging
    if num_conversions and len(pdf_paths) >= num_conversions:
        pdf_paths = pdf_paths[:num_conversions]
        typer.echo(
            f'len(pdf_paths) exceeds {num_conversions}. '
            f'Only first {num_conversions} pdfs passed.'
        )

    # Raise an error if no PDFs are found
    if not pdf_paths:
        raise ValueError(f'No PDFs found in the input directory {pdf_dir}.')

    # Print the number of PDFs to be parsed
    typer.echo(f'Converting {len(pdf_paths)} PDFs with oreo...')
    typer.echo(f'Parsed PDFs written to output directory: {output_dir}')

    # Setup parser kwargs
    parser_kwargs = {
        'name': 'oreo',
        'detection_weights_path': detection_weights_path,
        'text_cls_weights_path': text_cls_weights_path,
        'spv05_category_file_path': spv05_category_file_path,
        'yolov5_path': yolov5_path,
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


@app.command()
def balance_jsonl(
    input_dir: Path = typer.Option(  # noqa: B008
        ...,
        '--input_dir',
        '-i',
        help='The directory containing the JSONL files to balance.',
    ),
    output_dir: Path = typer.Option(  # noqa: B008
        ...,
        '--output_dir',
        '-o',
        help='The directory to write the balanced JSONL files to.',
    ),
    lines_per_file: int = typer.Option(
        1000,
        '--lines_per_file',
        '-l',
        help='Number of lines per balanced JSONL file.',
    ),
    num_workers: int = typer.Option(
        1,
        '--num_workers',
        '-n',
        help='Number of worker processes to use for balancing JSONL files.',
    ),
) -> None:
    """Rewrite JSONL files to balance the number of lines per file."""
    from adaparse.balance import balance_jsonl_files

    # Collect JSONL files
    jsonl_files = list(input_dir.glob('*.jsonl'))

    # If no JSONL files are found, raise an error
    if not jsonl_files:
        raise ValueError(
            f'No JSONL files found in the input directory {input_dir}.'
        )

    # Print the output directory
    typer.echo(f'Balanced JSONL files written to: {output_dir}')

    # Print the number of JSONL files to be balanced
    typer.echo(
        f'Balancing {len(jsonl_files)} JSONL files using'
        f' {lines_per_file} lines per file...'
    )

    # Balance the JSONL files
    balance_jsonl_files(
        jsonl_files=jsonl_files,
        output_dir=output_dir,
        lines_per_file=lines_per_file,
        num_workers=num_workers,
    )


@app.command()
def parse_timers(
    run_path: Path = typer.Option(  # noqa: B008
        ...,
        '--run_path',
        '-l',
        help='Path to the workflow run directory.',
    ),
    csv_path: Path = typer.Option(  # noqa: B008
        'timer_logs.csv',
        '--csv_path',
        '-c',
        help='Path to the CSV file to write the parsed timer logs to.',
    ),
) -> None:
    """Parse timer logs from the PDF workflow."""
    import pandas as pd

    from adaparse.timer import TimeLogger

    # Path to the timer logs
    log_dir = run_path / 'parsl' / '000' / 'submit_scripts'

    # Parse the timer logs
    # Note: there could be multiple logs for a single run
    # if the workflow submits multiple jobs back to back
    time_stats = []
    for log_path in log_dir.glob('*.stdout'):
        time_stats.extend(TimeLogger().parse_logs(log_path))

    # Write the parsed timer logs to a CSV file
    pd.DataFrame(time_stats).to_csv(csv_path, index=False)


@app.command()
def zip_pdfs(
    root_dir: Path = typer.Option(  # noqa: B008
        ...,
        '--input_dir',
        '-i',
        help='Path to the root directory containing pdfs.',
    ),
    output_dir: Path = typer.Option(  # noqa: B008
        ...,
        '--output_dir',
        '-o',
        help='Path to the output directory.',
    ),
    chunk_size: int = typer.Option(
        10,
        '--chunk_size',
        '-c',
        help='Number of PDF files per chunk.',
    ),
    glob_pattern: str = typer.Option(
        '**/*.pdf',
        '--glob_pattern',
        '-g',
        help='Glob pattern to search the root directory for.',
    ),
    num_cpus: int = typer.Option(
        1, '--num_cpus', '-n', help="Number of cpu's to use for zipping."
    ),
) -> None:
    """Zip PDF files in chunks."""
    import json
    from concurrent.futures import ProcessPoolExecutor

    from adaparse.utils import batch_data
    from adaparse.utils import zip_worker

    # Make output directory if it does not already exist
    output_dir.mkdir(exist_ok=True, parents=True)

    # Get all PDF files in the directory
    pdf_files = list(root_dir.glob(glob_pattern))
    total_files = len(pdf_files)
    print(f'Found {total_files} PDF files.')

    # Get batched data
    batched_data = batch_data(pdf_files, chunk_size=chunk_size)

    # Get output files
    output_files = [
        output_dir / f'chunk_{i}.zip' for i in range(len(batched_data))
    ]

    # Setup manifest and save
    manifest = {
        str(output_path.resolve()): [str(f.resolve()) for f in batch]
        for output_path, batch in zip(output_files, batched_data)
    }
    # Save a log that saves which zip file contains which pdf's
    with open(output_dir / 'manifest.json', 'w') as f:
        json.dump(manifest, f)

    with ProcessPoolExecutor(max_workers=num_cpus) as pool:
        pool.map(zip_worker, batched_data, output_files)

    print(f'Zipped files to {output_dir}')


def main() -> None:
    """Entry point for CLI."""
    app()


if __name__ == '__main__':
    main()
