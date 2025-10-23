"""CLI for the PDF workflow package."""

from __future__ import annotations

from pathlib import Path

import typer

app = typer.Typer()


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
