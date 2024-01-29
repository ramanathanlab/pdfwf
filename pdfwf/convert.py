"""PDF conversion workflow with marker."""
from __future__ import annotations

import sys
from argparse import ArgumentParser
from pathlib import Path

import parsl
from parsl import python_app

from pdfwf.parsl import ComputeSettingsTypes
from pdfwf.utils import BaseModel
from pdfwf.utils import setup_logging


@python_app
def marker_single_app(pdf_path: str, out_dir: str) -> str:
    """Process a single PDF with marker.

    Parameters
    ----------
    pdf_path : str
        Path to the PDF file to convert.
    out_dir : str
        Path to the output directory to place the parsed PDF contents.

    Returns:
    -------
    str
        The path to the output prefix (i.e., a reference to the output files).
    """
    import json
    from pathlib import Path

    from parsers.marker import MarkerParser

    # Initialize the marker parser. This loads the models into memory
    # and registers them in a global registry unique to the current
    # parsl worker process. This ensures that the models are only
    # loaded once per worker process (i.e., we warmstart the models)
    parser = MarkerParser()

    # Parse the PDF
    full_text, out_meta = parser.parse(pdf_path)

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

    return prefix.as_posix()


class WorkflowConfig(BaseModel):
    """Configuration for the PDF conversion workflow."""

    pdf_dir: Path
    """Directory containing pdfs to convert."""

    out_dir: Path
    """Directory to place converted pdfs in."""

    num_conversions: int = sys.maxsize
    """Number of pdfs to convert (useful for debugging)."""

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
    config = WorkflowConfig.parse_file(args.config)

    # Setup logging
    logger = setup_logging('pdfwf', config.out_dir)

    # Setup parsl for distributed computing
    parsl_cfg = config.compute_settings.get_config(args.out_dir / 'parsl')
    parsl.load(parsl_cfg)

    # Setup convsersions
    out_path = config.out_dir.resolve()
    out_path.mkdir(exist_ok=True, parents=True)

    # Submit jobs
    futures = []
    for pdf_path in config.pdf_dir.glob('**/*.pdf'):
        # Create output directory keeping the same directory structure
        text_outdir = (out_path / pdf_path.relative_to(config.pdf_dir)).parent
        text_outdir.mkdir(exist_ok=True, parents=True)
        # Submit job to convert the PDF to markdown
        future = marker_single_app(str(pdf_path), str(text_outdir))
        # Keep track of the future
        futures.append(future)

        if len(futures) >= config.num_conversions:
            logger.info(
                f'Reached max number of conversions ({config.num_conversions})'
            )
            break

    logger.info(f'Submitted {len(futures)} jobs')

    with open(config.out_dir / 'result_log.txt', 'w+') as f:
        for future in futures:
            try:
                res = future.result()
            except Exception as e:
                res = f'TID: {future.TID}\tError: {e}'

            f.write(f'{res}\n')

    logger.info(f'Completed {len(futures)} jobs')
