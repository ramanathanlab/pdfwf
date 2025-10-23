from __future__ import annotations

import time
from argparse import ArgumentParser
from functools import partial
from pathlib import Path

import yaml
from nougat.utils.dataset import LazyDataset
from parsers.nougat_ import NougatParser
from parsers.nougat_ import NougatParserConfig
from torch.profiler import profile
from torch.profiler import ProfilerActivity
from torch.utils.data import ConcatDataset
from torch.utils.data import DataLoader

from adaparse.utils import setup_logging


class WorkflowConfig:
    """Workflow-level configuration for PDF parsing."""

    def __init__(self, config_data):
        self.pdf_dir = Path(config_data.get('pdf_dir'))
        self.mmd_out = Path(config_data.get('parser_settings', {}).get('mmd_out'))
        self.parser_config = NougatParserConfig(**config_data['parser_settings'])


def load_config(config_path: Path) -> WorkflowConfig:
    """Load the YAML configuration file."""
    with open(config_path) as f:
        config_data = yaml.safe_load(f)
    return WorkflowConfig(config_data)


def create_dataloader(pdf_files, parser):
    """Create a DataLoader for batched PDF page processing."""
    datasets = []
    for pdf_file in pdf_files:
        if not pdf_file.exists():
            print(f'Skipping missing file: {pdf_file}')
            continue

        if parser.config.mmd_out:
            out_path = parser.config.mmd_out / pdf_file.with_suffix('.mmd').name
            if out_path.exists() and not parser.config.recompute:
                print(f'Skipping already processed file: {pdf_file}')
                continue

        try:
            dataset = LazyDataset(
                pdf_file,
                partial(
                    parser.prepare_input,
                    input_size=parser.model.encoder.input_size,
                    align_long_axis=parser.model.encoder.align_long_axis,
                    random_padding=False,
                ),
            )
            datasets.append(dataset)
        except Exception as e:
            print(f'Error loading file {pdf_file}: {e}')
            continue

    if not datasets:
        print('No valid datasets created.')
        return None

    return DataLoader(
        ConcatDataset(datasets),
        batch_size=parser.config.batchsize,
        pin_memory=True,
        num_workers=parser.config.num_workers,
        prefetch_factor=parser.config.prefetch_factor,
        shuffle=False,
        collate_fn=LazyDataset.ignore_none_collate,
    )


def parse_pdfs_in_batches(pdf_dir: Path, output_dir: Path, parser: NougatParser):
    """Parse PDFs in batches using the specified configuration."""
    pdf_files = list(pdf_dir.glob('**/*.pdf'))

    if not pdf_files:
        print('No PDF files found in the specified directory.')
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    dataloader = create_dataloader(pdf_files, parser)

    if not dataloader:
        return

    documents = []
    model_outputs = []
    start_time = time.time()

    for batch_idx, (sample, is_last_page) in enumerate(dataloader):
        print(f'Processing batch {batch_idx + 1}...')
        try:
            with profile(
                activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                record_shapes=True,
                with_stack=True,
            ) as prof:
                model_output = parser.model.inference(
                    image_tensors=sample, early_stopping=parser.config.skipping
                )

            # Save the profiling results
            prof.export_chrome_trace(str(output_dir / f'profiler_batch_{batch_idx + 1}.json'))
            print(prof.key_averages().table(sort_by='cuda_memory_usage', row_limit=10))

            model_outputs.append((model_output, is_last_page))
        except Exception as e:
            print(f'Error during inference for batch {batch_idx + 1}: {e}')
            continue

    print(f'Model inference completed in {time.time() - start_time:.2f} seconds.')

    for model_output, is_last_page in model_outputs:
        for output, last_page_flag in zip(model_output['predictions'], is_last_page):
            document = {
                'text': output,
                'last_page': last_page_flag,
                'parser': 'nougat',
            }
            documents.append(document)

    output_file = output_dir / 'parsed_results.jsonl'
    with open(output_file, 'w', encoding='utf-8') as f:
        for doc in documents:
            f.write(f'{doc}\\n')

    print(f'Results saved to {output_file}')


def main():
    parser = ArgumentParser(description='Run Nougat parser on a single GPU')
    parser.add_argument(
        '--config',
        type=Path,
        required=True,
        help='Path to the YAML configuration file',
    )
    args = parser.parse_args()

    # Load the configuration
    config = load_config(args.config)

    # Set up logging
    logger = setup_logging('nougat', config.parser_config.nougat_logs_path)

    # Log configuration details
    logger.info(f'Configuration loaded: {config.parser_config}')

    # Initialize the Nougat parser
    parser_instance = NougatParser(config=config.parser_config)

    # Parse PDFs
    parse_pdfs_in_batches(config.pdf_dir, config.mmd_out, parser_instance)


if __name__ == '__main__':
    main()
