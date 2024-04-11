"""The Nougat PDF parser."""


import re
import time
from functools import partial
from pathlib import Path
from typing import Any
from typing import Literal, Optional, List, Dict

from pydantic import field_validator

from pdfwf.parsers.base import BaseParser
from pdfwf.parsers.base import BaseParserConfig
from pdfwf.utils import exception_handler
from pdfwf.utils import setup_logging

__all__ = [
    'NougatParser',
    'NougatParserConfig',
]


class NougatParserConfig(BaseParserConfig):
    """Settings for the marker PDF parser."""

    # The name of the parser.
    name: Literal['nougat'] = 'nougat'  # type: ignore[assignment]
    # The batch size for the parser (10 is the max that fits in an A100).
    batchsize: int = 10
    # The number of workers to use for dataloading.
    num_workers: int = 1
    # The Number of batches loaded in advance by each worker. 2 means there
    # will be a total of 2 * num_workers batches prefetched across all workers.
    prefetch_factor: int = 4
    # The path to the Nougat model checkpoint.
    checkpoint: Path
    # The directory to write optional mmd outputs along with jsonls.
    mmd_out: Optional[Path] = None
    # Override pre-existing parsed outputs.
    recompute: bool = False
    # Use float32 instead of bfloat32.
    full_precision: bool = False
    # Whether to format the output as markdown.
    markdown: bool = True
    # Skip if the model falls in repetition.
    skipping: bool = True
    # The directory to write the logs to.
    nougat_logs_path: Path

    @field_validator('mmd_out')
    @classmethod
    def validate_mmd_out_is_dir(cls, value: Optional[Path]) -> Optional[Path]:
        """Create the output directory if it does not exist."""
        if value is not None:
            value.mkdir(exist_ok=True, parents=True)
        return value

    @field_validator('checkpoint')
    @classmethod
    def validate_ckpt_path_exists(cls, value: Path) -> Path:
        """Check if the directory exists."""
        if not value.exists():
            from nougat.utils.checkpoint import get_checkpoint

            print(
                'Checkpoint not found in the directory you specified. '
                'Downloading base model from the internet instead.'
            )
            value = get_checkpoint(value, model_tag='0.1.0-base')
        return value


class NougatParser(BaseParser):
    """Warmstart interface for the marker PDF parser.

    Initialization loads the Nougat models into memory and registers them in a
    global registry unique to the current process. This ensures that the models
    are only loaded once per worker process (i.e., we warmstart the models)
    """

    def __init__(self, config: NougatParserConfig) -> None:
        """Initialize the marker parser."""
        import torch
        from nougat import NougatModel
        from nougat.utils.device import move_to_device

        self.config = config
        self.model = NougatModel.from_pretrained(config.checkpoint)
        self.model.eval()
        self.model = torch.compile(self.model, fullgraph=True)
        self.model = move_to_device(
            self.model,
            bf16=not self.config.full_precision,
            cuda=self.config.batchsize > 0,
        )
        self.logger = setup_logging('pdfwf_nougat', config.nougat_logs_path)

        # Log the output data information
        if self.config.mmd_out is not None:
            self.logger.info(
                f'Writing markdown files to {self.config.mmd_out}'
            )
        else:
            self.logger.info(
                '`mmd_out` not specified, will not write markdown files.'
            )

    @exception_handler(default_return=None)
    def parse(self, pdf_files: list[str]) -> Optional[List[Dict[str, Any]]]:  # noqa: PLR0912, PLR0915
        """Parse a PDF file and extract markdown.

        Parameters
        ----------
        pdf_path : str
            Path to the PDF file to convert.

        Returns
        -------
        list[dict[str, Any]]
            The extracted documents.
        """
        from nougat.postprocessing import markdown_compatible
        from nougat.utils.dataset import LazyDataset
        from torch.utils.data import ConcatDataset
        from torch.utils.data import DataLoader
        import torch

        pdfs = [Path(pdf_file) for pdf_file in pdf_files]

        if self.config.batchsize <= 0:
            self.config.batchsize = 1

        datasets = []
        for pdf in pdfs:
            if not pdf.exists():
                self.logger.warning(f'Could not find {pdf}. Skipping.')
                continue
            if self.config.mmd_out is not None:
                out_path = self.config.mmd_out / pdf.with_suffix('.mmd').name
                if out_path.exists() and not self.config.recompute:
                    self.logger.info(
                        f'Skipping {pdf.name}: already extracted. '
                        ' Use --recompute config to override extraction.'
                    )
                    continue
            try:
                # TODO: Using self.model.encoder.prepare_input causes the data
                # loader processes to use GPU memory, since prepare_input is
                # a function tied to an nn.Module instance. This is a bug in
                # the Nougat library, but we can work around it by creating
                # a standalone prepare_input function. We leave this
                # to future work since the prepare_input functions calls other
                # class methods and uses some class attributes. See here for
                # more details:
                # https://discuss.pytorch.org/t/distributeddataparallel-causes-dataloader-workers-to-utilize-gpu-memory/88731/5
                dataset = LazyDataset(
                    pdf,
                    partial(
                        self.model.encoder.prepare_input, random_padding=False
                    ),
                )

            # PdfStreamError, ValueError, KeyError, pypdf.errors.PdfReadError,
            # and potentially other exceptions can be raised here.
            except Exception:
                self.logger.info(f'Could not load file {pdf!s}.')
                continue
            datasets.append(dataset)

        # If there are no PDFs to process, return None
        if len(datasets) == 0:
            return None

        dataloader = DataLoader(
            ConcatDataset(datasets),
            batch_size=self.config.batchsize,
            pin_memory=True,
            num_workers=self.config.num_workers,
            prefetch_factor=self.config.prefetch_factor,
            shuffle=False,
            collate_fn=LazyDataset.ignore_none_collate,
            multiprocessing_context='fork'  # This dataloader seems to break with parsl
        )
        documents = []
        predictions = []
        file_index = 0
        page_num = 0
        model_outputs = []

        start = time.time()

        # First pass to get the model outputs
        for sample, is_last_page in dataloader:
            model_output = self.model.inference(
                image_tensors=sample, early_stopping=self.config.skipping
            )
            model_outputs.append((model_output, is_last_page))

        self.logger.info(
            f'First pass took {time.time()-start:.2f} seconds. '
            'Processing the model outputs.'
        )
        start = time.time()

        # Second pass to process the model outputs
        for i, (model_output, is_last_page) in enumerate(model_outputs):
            # check if model output is faulty
            for j, output in enumerate(model_output['predictions']):
                if page_num == 0:
                    self.logger.info(
                        'Processing file %s with %i pages'
                        % (
                            datasets[file_index].name,
                            datasets[file_index].size,
                        )
                    )
                page_num += 1
                if output.strip() == '[MISSING_PAGE_POST]':
                    # uncaught repetitions -- most likely empty page
                    predictions.append(
                        f'\n\n[MISSING_PAGE_EMPTY:{page_num}]\n\n'
                    )
                elif (
                    self.config.skipping
                    and model_output['repeats'][j] is not None
                ):
                    if model_output['repeats'][j] > 0:
                        # If we end up here, it means the output is most
                        # likely not complete and was truncated.
                        self.logger.warning(
                            f'Skipping page {page_num} due to repetitions.'
                        )
                        predictions.append(
                            f'\n\n[MISSING_PAGE_FAIL:{page_num}]\n\n'
                        )
                    else:
                        # If we end up here, it means the document page is too
                        # different from the training domain.
                        # This can happen e.g. for cover pages.
                        predictions.append(
                            f'\n\n[MISSING_PAGE_EMPTY:'
                            f'{i * self.config.batchsize+j+1}]\n\n'
                        )
                else:
                    if self.config.markdown:
                        output = markdown_compatible(output)  # noqa: PLW2901
                    predictions.append(output)
                if is_last_page[j]:
                    out = ''.join(predictions).strip()
                    out = re.sub(r'\n{3,}', '\n\n', out).strip()

                    # TODO: Implement an LLM-based optional metadata extraction
                    # call to run on the first page for author and title.
                    document = {'path': str(is_last_page[j]), 'text': out}
                    documents.append(document)

                    if self.config.mmd_out is not None:
                        # writing the outputs to the markdown files a separate
                        # directory.
                        out_path = (
                            self.config.mmd_out
                            / Path(is_last_page[j]).with_suffix('.mmd').name
                        )
                        out_path.parent.mkdir(parents=True, exist_ok=True)
                        out_path.write_text(out, encoding='utf-8')

                    predictions = []
                    page_num = 0
                    file_index += 1

        self.logger.info(
            f'Second pass took {time.time()-start:.2f} seconds. '
            'Finished processing the model outputs.'
        )

        # workflow return
        return documents
