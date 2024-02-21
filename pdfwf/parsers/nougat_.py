"""The Nougat PDF parser."""
from __future__ import annotations

from typing import Any
from typing import Literal

from pdfwf.parsers.base import BaseParser
from pdfwf.parsers.base import BaseParserConfig
from pdfwf.registry import register
from pdfwf.utils import exception_handler, setup_logging

import sys
from pathlib import Path
import re
from argparse import ArgumentParser
from functools import partial
from torch.utils.data import ConcatDataset, DataLoader
from tqdm import tqdm
from pydantic import field_validator

from nougat import NougatModel
from nougat.utils.dataset import LazyDataset
from nougat.utils.device import move_to_device
from nougat.utils.checkpoint import get_checkpoint
from nougat.postprocessing import markdown_compatible
from pypdf.errors import PdfStreamError


__all__ = [
    'NougatParser',
    'NougatParserConfig',
]

class NougatParserConfig(BaseParserConfig):
    """Settings for the marker PDF parser."""

    # The name of the parser.
    name: Literal['nougat'] = 'nougat' # type: ignore[assignment]
    batchsize: int = 10 #max that fits in A100.
    checkpoint: Path
    model: str = "0.1.0-base"
    mmd_out: Path | None #if there is a path, will write mmd files.
    recompute: bool = False
    full_precision: bool = False
    markdown: bool = True
    skipping: bool = True
    nougat_logs_path: Path

    #TODO: Look into auto-downloading the checkpoint so user doesn't have to deal with copying it.
    @field_validator('checkpoint')
    @classmethod
    def validate_ckpt_path_exists(cls, value: Path) -> Path:
        """Check if the directory exists."""
        if not value.exists():
            print(f"Checkpoint not found in the directory you specified. Downloading base model from the internet instead.")
            value = get_checkpoint(value, model_tag=cls.model_fields['model'].default)
        return value

@register() # type: ignore[arg-type]
class NougatParser(BaseParser):
    """Warmstart interface for the marker PDF parser.

    Initialization loads the Nougat models into memory and registers them in a
    global registry unique to the current process. This ensures that the models
    are only loaded once per worker process (i.e., we warmstart the models)
    """

    def __init__(self, config: NougatParserConfig) -> None:
        """Initialize the marker parser."""

        self.config: NougatParserConfig = config
        self.model: NougatModel = NougatModel.from_pretrained(config.checkpoint) # type: ignore[assignment]
        self.logger = setup_logging("pdfwf_nougat", config.nougat_logs_path)

    @exception_handler(default_return=None)
    def parse(self, pdf_files: list[str]) -> list[dict[str, Any]] | None:
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
        pdfs = [Path(pdf_file) for pdf_file in pdf_files] #why are they not Path in the interface anyway?

        if self.config.mmd_out:
            self.logger.info(f"Writing markdown files to {self.config.mmd_out}")

            if not self.config.mmd_out.is_dir():
                self.logger.warning("Markdown output path cannot be a file. Please specify a directory.")
                sys.exit(1)
            else:
                if not self.config.mmd_out.exists():
                    self.logger.info(f"Markdown output directory {self.config.mmd_out} does not exist. Creating it now.")
                    self.config.mmd_out.mkdir(parents=True)
        else:
            self.logger.info("No markdown output path specified. Will not write markdown files.")

        model = move_to_device(self.model, bf16=not self.config.full_precision, cuda=self.config.batchsize > 0)

        if self.config.batchsize <= 0:
         self.config.batchsize = 1
        self.model.eval()
        datasets = []
        for pdf in pdfs:
            if not pdf.exists():
                self.logger.warning(f"Could not find {pdf}. Skipping.")
                continue
            if self.config.mmd_out:
                out_path = self.config.mmd_out / pdf.with_suffix(".mmd").name
                if out_path.exists() and not self.config.recompute:
                    self.logger.info(f"Skipping {pdf.name}: already extracted. Use --recompute config to override extraction.")
                    continue
            try:
                dataset = LazyDataset(
                    pdf,
                    partial(model.encoder.prepare_input, random_padding=False)
                )

            except PdfStreamError:
                self.logger.info(f"Could not load file {str(pdf)}.")
                continue
            datasets.append(dataset)
        if len(datasets) == 0:
            return
        dataloader = DataLoader(
            ConcatDataset(datasets),
            batch_size = self.config.batchsize,
            shuffle=False,
            collate_fn=LazyDataset.ignore_none_collate,
        )
        documents = []
        predictions = []
        file_index = 0
        page_num = 0
        for i, (sample, is_last_page) in enumerate(tqdm(dataloader)):
            model_output = model.inference(
                image_tensors=sample, early_stopping = self.config.skipping
            )
            # check if model output is faulty
            for j, output in enumerate(model_output["predictions"]):
                if page_num == 0:
                    self.logger.info(
                        "Processing file %s with %i pages"
                        % (datasets[file_index].name, datasets[file_index].size)
                    )
                page_num += 1
                if output.strip() == "[MISSING_PAGE_POST]":
                    # uncaught repetitions -- most likely empty page
                    predictions.append(f"\n\n[MISSING_PAGE_EMPTY:{page_num}]\n\n")
                elif self.config.skipping and model_output["repeats"][j] is not None:
                    if model_output["repeats"][j] > 0:
                        # If we end up here, it means the output is most likely not complete and was truncated.
                        self.logger.warning(f"Skipping page {page_num} due to repetitions.")
                        predictions.append(f"\n\n[MISSING_PAGE_FAIL:{page_num}]\n\n")
                    else:
                        # If we end up here, it means the document page is too different from the training domain.
                        # This can happen e.g. for cover pages.
                        predictions.append(
                            f"\n\n[MISSING_PAGE_EMPTY:{i * self.config.batchsize+j+1}]\n\n"
                        )
                else:
                    if self.config.markdown:
                        output = markdown_compatible(output)
                    predictions.append(output)
                if is_last_page[j]:
                    out = "".join(predictions).strip()
                    out = re.sub(r"\n{3,}", "\n\n", out).strip()

                    #TODO: Implement an LLM-based optional metadata extraction call to run on the first page for author and title.
                    document = {
                        "text" : out,
                        "path" : str(pdf),
                        "metadata" : None
                    }
                    documents.append(document)

                    if self.config.mmd_out:
                        #writing the outputs to the markdown files a separate directory.
                        out_path = self.config.mmd_out / Path(is_last_page[j]).with_suffix(".mmd").name
                        out_path.parent.mkdir(parents=True, exist_ok=True)
                        out_path.write_text(out, encoding="utf-8")

                    predictions = []
                    page_num = 0
                    file_index += 1

        #workflow return
        return documents


# #temporary test driver.
# if __name__ == "__main__":
#     parser = ArgumentParser()
#     parser.add_argument("--config", "-c", type=str, help="Path to the configuration file.")
#     args = parser.parse_args()
#     nougat_parser_cfg = NougatParserConfig.from_yaml(args.config)
#     nougat_parser = NougatParser(nougat_parser_cfg)
#     import glob
#     docs = nougat_parser.parse(glob.glob("/home/ogokdemir/nougat_wf/sample_pdfs/*.pdf"))

