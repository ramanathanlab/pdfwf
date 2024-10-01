"""The (Py)Tesseract PDF parser."""

from __future__ import annotations

import os
from typing import Any
from typing import Literal
from pathlib import Path
import pymupdf
from PIL import Image
import pytesseract
from typing import Any

from pydantic import field_validator

from pdfwf.parsers.base import BaseParser
from pdfwf.parsers.base import BaseParserConfig
from pdfwf.utils import exception_handler

__all__ = [
    'TesseractParser',
    'TesseractParserConfig',
]

class TesseractParserConfig(BaseParserConfig):
    """Settings for the (Py)Tesseract PDF parser."""

    # The name of the parser.
    name: Literal['tesseract'] = 'tesseract'  # type: ignore[assignment]

    # desired resolution of page image in dots per inch (dpi); >=300 sugg.
    dpi: int = 300
    # language used in the PDF document
    lang: str = 'eng'
    # path to tessdata
    tessdata_path: Path = Path('/eagle/projects/argonne_tpc/siebenschuh/aurora_gpt/tesseract/tesseract-5.3.0/tessdata/')

    @field_validator('tessdata_path')
    @classmethod
    def validate_tessdata_path_is_dir(cls, value: Path) -> Path:
        """Check if tessdata path exists and is a directory."""
        if not value.is_dir():
            raise ValueError(f"Tessdata path '{value}' does not exist or is not a directory.")
        return value

    @field_validator('lang')
    @classmethod
    def validate_lang(cls, value: str) -> str:
        """Ensure the language is one of the supported languages (currently only 'eng')."""
        supported_languages = ['eng']  # Expand this list if needed
        if value not in supported_languages:
            raise ValueError(f"Unsupported language '{value}'. Supported languages are: {supported_languages}")
        return value

class TesseractParser(BaseParser):
    """Interface for the (Py)Tesseract PDF parser."""

    def __init__(self, config: TesseractParserConfig) -> None:
        """Initialize the parser."""

        self.config = config

        # load tesseract data path
        os.environ['TESSDATA_PREFIX'] = str(self.config.tessdata_path)

    @exception_handler(default_return=None)
    def parse_pdf(self, pdf_path: str) -> tuple[str, dict[str, str]] | None:
        """Parse a PDF file.

        Parameters
        ----------
        pdf_path : str
            Path to the PDF file to convert.

        Returns
        -------
        tuple[str, dict[str, str]] | None
            A tuple containing the full text of the PDF and the metadata
            extracted from the PDF. If parsing fails, return None.
        """
        # Open pdf
        doc = pymupdf.open(pdf_path)

        # Parse text from image from document page
        text_list = []
        for page in doc:
            try:
                # convert the page to image (pixmap) given config`s dpi
                pix = page.get_pixmap(dpi=self.config.dpi)

                # Convert the pixmap to a PIL image
                img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

                # Use pytesseract to perform OCR on the image
                page_text = pytesseract.image_to_string(img, lang=self.config.lang)

                # Append the text to the list
                text_list.append(page_text)
            except Exception as e:
                print(f"An error occurred on page {page.number}: {e}")

        # cloe document
        doc.close()

        # merge page-wise texts
        full_text = '\n'.join(text_list)

        # return full text
        return full_text

    @exception_handler(default_return=None)
    def parse(self, pdf_files: list[str]) -> list[dict[str, Any]] | None:
        """Parse a list of pdf files and return the parsed data."""
        documents = []
        # Process each PDF
        for pdf_file in pdf_files:
            # Parse the PDF
            output = self.parse_pdf(pdf_file)

            # Check if the PDF was parsed successfully
            if output is None:
                print(f'Error: Failed to parse {pdf_file}')
                continue

            # output is full-text-only
            text = output

            # Setup the document fields to be stored
            document = {
                'text': text,
                'path': str(pdf_file),
            }
            documents.append(document)

        return documents
