"""The AdaParse PDF parser."""

from __future__ import annotations

from typing import Any
from typing import Literal

from pydantic import Field

from pdfwf.parsers.base import BaseParser
from pdfwf.parsers.base import BaseParserConfig
from pdfwf.parsers.nougat_ import NougatParser
from pdfwf.parsers.nougat_ import NougatParserConfig
from pdfwf.parsers.pymupdf import PyMuPDFParser
from pdfwf.parsers.pymupdf import PyMuPDFParserConfig
from pdfwf.utils import exception_handler

__all__ = [
    'AdaParse',
    'AdaParseConfig',
]


class AdaParseConfig(BaseParserConfig):
    """Settings for the AdaParse parser."""

    # The name of the parser.
    name: Literal['adaparse'] = 'adaparse'  # type: ignore[assignment]

    pymupdf_config: PyMuPDFParserConfig = Field(
        default_factory=PyMuPDFParserConfig,
        description='Settings for the PyMuPDF-PDF parser.',
    )
    nougat_config: NougatParserConfig = Field(
        description='Settings for the Nougat-PDF parser.',
    )


class AdaParse(BaseParser):
    """Interface for the AdaParse PDF parser."""

    def __init__(self, config: AdaParseConfig) -> None:
        """Initialize the parser."""
        self.pymudf_parser = PyMuPDFParser(config=config.pymupdf_config)
        self.nougat_parser = NougatParser(config=config.nougat_config)

        # TODO: Implement the quality check regressor
        # Return a 0 or 1 for each pdf file. If 0, the pdf is of low quality
        # and should be parsed with the Nougat parser.
        self.regressor = lambda x: [0 for _ in range(len(x))]

    @exception_handler(default_return=None)
    def parse(self, pdf_files: list[str]) -> list[dict[str, Any]] | None:
        """Parse a list of pdf files and return the parsed data."""
        # First, parse the PDFs using PyMuPDF
        documents = self.pymudf_parser.parse(pdf_files)

        # If no documents, there was an error parsing the PDFs with PyMuPDF
        if documents is None:
            return None

        # Apply the quality check regressor
        qualities = self.regressor(documents)

        # Remove the documents that failed the quality check
        documents = [d for d, q in zip(documents, qualities) if q]

        # Collect the documents that failed the quality check
        low_quality_pdfs = [p for p, q in zip(pdf_files, qualities) if not q]

        # If no low-quality documents, return the parsed documents
        if not low_quality_pdfs:
            return documents

        # Parse the low-quality documents using the Nougat parser
        nougat_documents = self.nougat_parser.parse(low_quality_pdfs)

        # If Nougat documents were parsed, add them to the output
        if nougat_documents is not None:
            documents.extend(nougat_documents)

        # Finally, return the parsed documents from both parsers
        return documents
