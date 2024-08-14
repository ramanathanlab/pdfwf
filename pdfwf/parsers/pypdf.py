"""The PyPDF (not PyMuPDF!) PDF parser."""

from __future__ import annotations

from pypdf import PdfReader
import re
import logging

from typing import Any
from typing import Literal

from pdfwf.parsers.base import BaseParser
from pdfwf.parsers.base import BaseParserConfig
from pdfwf.utils import exception_handler


__all__ = [
    'PyPDFParser',
    'PyPDFParserConfig',
]


class PyPDFParserConfig(BaseParserConfig):
    """Settings for the pypdf-PDF parser."""

    # The name of the parser.
    name: Literal['pypdf'] = 'pypdf'  # type: ignore[assignment]


class PyPDFParser(BaseParser):
    """Warmstart interface for the PyPDF PDF parser.

    No warmsart eneded as PyPDF is a Python library using CPUs only
    """

    def __init__(self, config: PyPDFParserConfig) -> None:
        """Initialize the marker parser."""
        self.config = config
        self.abstract_threshold = 580

        # pypdf is verbose
        logging.getLogger().setLevel(logging.ERROR)

    def extract_doi_info(self, input_str:str) -> str:
        """
        Extracts doi from pypdf metadata entry (if present)
        """
        match = re.search(r'(doi:\s*|doi\.org/)(\S+)', input_str)
        if match:
            return match.group(2)
        else:
            return ''

    def convert_single_pdf(self, pdf_path) -> str:
        """Wraps pypdf functionality"""
        # open
        reader = PdfReader(pdf_path)

        # scrape text
        text_text=''
        for page in reader.pages:
                full_text += page.extract_text(extraction_mode="layout")

        first_page = reader.pages[0] if len(reader.pages[0]) > 0 else ''
        meta = reader.metadata

        # metadata (available to pypdf)
        title = meta.get('/Title', '')
        authors = meta.get('/Author', '')
        createdate = meta.get('/CreationDate', '')
        keywords = meta.get('/Keywords', '')
        doi = meta.get('/doi', '') if meta.get('/doi', '')!='' else self.extract_doi_info(meta.get('/Subject', ''))  # Use .get() to handle the missing DOI key
        prod = meta.get('/Producer', '')
        form = meta.get('/Format', '')  # Not included for pypdf, so we set it directly
        abstract = meta.get('/Subject', '') if len(meta.get('/Subject', '')) > self.abstract_threshold else ''

        # - assemble
        out_meta = {'title' : title,
                    'authors' : authors,
                    'createdate' : createdate,
                    'keywords' : keywords,
                    'doi' : doi,
                    'producer' : prod,
                    'format' : form,
                    'first_page' : first_page_text,
                    'abstract' : abstract,
        }

        # full text & metadata entries
        output = full_text, out_meta

        return output

    @exception_handler(default_return=None)
    def parse_pdf(self, pdf_path: str) -> tuple[str, dict[str, str]] | None:
        """Parse a PDF file and extract markdown.

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

        full_text, out_meta = self.convert_single_pdf(pdf_path)

        return full_text, out_meta

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

            # Unpack the output
            text, metadata = output

            # Setup the document fields to be stored
            document = {
                'text': text,
                'path': str(pdf_file),
                'metadata': metadata,
            }
            documents.append(document)

        return documents
