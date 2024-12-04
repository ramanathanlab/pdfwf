"""The PyPDF (not PyMuPDF!) PDF parser."""

from __future__ import annotations

import logging
import re
from typing import Any
from typing import Literal

from pypdf import PdfReader

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

    def extract_doi_info(self, input_str: str) -> str:
        """Extract doi from pypdf metadata entry (if present)."""
        match = re.search(r'(doi:\s*|doi\.org/)(\S+)', input_str)
        return match.group(2) if match else ''

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
        # TODO: This needs to be closed
        # Open
        reader = PdfReader(pdf_path)

        # Scrape text
        full_text = ''
       
        # - page char indices
        cumm_idx = 0
        page_indices = [0]
        
        # loop pages
        for page in reader.pages:
            page_txt = page.extract_text(extraction_mode='layout')
            full_text += page_txt
            # - char indices
            cumm_idx += len(page_txt)
            page_indices.append(cumm_idx)
        
        # remove trailing index 
        page_indices = page_indices[:-1]

        # 1st page
        first_page_text = (
            reader.pages[0].extract_text(extraction_mode='layout')
            if len(reader.pages[0]) > 0
            else ''
        )
        meta = reader.metadata

        # Metadata (available to pypdf)
        title = meta.get('/Title', '')
        authors = meta.get('/Author', '')
        createdate = meta.get('/CreationDate', '')
        keywords = meta.get('/Keywords', '')
        # Use .get() to handle the missing DOI key
        doi = (
            meta.get('/doi', '')
            if meta.get('/doi', '') != ''
            else self.extract_doi_info(meta.get('/Subject', ''))
        )
        prod = meta.get('/Producer', '')
        # Not included for pypdf, so we set it directly
        form = meta.get('/Format', '')
        abstract = (
            meta.get('/Subject', '')
            if len(meta.get('/Subject', '')) > self.abstract_threshold
            else ''
        )

        # Assemble the metadata
        out_meta = {
            'title': title,
            'authors': authors,
            'createdate': createdate,
            'keywords': keywords,
            'doi': doi,
            'producer': prod,
            'format': form,
            'first_page': first_page_text,
            'abstract': abstract,
            'page_char_idx' : page_indices,
        }

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
                'parser': self.config.name
            }
            documents.append(document)

        return documents
