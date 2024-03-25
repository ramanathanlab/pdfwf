"""The marker PDF parser."""
from __future__ import annotations

from typing import Any
from typing import Literal

from pdfwf.parsers.base import BaseParser
from pdfwf.parsers.base import BaseParserConfig
from pdfwf.utils import exception_handler

__all__ = [
    'MarkerParser',
    'MarkerParserConfig',
]


class MarkerParserConfig(BaseParserConfig):
    """Settings for the marker PDF parser."""

    # The name of the parser.
    name: Literal['marker'] = 'marker'  # type: ignore[assignment]


class MarkerParser(BaseParser):
    """Warmstart interface for the marker PDF parser.

    Initialization loads the marker models into memory and registers them in a
    global registry unique to the current process. This ensures that the models
    are only loaded once per worker process (i.e., we warmstart the models)
    """

    def __init__(self, config: MarkerParserConfig) -> None:
        """Initialize the marker parser."""
        from marker.models import load_all_models

        self.config = config
        self.model_lst = load_all_models()

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
        from marker.convert import convert_single_pdf

        full_text, out_meta = convert_single_pdf(pdf_path, self.model_lst)

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
            # TODO: We should figure out a more explicit way to store the
            # the metadata in the document. This is a temporary solution.
            document = {
                'text': text,
                'path': str(pdf_file),
                'metadata': metadata,
            }
            documents.append(document)

        return documents
