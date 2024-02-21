"""The parsers module storing different PDF parsers."""
from __future__ import annotations

from typing import Any

from pdfwf.parsers.base import BaseParser
from pdfwf.parsers.marker import MarkerParserConfig
from pdfwf.parsers.nougat_ import NougatParserConfig
from pdfwf.parsers.oreo import OreoParserConfig

ParserTypes = MarkerParserConfig | OreoParserConfig | NougatParserConfig


def get_parser(parser_kwargs: dict[str, Any]) -> BaseParser:
    """Get the parser instance based on the parser name and kwargs.

    Parameters
    ----------
    parser_kwargs : dict[str, Any]
        The parser configuration. Contains an extra `name` argument to specify
        the parser to use.

    Returns
    -------
    BaseParser
        The parser instance.

    Raises
    ------
    ValueError
        If the parser name is unknown.
    """
    parser_name = parser_kwargs.get('name', '')
    if parser_name == 'marker':
        from pdfwf.parsers.marker import MarkerParser
        from pdfwf.parsers.marker import MarkerParserConfig

        return MarkerParser(MarkerParserConfig(**parser_kwargs))
    elif parser_name == 'oreo':
        from pdfwf.parsers.oreo import OreoParser
        from pdfwf.parsers.oreo import OreoParserConfig

        return OreoParser(OreoParserConfig(**parser_kwargs))
    elif parser_name == 'nougat':
        from pdfwf.parsers.nougat_ import NougatParser
        from pdfwf.parsers.nougat_ import NougatParserConfig

        return NougatParser(NougatParserConfig(**parser_kwargs))
    else:
        raise ValueError(f'Unknown parser name: {parser_name}')
