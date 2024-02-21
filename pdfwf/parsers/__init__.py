"""The parsers module storing different PDF parsers."""
from __future__ import annotations

from typing import Any

from pdfwf.parsers.base import BaseParser
from pdfwf.parsers.base import BaseParserConfig
from pdfwf.parsers.marker import MarkerParser
from pdfwf.parsers.marker import MarkerParserConfig
from pdfwf.parsers.nougat_ import NougatParser
from pdfwf.parsers.nougat_ import NougatParserConfig
from pdfwf.parsers.oreo import OreoParser
from pdfwf.parsers.oreo import OreoParserConfig
from pdfwf.registry import registry

ParserConfigTypes = MarkerParserConfig | OreoParserConfig | NougatParserConfig
ParserTypes = MarkerParser | OreoParser | NougatParser

_ParserTypes = tuple[type[ParserConfigTypes], type[ParserTypes]]

PARSER_STRATEGIES: dict[str, _ParserTypes] = {
    'marker': (MarkerParserConfig, MarkerParser),
    'oreo': (OreoParserConfig, OreoParser),
    'nougat': (NougatParserConfig, NougatParser),
}


def get_parser(
    parser_kwargs: dict[str, Any],
    register: bool = False,
) -> ParserTypes:
    """Get the parser instance based on the parser name and kwargs.

    Caches the parser instance based on the parser name and kwargs.
    Currently supports the following parsers: marker, oreo, and nougat.

    Parameters
    ----------
    parser_kwargs : dict[str, Any]
        The parser configuration. Contains an extra `name` argument
        to specify the parser to use.
    register : bool, optional
        Register the parser instance for warmstart, by default False.

    Returns
    -------
    ParserTypes
        The parser instance.

    Raises
    ------
    ValueError
        If the embedder name is unknown.
    """
    name = parser_kwargs.get('name', '')
    parser_strategy = PARSER_STRATEGIES.get(name)
    if not parser_strategy:
        raise ValueError(f'Unknown parser name: {name}')

    # Unpack the parser strategy
    config_cls, parser_cls = parser_strategy

    # Make a function to combine the config and parser initialization
    # since the registry only accepts functions with hashable arguments.
    def parser_factory(**parser_kwargs: dict[str, Any]) -> ParserTypes:
        # Create the parser config
        config = config_cls(**parser_kwargs)
        # Create the parser instance
        return parser_cls(config)

    # Register and create the parser instance
    if register:
        registry.register(parser_factory)
        parser = registry.get(parser_factory, **parser_kwargs)
    else:
        parser = parser_factory(**parser_kwargs)

    return parser
