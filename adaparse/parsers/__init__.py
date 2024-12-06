"""The parsers module storing different PDF parsers."""

from __future__ import annotations

from typing import Any

from adaparse.parsers.adaparse import AdaParse
from adaparse.parsers.adaparse import AdaParseConfig
from adaparse.parsers.base import BaseParser
from adaparse.parsers.base import BaseParserConfig
from adaparse.parsers.marker import MarkerParser
from adaparse.parsers.marker import MarkerParserConfig
from adaparse.parsers.nougat_ import NougatParser
from adaparse.parsers.nougat_ import NougatParserConfig
from adaparse.parsers.pymupdf import PyMuPDFParser
from adaparse.parsers.pymupdf import PyMuPDFParserConfig
from adaparse.parsers.pypdf import PyPDFParser
from adaparse.parsers.pypdf import PyPDFParserConfig
from adaparse.parsers.tesseract import TesseractParser
from adaparse.parsers.tesseract import TesseractParserConfig
from adaparse.registry import registry

# from adaparse.parsers.oreo import OreoParser
# from adaparse.parsers.oreo import OreoParserConfig
ParserConfigTypes = (
    # | AdaFastConfig
    AdaParseConfig
    | MarkerParserConfig
    | NougatParserConfig
    | PyMuPDFParserConfig
    | PyPDFParserConfig
    | TesseractParserConfig
)
ParserTypes = (
    # | AdaFast
    AdaParse
    | MarkerParser
    | NougatParser
    | PyMuPDFParser
    | PyPDFParser
    | TesseractParser
)
_ParserTypes = tuple[type[ParserConfigTypes], type[ParserTypes]]

STRATEGIES: dict[str, _ParserTypes] = {
    'adaparse': (AdaParseConfig, AdaParse),
    'marker': (MarkerParserConfig, MarkerParser),
    'nougat': (NougatParserConfig, NougatParser),
    'pymupdf': (PyMuPDFParserConfig, PyMuPDFParser),
    'pypdf': (PyPDFParserConfig, PyPDFParser),
    'tesseract': (TesseractParserConfig, TesseractParser),
}


# This is a workaround to support optional registration.
# Make a function to combine the config and instance initialization
# since the registry only accepts functions with hashable arguments.
def _factory_fn(**kwargs: dict[str, Any]) -> ParserTypes:
    name = kwargs.get('name', '')
    strategy = STRATEGIES.get(name)  # type: ignore[arg-type]
    if not strategy:
        raise ValueError(
            f'Unknown parser name: {name}.'
            f' Available: {set(STRATEGIES.keys())}',
        )

    # Get the config and classes
    config_cls, cls = strategy

    return cls(config_cls(**kwargs))


def get_parser(
    kwargs: dict[str, Any],
    register: bool = False,
) -> ParserTypes:
    """Get the instance based on the kwargs.

    Currently supports the following strategies:
    - adaparse
    - marker
    - oreo
    - nougat
    - pymupdf
    - pypdf
    - tesseract

    Parameters
    ----------
    kwargs : dict[str, Any]
        The configuration. Contains a `name` argument
        to specify the strategy to use.
    register : bool, optional
        Register the instance for warmstart. Caches the
        instance based on the kwargs, by default False.

    Returns
    -------
    ParserTypes
        The instance.

    Raises
    ------
    ValueError
        If the `name` is unknown.
    """
    # Create and register the instance
    if register:
        registry.register(_factory_fn)
        return registry.get(_factory_fn, **kwargs)

    return _factory_fn(**kwargs)
