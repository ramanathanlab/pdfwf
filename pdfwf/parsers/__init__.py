"""The parsers module storing different PDF parsers."""


from typing import Any, Union, Tuple, Dict

from pdfwf.parsers.base import BaseParser
from pdfwf.parsers.base import BaseParserConfig
from pdfwf.parsers.marker import MarkerParser
from pdfwf.parsers.marker import MarkerParserConfig
from pdfwf.parsers.nougat_ import NougatParser
from pdfwf.parsers.nougat_ import NougatParserConfig
from pdfwf.parsers.oreo import OreoParser
from pdfwf.parsers.oreo import OreoParserConfig
from pdfwf.registry import registry

ParserConfigTypes = Union[MarkerParserConfig, OreoParserConfig, NougatParserConfig]
ParserTypes = Union[MarkerParser, OreoParser, NougatParser]

_ParserTypes = Tuple[type[ParserConfigTypes], type[ParserTypes]]

STRATEGIES: Dict[str, _ParserTypes] = {
    'marker': (MarkerParserConfig, MarkerParser),
    'oreo': (OreoParserConfig, OreoParser),
    'nougat': (NougatParserConfig, NougatParser),
}


# This is a workaround to support optional registration.
# Make a function to combine the config and instance initialization
# since the registry only accepts functions with hashable arguments.
def _factory_fn(**kwargs: Dict[str, Any]) -> ParserTypes:
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
    kwargs: Dict[str, Any],
    register: bool = False,
) -> ParserTypes:
    """Get the instance based on the kwargs.

    Currently supports the following strategies:
    - marker
    - oreo
    - nougat

    Parameters
    ----------
    kwargs : Dict[str, Any]
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
