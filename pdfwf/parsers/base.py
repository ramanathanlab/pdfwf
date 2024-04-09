"""Base parser class for all parsers to inherit from."""



import uuid
from abc import ABC
from abc import abstractmethod
from typing import Any
from typing import Literal, Optional

from pdfwf.utils import BaseModel

# TODO: We should update the parser return type to be a dataclass


class BaseParser(ABC):
    """Base parser class for all parsers to inherit from."""

    @property
    def unique_id(self) -> str:
        """Get the unique identifier for the parser."""
        if not hasattr(self, '_unique_id'):
            self._unique_id = str(uuid.uuid4())
        return self._unique_id

    @abstractmethod
    def parse(self, pdf_files: list[str]) -> Optional[list[dict[str, Any]]]:
        """Parse a list of pdf files and return the parsed data."""
        pass


class BaseParserConfig(BaseModel, ABC):
    """Base settings for all parsers."""

    # The name of the parser.
    name: Literal[''] = ''
