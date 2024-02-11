"""Base parser class for all parsers to inherit from."""

from __future__ import annotations

import uuid
from abc import ABC
from abc import abstractmethod
from typing import Any
from typing import Literal

from pdfwf.utils import BaseModel

# TODO: We should update the parser return type to be a dataclass


class BaseParser(ABC):
    """Base parser class for all parsers to inherit from."""

    @property
    def id(self) -> str:
        """Get the unique identifier for the parser."""
        if not hasattr(self, '_id'):
            self._id = str(uuid.uuid4())
        return self._id

    @abstractmethod
    def parse(self, pdf_files: list[str]) -> list[dict[str, Any]]:
        """Parse a list of pdf files and return the parsed data."""
        pass


class BaseParserSettings(BaseModel, ABC):
    """Base settings for all parsers."""

    name: Literal[''] = ''
    """Name of the parser to use."""
