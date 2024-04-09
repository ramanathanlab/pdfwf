"""Utilities for the PDF workflow."""


import json
import logging
import sys
import traceback
from pathlib import Path
from typing import Any
from typing import Callable
from typing import TypeVar
from typing import Union
from typing import Optional
from typing import List

import yaml
from pydantic import BaseModel as _BaseModel

if sys.version_info >= (3, 10):
    from typing import ParamSpec
else:
    from typing_extensions import ParamSpec

T = TypeVar('T')
P = ParamSpec('P')

PathLike = Union[str, Path]


class BaseModel(_BaseModel):
    """An interface to add JSON/YAML serialization to Pydantic models."""

    def write_json(self, path: PathLike) -> None:
        """Write the model to a JSON file.

        Parameters
        ----------
        path : str
            The path to the JSON file.
        """
        with open(path, 'w') as fp:
            json.dump(self.dict(), fp, indent=2)

    @classmethod
    def from_json(cls: type[T], path: PathLike) -> T:
        """Load the model from a JSON file.

        Parameters
        ----------
        path : str
            The path to the JSON file.

        Returns
        -------
        T
            A specific BaseModel instance.
        """
        with open(path) as fp:
            data = json.load(fp)
        return cls(**data)

    def write_yaml(self, path: PathLike) -> None:
        """Write the model to a YAML file.

        Parameters
        ----------
        path : str
            The path to the YAML file.
        """
        with open(path, mode='w') as fp:
            yaml.dump(json.loads(self.json()), fp, indent=4, sort_keys=False)

    @classmethod
    def from_yaml(cls: type[T], path: PathLike) -> T:
        """Load the model from a YAML file.

        Parameters
        ----------
        path : PathLike
            The path to the YAML file.

        Returns
        -------
        T
            A specific BaseModel instance.
        """
        with open(path) as fp:
            raw_data = yaml.safe_load(fp)
        return cls(**raw_data)


def exception_handler(
    default_return: Any = None,
) -> Callable[[Callable[P, T]], Callable[P, T]]:
    """Handle exceptions in a function by returning a `default_return` value.

    A decorator factory that returns a decorator formatted with the
    default_return that wraps a function.
    """

    def decorator(func: Callable[P, T]) -> Callable[P, T]:
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> Union[T, Any]:
            try:
                return func(*args, **kwargs)
            except Exception as e:
                print(
                    f'{func.__name__} raised an exception: {e} '
                    f'On input {args}, {kwargs}\nReturning {default_return}',
                )
                traceback.print_exc()
                return default_return

        return wrapper

    return decorator


def setup_logging(
    logger_name: str, out_dir: Optional[Path] = None
) -> logging.Logger:
    """Set up logging for the PDF workflow."""
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)

    # Create the output directory if it does not exist
    if out_dir is not None:
        out_dir.mkdir(exist_ok=True, parents=True)

    # Set the format for the log messages
    formatter = logging.Formatter(
        '[%(asctime)s] [%(levelname)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
    )

    # Add a console log
    handlers: List[logging.Handler] = [logging.StreamHandler()]

    # Add a file log if an output directory is provided
    if out_dir is not None:
        handlers.append(logging.FileHandler(out_dir / f'{logger_name}.log'))

    # Set the format for the log messages
    for handler in handlers:
        handler.setLevel(logging.INFO)
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger


def batch_data(data: List[T], chunk_size: int) -> List[List[T]]:
    """Batch data into chunks of size chunk_size."""
    batches = [
        data[i * chunk_size : (i + 1) * chunk_size]
        for i in range(0, len(data) // chunk_size)
    ]
    if len(data) > chunk_size * len(batches):
        batches.append(data[len(batches) * chunk_size :])
    return batches
