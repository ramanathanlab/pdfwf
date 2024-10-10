"""The AdaParse PDF parser."""

from __future__ import annotations

import functools
from abc import ABC
from abc import abstractmethod
from pathlib import Path
from typing import Any
from typing import Literal
import joblib

import torch
import numpy as np
import fasttext
import fasttext.util
import os
from sklearn.linear_model import Ridge

from pydantic import BaseModel
from pydantic import Field
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

from concurrent.futures import ProcessPoolExecutor
from typing import Any, List, Dict

from pdfwf.parsers.base import BaseParser
from pdfwf.parsers.nougat_ import NougatParser
from pdfwf.parsers.nougat_ import NougatParserConfig
from pdfwf.parsers.pymupdf import PyMuPDFParser
from pdfwf.parsers.pymupdf import PyMuPDFParserConfig
from pdfwf.timer import Timer
from pdfwf.utils import exception_handler

__all__ = [
    'AdaFast',
    'AdaFastConfig',
]


class TextDataset(Dataset):
    """Dataset for sequence classification."""

    def __init__(self, texts: list[str]) -> None:
        """Initialize the dataset."""
        self.texts = texts

    def __len__(self) -> int:
        """Return the number of text."""
        return len(self.texts)

    def __getitem__(self, idx: int) -> str:
        """Return a sequence."""
        return self.texts[idx]


class TextRegressorConfig(BaseModel):
    """Settings for the text classifier."""

    scikit_path: Path = Field(
        description='The path to the scitkit-learn (multivariate) Ridge regression model.',
    )
    fasttext_dir: Path = Field(
        description='The temporary dir for fasttext.'
    )
    max_character_length: int = Field(
        default=1600,
        description='The maximum length of the input text (in characters) for stat. model.',
    )
    min_character_length: int = Field(
        default=10,
        description='The minimum length of the input text to not go directly into Nougat.',
    )
    num_data_workers: int = Field(
        default=1,
        description='The number of data workers for the classifier.',
    )


class TextRegressor(ABC):
    """Text classifier."""

    def __init__(self, config: TextRegressorConfig) -> None:
        """Initialize the classifier."""

        # load ridge regression model
        self.config = config
        self.model = joblib.load(self.config.scikit_path)
        self.fasttext_dir = self.config.fasttext_dir

        # load
        self.fasttext_model = fasttext.load_model(f'{self.fasttext_dir}/cc.en.300.bin')
        fasttext.util.reduce_model(self.fasttext_model, 100)

    @abstractmethod
    def decision_function(self, preds: np.array, sign_threshold: float) -> np.array:
        """Return the decision function.

        Parameters
        ----------
        preds : np.array
            Predicted response matrix Y from multivariate BLEU predictor

        sign_threshold: float
            Significance threshold after which Nougat should be triggered.
            (Translates predicted BLEU difference into a binary decision)

        Returns
        -------
        torch.Tensor
            The decision function result (tensor of ints).
        """

        pass

    @torch.no_grad()
    def predict(self, text: list[str]) -> list[int]:
        """Classify the input text.

        Parameters
        ----------
        text : list[str]
            The input text to classify.

        Returns
        -------
        list[int]
            The predicted classes.
        """

        # save the training data to a file
        tmp_train_file_path = Path(self.fasttext_dir) / "train.txt"
        truncated_text_list = [t_elem if len(t_elem) < self.config.max_character_length else t_elem[:self.config.max_character_length] for t_elem in text]

        # clean
        #cleaned_text_list = [text.replace('\n', ' ') for text in truncated_text_list]

        # DEBUG
        #print('len(cleaned_text_list) : ', len(cleaned_text_list))

        # - store locally
        #with open(tmp_train_file_path, "w") as f:
        #    f.write("\n".join(cleaned_text_list))

        # Load fastText model
        #ft_model = fasttext.train_unsupervised(str(tmp_train_file_path), model='skipgram')

        # inner fnc to turn text to vectors
        def text_to_vector(text_series):
            #return np.vstack([ft_model.get_sentence_vector(text) for text in text_series]) # LEGACY
            return np.vstack([self.fasttext_model.get_sentence_vector(text.replace('\n', ' ')) for text in text_series]) # next attempt

        # numerical repr. of text (via fasttext embedding)
        X_vec = text_to_vector(truncated_text_list)

        # generate prediction
        predictions = self.model.predict(X_vec)

        # DEBUG
        print('predictions.shape: ', predictions.shape)

        return predictions

# TODO
class NougatTextClassifier(TextRegressor):
    """Text Regressor that selective launches Nougat parser."""

    def decision_function(self, preds: np.array, sign_threshold: float = 0.1) -> np.array:
        """Return the decision function to trigger Nougat parser.

        Parameters
        ----------
        preds : np.array
            The model predictions (2D array) of PyMuPDF vs. Nougat accuracy.
        sign_threshold : float
            Significance threshold after which Nougat should be triggered.

        Returns
        -------
        np.array
            The decision function result (array of 1s and 0s).
        """
        # Compute the difference between the two columns (predictions)
        preds_diff = preds[:, 0] - preds[:, 1]

        # Return 1 if the difference exceeds the threshold, otherwise 0
        decision_result = np.where(preds_diff > sign_threshold, 1, 0)

        return decision_result


class AdaFastConfig(
    PyMuPDFParserConfig, NougatParserConfig, TextRegressorConfig
):
    """Settings for the AdaFast parser."""

    # The name of the parser.
    name: Literal['adafast'] = 'adafast'  # type: ignore[assignment]

    # DEV NOTE: The following are convenience properties to access the
    # individual parser configurations (we need a flat configuration for
    # the parser to be compatible with the warmstart registry module).
    @property
    def pymupdf_config(self) -> PyMuPDFParserConfig:
        """Return the PyMuPDF parser configuration."""
        return PyMuPDFParserConfig()

    @property
    def nougat_config(self) -> NougatParserConfig:
        """Return the Nougat parser configuration."""
        return NougatParserConfig(
            batchsize=self.batchsize,
            num_workers=self.num_workers,
            prefetch_factor=self.prefetch_factor,
            checkpoint=self.checkpoint,
            mmd_out=self.mmd_out,
            recompute=self.recompute,
            full_precision=self.full_precision,
            markdown=self.markdown,
            skipping=self.skipping,
            nougat_logs_path=self.nougat_logs_path,
        )

    @property
    def regressor_config(self) -> TextRegressorConfig:
        """Return the text classifier configuration."""
        return TextRegressorConfig(
            scikit_path=self.scikit_path,
            fasttext_dir=self.fasttext_dir,
            max_character_length=self.max_character_length,
            min_character_length=self.min_character_length,
            num_data_workers=self.num_data_workers,
        )

class AdaFast(BaseParser):
    """Interface for the AdaFast (not AdaParse!) PDF parser."""

    def __init__(self, config: AdaFastConfig) -> None:
        """Initialize the parser."""
        # Initialize the PyMuPDF and Nougat parsers
        self.pymudf_parser = PyMuPDFParser(config=config.pymupdf_config)
        self.nougat_parser = NougatParser(config=config.nougat_config)

        # Initialize the quality check classifier
        # Return a 0 or 1 for each parsed text. If 0, the pdf text, as parsed
        # by pymupdf is of high quality. If not 0, the pdf text should be
        # parsed with Nougat.
        self.classifier = NougatTextClassifier(config=config.regressor_config)

    @exception_handler(default_return=None)
    def parse(self, pdf_files: list[str]) -> list[dict[str, Any]] | None:
        """Parse a list of pdf files and return the parsed data."""

        # enable parallelism
        #def process_file(pdf_file: str) -> dict[str, Any]:
        #    """Helper function to parse a single PDF file."""
        #    return self.pymudf_parser.parse([pdf_file])

        # First, parse the PDFs using PyMuPDF
        with Timer('adafast-pymupdf-parsing', self.unique_id):
            #with ProcessPoolExecutor(max_workers=4) as executor:
            #    # Submit tasks for each PDF file in parallel
            #    documents = list(executor.map(process_file, pdf_files))
            #    # DEBUG
            #    print('len(documents) : ', len(documents))
            # works?!
            documents = self.pymudf_parser.parse(pdf_files)
            # DEBUG
            print('Post ... self.pymudf_parser.parse .... len(documents) = ', len(documents))

        # If no documents, there was an error parsing the PDFs with PyMuPDF
        if documents is None:
            return None

        # - init instant Nougat candidates (when no text could be extracted)
        qualities = [1 if len(d['text']) <= self.classifier.config.min_character_length else None for d in documents]

        # - fasttext/ridge reg inference on the other docs
        filtered_document_text = [d['text'] for d in documents if len(d['text']) > self.classifier.config.min_character_length]

        # Apply the quality check regressor
        with Timer('adafast-quality-check', self.unique_id):
            filtered_qualities = self.classifier.decision_function(self.classifier.predict(filtered_document_text))
            #print('filtered_qualities : ', filtered_qualities)

        # Insert filtered_qualities back into the correct positions
        filtered_index = 0
        for i, d in enumerate(documents):
            if len(d['text']) > self.classifier.config.min_character_length:
                qualities[i] = filtered_qualities[filtered_index]
                filtered_index += 1

                # Log the percentage of low-quality documents
                low_quality_num = sum(q != 0 for q in qualities)
                low_quality_percentage = (low_quality_num / len(qualities)) * 100
                #print(f'Low-quality documents: {low_quality_percentage:.2f}%')

        # Collect the documents that passed the quality check
        documents = [d for d, q in zip(documents, qualities) if q == 0]

        # Collect the pdf files that failed the quality check
        low_quality_pdfs = [p for p, q in zip(pdf_files, qualities) if q != 0]

        # If no low-quality documents, return the parsed documents
        if not low_quality_pdfs:
            print('No low quality pdfs.. return len(documents) = ', len(documents))
            return documents

        # Parse the low-quality documents using the Nougat parser
        with Timer('adafast-nougat-parsing', self.unique_id):
            nougat_documents = self.nougat_parser.parse(low_quality_pdfs)
            print('nougat_documentsL len(nougat_documents) = ', len(nougat_documents))

        # DEBUG
        print(f'len(documents) : {len(documents)}')

        # If Nougat documents were parsed, add them to the output
        if nougat_documents is not None:
            print(f'Nougat parsed documents: {len(nougat_documents)}')
            documents.extend(nougat_documents)

        # DEBUG
        print('Before return documents ...: ... len(documents) = ', len(documents))

        # DEBUG
        #print(f'len(nougat_documents) : {len(nougat_documents)}')
        #print(f'len(nougat_documents) : {len(nougat_documents)}')

        # Finally, return the parsed documents from both parsers
        return documents
