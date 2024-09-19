"""The AdaParse PDF parser."""

from __future__ import annotations

import functools
from abc import ABC
from abc import abstractmethod
from pathlib import Path
from typing import Any
from typing import Literal

import torch
from pydantic import BaseModel
from pydantic import Field
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

from pdfwf.parsers.base import BaseParser
from pdfwf.parsers.nougat_ import NougatParser
from pdfwf.parsers.nougat_ import NougatParserConfig
from pdfwf.parsers.pymupdf import PyMuPDFParser
from pdfwf.parsers.pymupdf import PyMuPDFParserConfig
from pdfwf.timer import Timer
from pdfwf.utils import exception_handler

__all__ = [
    'AdaParse',
    'AdaParseConfig',
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


class TextClassifierConfig(BaseModel):
    """Settings for the text classifier."""

    weights_path: Path = Field(
        description='The path to the fine-tuned model weights.',
    )
    batch_size: int = Field(
        default=8,
        description='The batch size for the classifier.',
    )
    max_character_length: int = Field(
        default=3200,
        description='The maximum length of the input text (in characters).',
    )
    num_data_workers: int = Field(
        default=1,
        description='The number of data workers for the classifier.',
    )
    pin_memory: bool = Field(
        default=True,
        description='Whether to pin memory for the classifier.',
    )


class TextClassifier(ABC):
    """Text classifier."""

    def __init__(self, config: TextClassifierConfig) -> None:
        """Initialize the classifier."""
        from peft import PeftModel
        from transformers import AutoModelForSequenceClassification
        from transformers import AutoTokenizer

        # Load the tokenizer
        tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

        # Load the base model
        model = AutoModelForSequenceClassification.from_pretrained(
            'bert-base-uncased', num_labels=11
        )

        # Load the fine-tuned model with LoRA adapters
        model = PeftModel.from_pretrained(model, config.weights_path)

        # Move the model to the appropriate device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)

        # Set the model to evaluation mode
        model.eval()

        self.config = config
        self.device = device
        self.model = model
        self.tokenizer = tokenizer

    @abstractmethod
    def decision_function(self, logits: torch.Tensor) -> torch.Tensor:
        """Return the decision function.

        Parameters
        ----------
        logits : torch.Tensor
            The model logits.

        Returns
        -------
        torch.Tensor
            The decision function result (tensor of ints).
        """
        ...

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
        # Truncate the text
        _text = [t[: self.config.max_character_length] for t in text]

        # Create the dataset
        dataset = TextDataset(_text)

        # Create the data collator (tokenization function)
        collater_fn = functools.partial(
            self.tokenizer,
            return_tensors='pt',
            truncation=True,
            padding=True,
            return_special_tokens_mask=False,
        )

        # Create the data loader
        dataloader = DataLoader(
            dataset,
            collate_fn=collater_fn,
            batch_size=self.config.batch_size,
            pin_memory=self.config.pin_memory,
            num_workers=self.config.num_data_workers,
        )

        # Collect the predictions
        predictions = []

        # Iterate over each batch of the data loader
        for batch in dataloader:
            # Move the inputs to the appropriate device
            inputs = {k: v.to(self.device) for k, v in batch.items()}

            # Run the model forward pass
            outputs = self.model(**inputs)

            # Call the decision function
            y_pred = self.decision_function(outputs.logits)

            # Collect the predictions
            predictions.extend(y_pred.tolist())

        return predictions


class NougatTextClassifier(TextClassifier):
    """Text classifier for the Nougat parser."""

    def decision_function(self, logits: torch.Tensor) -> torch.Tensor:
        """Return the decision function.

        Parameters
        ----------
        logits : torch.Tensor
            The model logits.

        Returns
        -------
        torch.Tensor
            The decision function result (tensor of ints).
        """
        # Get the predicted classes
        y_pred = logits.argmax(dim=1)

        # We only care about the class 0 (high quality) and class 1
        # (low quality). Assign 0 to class 0 and 1 to all other classes.
        
        # NEW (SIMPLE PERF TEST)
        #probability = 0.05  # 5%
        #mask = torch.rand(y_pred.shape, dtype=torch.float32) < probability
        #mask = mask.to(y_pred.dtype)
        #y_pred[mask.bool()] = 1  # Ensure the mask is boolean for indexing
        # legacy: always 0
        y_pred[y_pred != 0] = 0

        return y_pred


class AdaParseConfig(
    PyMuPDFParserConfig, NougatParserConfig, TextClassifierConfig
):
    """Settings for the AdaParse parser."""

    # The name of the parser.
    name: Literal['adaparse'] = 'adaparse'  # type: ignore[assignment]

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
    def classifier_config(self) -> TextClassifierConfig:
        """Return the text classifier configuration."""
        return TextClassifierConfig(
            weights_path=self.weights_path,
            batch_size=self.batch_size,
            max_character_length=self.max_character_length,
            num_data_workers=self.num_data_workers,
            pin_memory=self.pin_memory,
        )


class AdaParse(BaseParser):
    """Interface for the AdaParse PDF parser."""

    def __init__(self, config: AdaParseConfig) -> None:
        """Initialize the parser."""
        # Initialize the PyMuPDF and Nougat parsers
        self.pymudf_parser = PyMuPDFParser(config=config.pymupdf_config)
        self.nougat_parser = NougatParser(config=config.nougat_config)

        # Initialize the quality check classifier
        # Return a 0 or 1 for each parsed text. If 0, the pdf text, as parsed
        # by pymupdf is of high quality. If not 0, the pdf text should be
        # parsed with Nougat.
        self.classifier = NougatTextClassifier(config=config.classifier_config)

    @exception_handler(default_return=None)
    def parse(self, pdf_files: list[str]) -> list[dict[str, Any]] | None:
        """Parse a list of pdf files and return the parsed data."""
        # First, parse the PDFs using PyMuPDF
        with Timer('adaparse-pymupdf-parsing', self.unique_id):
            documents = self.pymudf_parser.parse(pdf_files)

        # If no documents, there was an error parsing the PDFs with PyMuPDF
        if documents is None:
            return None

        # Apply the quality check regressor
        with Timer('adaparse-quality-check', self.unique_id):
            document_text = [d['text'] for d in documents]
            qualities = self.classifier.predict(document_text)

        # Log the percentage of low-quality documents
        low_quality_num = sum(q != 0 for q in qualities)
        low_quality_percentage = (low_quality_num / len(qualities)) * 100
        print(f'Low-quality documents: {low_quality_percentage:.2f}%')

        # Collect the documents that passed the quality check
        documents = [d for d, q in zip(documents, qualities) if q == 0]

        # Collect the pdf files that failed the quality check
        low_quality_pdfs = [p for p, q in zip(pdf_files, qualities) if q != 0]

        # If no low-quality documents, return the parsed documents
        if not low_quality_pdfs:
            return documents

        # Parse the low-quality documents using the Nougat parser
        with Timer('adaparse-nougat-parsing', self.unique_id):
            nougat_documents = self.nougat_parser.parse(low_quality_pdfs)

        # If Nougat documents were parsed, add them to the output
        if nougat_documents is not None:
            print(f'Nougat parsed documents: {len(nougat_documents)}')
            documents.extend(nougat_documents)

        # Finally, return the parsed documents from both parsers
        return documents
