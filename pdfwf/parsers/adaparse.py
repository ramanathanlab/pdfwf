"""The AdaParse PDF parser."""

from __future__ import annotations

import functools
from pathlib import Path
from typing import Any
from typing import Literal

import torch
from pydantic import BaseModel
from pydantic import Field
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

from pdfwf.parsers.base import BaseParser
from pdfwf.parsers.base import BaseParserConfig
from pdfwf.parsers.nougat_ import NougatParser
from pdfwf.parsers.nougat_ import NougatParserConfig
from pdfwf.parsers.pymupdf import PyMuPDFParser
from pdfwf.parsers.pymupdf import PyMuPDFParserConfig
from pdfwf.utils import exception_handler

__all__ = [
    'AdaParse',
    'AdaParseConfig',
]


class TextDataset(Dataset):
    """Dataset for sequence classification."""

    def __init__(self, sequences: list[str]) -> None:
        """Initialize the dataset."""
        self.sequences = sequences

    def __len__(self) -> int:
        """Return the number of sequences."""
        return len(self.sequences)

    def __getitem__(self, idx: int) -> str:
        """Return a sequence."""
        return self.sequences[idx]


class TextClassifierConfig(BaseModel):
    """Settings for the text classifier."""

    weights_path: Path = Field(
        description='The path to the fine-tuned model weights.',
    )
    batch_size: int = Field(
        default=8,
        description='The batch size for the classifier.',
    )
    num_data_workers: int = Field(
        default=1,
        description='The number of data workers for the classifier.',
    )
    pin_memory: bool = Field(
        default=True,
        description='Whether to pin memory for the classifier.',
    )


class TextClassifier:
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
        # Create the dataset
        dataset = TextDataset(text)

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

            # Get the predicted logits
            y_pred = outputs.logits.argmax(dim=1)

            # Collect the predictions
            predictions.extend(y_pred.tolist())

        return predictions


class AdaParseConfig(BaseParserConfig):
    """Settings for the AdaParse parser."""

    # The name of the parser.
    name: Literal['adaparse'] = 'adaparse'  # type: ignore[assignment]

    pymupdf_config: PyMuPDFParserConfig = Field(
        default_factory=PyMuPDFParserConfig,
        description='Settings for the PyMuPDF-PDF parser.',
    )
    nougat_config: NougatParserConfig = Field(
        description='Settings for the Nougat-PDF parser.',
    )
    classifier_config: TextClassifierConfig = Field(
        description='Settings for the text classifier.',
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
        # by pymupdf is of low quality and should be parsed with Nougat.
        self.classifier = TextClassifier(config=config.classifier_config)

    @exception_handler(default_return=None)
    def parse(self, pdf_files: list[str]) -> list[dict[str, Any]] | None:
        """Parse a list of pdf files and return the parsed data."""
        # First, parse the PDFs using PyMuPDF
        documents = self.pymudf_parser.parse(pdf_files)

        # If no documents, there was an error parsing the PDFs with PyMuPDF
        if documents is None:
            return None

        # Apply the quality check regressor
        document_text = [d['text'] for d in documents]
        qualities = self.classifier.predict(document_text)

        # Remove the documents that failed the quality check
        documents = [d for d, q in zip(documents, qualities) if q]

        # Collect the pdf files that failed the quality check
        low_quality_pdfs = [p for p, q in zip(pdf_files, qualities) if not q]

        # If no low-quality documents, return the parsed documents
        if not low_quality_pdfs:
            return documents

        # Parse the low-quality documents using the Nougat parser
        nougat_documents = self.nougat_parser.parse(low_quality_pdfs)

        # If Nougat documents were parsed, add them to the output
        if nougat_documents is not None:
            documents.extend(nougat_documents)

        # Finally, return the parsed documents from both parsers
        return documents
