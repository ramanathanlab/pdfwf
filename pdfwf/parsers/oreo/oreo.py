"""The Oreo parser for extracting text and visual content from PDFs."""
from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

if sys.version_info >= (3, 10):
    from typing import Self
else:
    from typing_extensions import Self

import torch
from pydantic import field_validator
from pydantic import model_validator

from pdfwf.parsers.base import BaseParser
from pdfwf.parsers.base import BaseParserSettings


class OreoParserSettings(BaseParserSettings):
    """Configuration for the Oreo parser."""

    # The name of the parser
    name: str = 'oreo'
    # Weights to layout detection model.
    detection_weights_path: Path
    # Model weights for (meta) text classifier.
    text_cls_weights_path: Path
    # File type to be parsed (ignores other files in the input_dir).
    detect_only: bool = False
    # Only parse PDFs for meta data
    meta_only: bool = False
    # Include equations into the text categories
    equation: bool = False
    # Include table visualizations (will be stored).
    table: bool = False
    # Include figure  (will be stored).
    figure: bool = False
    # Include secondary meta data (footnote, headers).
    secondary_meta: bool = False
    # If true, accelerate inference by packing non-meta text patches.
    accelerate: bool = False
    # Main batch size for detection/# of images loaded per batch.
    batch_yolo: int = 128
    # Batch size of pre-processed patches for ViT pseudo-OCR inference.
    batch_vit: int = 512
    # Batch size K for subsequent text processing.
    batch_cls: int = 512
    # Number of pixels along which
    bbox_offset: int = 2

    @field_validator('detection_weights_path', 'text_cls_weights_path')
    @classmethod
    def validate_path_existence(cls, value: Path) -> Path:
        """Check if the directory exists."""
        if not value.exists():
            raise FileNotFoundError(f'Path does not exist: {value}')
        return value

    @model_validator(mode='after')
    def validate_flags(self) -> Self:
        """Validate the flags."""
        check_flags = (
            self.equation
            or self.table
            or self.figure
            or self.secondary_meta
            or self.meta_only
        )

        if self.detect_only and check_flags:
            raise ValueError(
                'The `detect_only` flag cannot be used with any other flag.'
            )
        if self.meta_only and check_flags:
            raise ValueError(
                'The `meta_only` flag cannot be used with any other flag.'
            )

        return self


class OreoParser(BaseParser):
    """The Oreo parser."""

    def __init__(
        self,
        detection_weights_path: Path,
        txt_cls_weights_path: Path,
        meta_only: bool,
        equation_flag: bool,
        table_flag: bool,
        fig_flag: bool,
        secondary_meta: bool,
        accelerate_flag: bool,
        batch_yolo: int,
        batch_vit: int,
        batch_cls: int,
        bbox_offset: int,
    ) -> None:
        """Initialize the Oreo parser.

        See the `OreoParserSettings` class for the parameter descriptions.
        """
        import torch
        from pylatexenc.latex2text import LatexNodes2Text
        from tensor_utils import get_relevant_text_classes
        from tensor_utils import get_relevant_visual_classes
        from texify.model.model import load_model
        from texify.model.processor import load_processor
        from transformers import AutoModelForSequenceClassification
        from transformers import AutoTokenizer

        # Set device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # load models
        # - (1.) detection: Yolov5
        detect_model = torch.hub.load(
            'ultralytics/yolov5',
            'custom',
            path=detection_weights_path,
            force_reload=True,
        )
        detect_model.to(device)
        detect_model.eval()

        # - (2.) text classifier for meta data
        txt_cls_model = AutoModelForSequenceClassification.from_pretrained(
            txt_cls_weights_path
        ).to(device)

        tokenizer = AutoTokenizer.from_pretrained(
            'distilbert-base-uncased', device=device
        )
        txt_cls_model.eval()

        # - (3.) load ViT (i.e. pseudo-OCR) model
        ocr_model = load_model()
        ocr_processor = load_processor()

        # compile
        ocr_model = torch.compile(ocr_model, fullgraph=True)

        # LaTeX -> Tex Decoder
        latex_to_text = LatexNodes2Text()

        # identify relevant classes and group by treatment
        rel_txt_classes = get_relevant_text_classes(
            'pdf',
            meta_only,
            equation_flag,
            table_flag,
            fig_flag,
            secondary_meta,
        )
        rel_meta_txt_classes = get_relevant_text_classes('pdf', meta_only=True)
        rel_visual_classes = get_relevant_visual_classes(
            'pdf', table_flag=table_flag, fig_flag=fig_flag
        )

        unpackable_classes = {}

        # determine unpackable_classes
        if accelerate_flag:
            # only exclude `meta` cats
            unpackable_classes = rel_meta_txt_classes
        rel_txt_classes.update(rel_meta_txt_classes)

        # Set attributes
        self.meta_only = meta_only
        self.batch_yolo = batch_yolo
        self.batch_vit = batch_vit
        self.batch_cls = batch_cls
        self.bbox_offset = bbox_offset
        self.device = device
        self.detect_model = detect_model
        self.txt_cls_model = txt_cls_model
        self.tokenizer = tokenizer
        self.ocr_model = ocr_model
        self.ocr_processor = ocr_processor
        self.rel_meta_txt_classes = rel_meta_txt_classes
        self.rel_visual_classes = rel_visual_classes
        self.latex_to_text = latex_to_text
        self.rel_txt_classes: list[int] = list(rel_txt_classes.values())
        self.unpackable_classes: list[int] = list(unpackable_classes.values())

    @torch.no_grad()
    def parse(self, pdf_files: list[str]) -> list[dict[str, Any]]:
        """Parse a PDF file and extract markdown.

        Parameters
        ----------
        pdf_path : str
            Path to the PDF file to convert.

        Returns:
        -------
        list[dict[str, Any]]
            The extracted documents.
        """
        from tensor_utils import accelerated_batch_inference
        from tensor_utils import assign_text_inferred_meta_classes
        from tensor_utils import custom_collate
        from tensor_utils import format_documents
        from tensor_utils import get_packed_patch_tensor
        from tensor_utils import PDFDataset
        from tensor_utils import pre_processing
        from tensor_utils import update_main_content_dict
        from torch.utils.data import DataLoader

        # load dataset
        dataset = PDFDataset(pdf_paths=pdf_files, meta_only=self.meta_only)

        # TODO: Experiment with num_workers and pin_memory
        # Create a DataLoader for batching and shuffling
        data_loader = DataLoader(
            dataset=dataset,
            batch_size=self.batch_yolo,
            shuffle=False,
            collate_fn=custom_collate,
            # num_workers=1,
            # pin_memory=True
        )

        # Maps the keys [Text, Title, Keywords, Tables, Figures, Equations
        # Author] to a another dictionary containing the file_id and the list
        # of text
        doc_dict: dict[str, dict[int, list[str]]] = defaultdict(
            lambda: defaultdict(list)
        )
        # Maps the file_id to the file_path
        doc_file_paths: dict[int, Path] = {}

        # init visual extraction variables
        if self.rel_visual_classes:
            raise NotImplementedError(
                'Visual extraction is not yet implemented.'
            )
            # i_tab, i_fig, prev_file_id = 0, 0, -1
        else:
            vis_path_dict = {}

        # Iterate through the DataLoader
        for batch in data_loader:
            tensors, file_ids, file_paths = batch
            tensors = tensors.to(self.device)

            # Yolov5 inference (object detecion)
            results = self.detect_model(tensors)

            # y : dataframe of patch features
            y = pre_processing(
                results=results,
                file_ids=file_ids,
                rel_class_ids=self.rel_txt_classes,
                iou_thres=0.001,
            )

            # metadata specific extraction
            (
                pack_patch_tensor,
                idx_quad,
                curr_file_ids,
            ) = get_packed_patch_tensor(
                tensors=tensors,
                y=y,
                rel_class_ids=self.rel_txt_classes,
                unpackable_class_ids=self.unpackable_classes,
                sep_symbol_flag=False,
                btm_pad=4,
                by=['file_id'],
                offset=self.bbox_offset,
                sep_symbol_tensor=None,
            )

            # TODO: Implement this such that the I/O is decoupled from the
            #       main parsing logic.
            # store visual patches (tables, figures)
            # if self.rel_visual_classes:
            # vis_path_dict, i_tab, i_fig, prev_file_id = store_visuals(
            #     tensors=tensors,
            #     y=y,
            #     rel_visual_classes=self.rel_visual_classes,
            #     file_paths=file_paths,
            #     file_ids=file_ids,
            #     output_dir=args.output_dir,
            #     i_tab=i_tab,
            #     i_fig=i_fig,
            #     prev_file_id=prev_file_id,
            # )

            # no use for page images pass this point
            tensors = None

            # ViT: pseudo-OCR inference
            text_results = accelerated_batch_inference(
                tensors=pack_patch_tensor,
                model=self.ocr_model,
                processor=self.ocr_processor,
                batch_size=self.batch_vit,
            )

            # re-assess meta text categories
            index_quadruplet = assign_text_inferred_meta_classes(
                txt_cls_model=self.txt_cls_model,
                tokenizer=self.tokenizer,
                batch_size=self.batch_cls,
                index_quadruplet=idx_quad,
                text_results=text_results,
            )

            # assign decoded text to file docs
            doc_dict = update_main_content_dict(
                doc_dict=doc_dict,
                text_results=text_results,
                index_quadruplet=index_quadruplet,
                curr_file_ids=curr_file_ids,
                vis_path_dict=vis_path_dict,
            )

            # Store the file path for each file_id in the batch
            doc_file_paths.update(dict(zip(file_ids, file_paths)))

        # Store the parsed documents
        documents = format_documents(
            doc_dict=doc_dict,
            doc_file_paths=doc_file_paths,
            LaTex2Text=self.latex_to_text,
        )

        return documents
