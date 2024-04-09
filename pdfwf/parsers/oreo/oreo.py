"""The Oreo parser for extracting text and visual content from PDFs."""
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any
from typing import Literal, Optional, Union, Optional, List, Dict

if sys.version_info >= (3, 11):
    from typing import Self
else:
    from typing_extensions import Self

import traceback

import torch
from pydantic import field_validator
from pydantic import model_validator

from pdfwf.parsers.base import BaseParser
from pdfwf.parsers.base import BaseParserConfig
from pdfwf.utils import exception_handler


class OreoParserConfig(BaseParserConfig):
    """Configuration for the Oreo parser."""

    # The name of the parser.
    name: Literal['oreo'] = 'oreo'  # type: ignore[assignment]
    # Weights to layout detection model.
    detection_weights_path: Path
    # Model weights for (meta) text classifier.
    text_cls_weights_path: Path
    # Path to the SPV05 category file.
    spv05_category_file_path: Path
    # Path to a local copy of the ultralytics/yolov5 repository.
    yolov5_path: Optional[Path] = None
    # Only scan PDFs for meta statistics on its attributes.
    detect_only: bool = False
    # Only parse PDFs for meta data.
    meta_only: bool = False
    # Include equations into the text categories.
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
    # Number of pixels along which.
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

        # if self.detect_only and check_flags:
        #    raise ValueError(
        #        'The `detect_only` flag cannot be used with any other flag.'
        #    )
        if self.meta_only and check_flags:
            raise ValueError(
                'The `meta_only` flag cannot be used with any other flag.'
            )

        return self


class OreoParser(BaseParser):
    """The Oreo parser."""

    def __init__(self, config: OreoParserConfig) -> None:
        """Initialize the Oreo parser.

        See the `OreoParserConfig` class for the parameter descriptions.
        """
        import torch
        from pylatexenc.latex2text import LatexNodes2Text
        from texify.model.model import load_model
        from texify.model.processor import load_processor

        from pdfwf.parsers.oreo.tensor_utils import get_relevant_text_classes
        from pdfwf.parsers.oreo.tensor_utils import get_relevant_visual_classes

        # Set device
        if hasattr(torch, 'cuda'):
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        elif hasattr(torch, 'xpu'):
            device = torch.device('xpu' if torch.xpu.is_available() else 'cpu')
        else:
            device = torch.device('cpu')
        # load models
        # - (1.) detection: Yolov5
        yolo_path = (
            config.yolov5_path if config.yolov5_path else 'ultralytics/yolov5'
        )
        detect_model = torch.hub.load(
            yolo_path,
            'custom',
            source='local' if config.yolov5_path else 'github',
            path=config.detection_weights_path,
            skip_validation=True,
        )
        detect_model.to(device)
        detect_model.eval()

        # TODO: Determine if subsequent text classification is even useful.
        # - (2.) text classifier for meta data
        # txt_cls_model = AutoModelForSequenceClassification.from_pretrained(
        #    config.text_cls_weights_path
        # ).to(device)

        # tokenizer = AutoTokenizer.from_pretrained(
        #    'distilbert-base-uncased', device=device
        # )
        # txt_cls_model.eval()

        # - (3.) load ViT (i.e. pseudo-OCR) model
        ocr_model = load_model()
        ocr_processor = load_processor()

        # compile
        ocr_model = torch.compile(ocr_model, fullgraph=True)

        # LaTeX -> Tex Decoder
        latex_to_text = LatexNodes2Text()

        # identify relevant classes and group by treatment
        rel_txt_classes = get_relevant_text_classes(
            spv05_category_file_path=config.spv05_category_file_path,
            file_type='pdf',
            meta_only=config.meta_only,
            equation_flag=config.equation,
            table_flag=config.table,
            fig_flag=config.figure,
            secondary_meta=config.secondary_meta,
        )
        rel_meta_txt_classes = get_relevant_text_classes(
            spv05_category_file_path=config.spv05_category_file_path,
            file_type='pdf',
            meta_only=True,
        )
        rel_visual_classes = get_relevant_visual_classes(
            spv05_category_file_path=config.spv05_category_file_path,
            file_type='pdf',
            table_flag=config.table,
            fig_flag=config.figure,
        )

        unpackable_classes = {}

        # determine unpackable_classes
        if config.accelerate:
            # only exclude `meta` cats
            unpackable_classes = rel_meta_txt_classes
        rel_txt_classes.update(rel_meta_txt_classes)

        # Set attributes
        self.config = config
        self.device = device
        self.detect_model = detect_model
        # self.txt_cls_model = txt_cls_model
        # self.tokenizer = tokenizer
        self.ocr_model = ocr_model
        self.ocr_processor = ocr_processor
        self.rel_meta_txt_classes = rel_meta_txt_classes
        self.rel_visual_classes = rel_visual_classes
        self.latex_to_text = latex_to_text
        self.rel_txt_classes: list[int] = list(rel_txt_classes.values())
        self.unpackable_classes: list[int] = list(unpackable_classes.values())

    @torch.no_grad()
    @exception_handler(default_return=None)
    def parse(self, pdf_files: list[str]) -> Optional[List[dict[str, Any]]]:  # noqa: PLR0912, PLR0915
        """Parse a PDF file and extract markdown.

        Parameters
        ----------
        pdf_path : str
            Path to the PDF file to convert.

        Returns
        -------
        list[dict[str, Any]]
            The extracted documents.
        """
        from torch.utils.data import DataLoader

        from pdfwf.parsers.oreo.tensor_utils import accelerated_batch_inference
        from pdfwf.parsers.oreo.tensor_utils import custom_collate
        from pdfwf.parsers.oreo.tensor_utils import format_documents
        from pdfwf.parsers.oreo.tensor_utils import get_packed_patch_tensor
        from pdfwf.parsers.oreo.tensor_utils import PDFDataset
        from pdfwf.parsers.oreo.tensor_utils import pre_processing
        from pdfwf.parsers.oreo.tensor_utils import update_main_content_dict

        # load dataset
        dataset = PDFDataset(
            pdf_paths=pdf_files, meta_only=self.config.meta_only
        )

        # Create a DataLoader for batching and shuffling
        data_loader = DataLoader(
            dataset=dataset,
            batch_size=self.config.batch_yolo,
            shuffle=False,
            collate_fn=custom_collate,
            num_workers=4,
            pin_memory=True,
            prefetch_factor=2,
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
        vis_path_dict: dict[int, dict[int, list[str]]] = {}
        if self.rel_visual_classes:
            raise NotImplementedError(
                'Visual extraction is not yet implemented.'
            )
            # i_tab, i_fig, prev_file_id = 0, 0, -1

        # Iterate through the DataLoader
        for batch in data_loader:
            tensors, file_ids, file_paths = batch
            tensors = tensors.to(self.device)

            try:
                # Yolov5 inference (object detection)
                results = self.detect_model(tensors)
            except Exception:
                traceback.print_exc()
                print('Error in Yolov5 inference. Skipping batch.')
                continue

            try:
                # y : dataframe of patch features
                y = pre_processing(
                    results=results,
                    file_ids=file_ids,
                    rel_class_ids=self.rel_txt_classes,
                    iou_thres=0.001,
                )
            except Exception:
                traceback.print_exc()
                print('Error in pre_processing. Skipping batch.')
                continue

            try:
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
                    offset=self.config.bbox_offset,
                    sep_symbol_tensor=None,
                )
            except Exception:
                traceback.print_exc()
                print('Error in get_packed_patch_tensor. Skipping batch.')
                continue

            # skip empty document batches
            if pack_patch_tensor is None:
                continue

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
            # )

            # no use for page images pass this point
            tensors = None

            # ViT: pseudo-OCR inference
            try:
                text_results = accelerated_batch_inference(
                    tensors=pack_patch_tensor,
                    model=self.ocr_model,
                    processor=self.ocr_processor,
                    batch_size=self.config.batch_vit,
                )
            except torch.cuda.OutOfMemoryError:
                print('OOM error. Skipping batch.')
                continue

            # TODO: *Retool* this. Use (text) Transformer to get text patch
            #       embedding
            #       Regardless, no use for this method anyway as subsequent
            #       text classification does not appear to be favorable for
            #       accuracy
            # re-assess meta text categories
            # index_quadruplet = assign_text_inferred_meta_classes(
            #     txt_cls_model=self.txt_cls_model,
            #     tokenizer=self.tokenizer,
            #     batch_size=self.batch_cls,
            #     index_quadruplet=idx_quad,
            #     text_results=text_results,
            # )

            try:
                # assign decoded text to file docs
                doc_dict = update_main_content_dict(
                    doc_dict=doc_dict,
                    text_results=text_results,
                    index_quadruplet=idx_quad,
                    curr_file_ids=curr_file_ids,
                    vis_path_dict=vis_path_dict,
                )
            except Exception:
                print('Error in update_main_content_dict. Skipping batch.')
                continue

            # Store the file path for each file_id in the batch
            doc_file_paths.update(dict(zip(file_ids, file_paths)))

        # Store the parsed documents
        documents = format_documents(
            doc_dict=doc_dict,
            doc_file_paths=doc_file_paths,
            latex_to_text=self.latex_to_text,
        )

        return documents
