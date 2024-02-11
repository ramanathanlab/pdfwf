from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
from typing import Any

from pdfwf.parsers.base import BaseParser


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    # parser
    parser = argparse.ArgumentParser(
        description='Process command line arguments'
    )

    # paths
    parser.add_argument(
        '-in',
        '--input_dir',
        type=str,
        help='Input directory path from which document files are sourced.',
    )
    parser.add_argument(
        '-out',
        '--output_dir',
        type=str,
        help='Output path into which json/visuals are stored.',
    )
    parser.add_argument(
        '-f',
        '--file_type',
        type=str,
        default='pdf',
        help='File type to be parsed (ignores other files in the input_dir).',
    )

    # boolean arguments (subsetting relevant classes)
    parser.add_argument(
        '--detect_only',
        action='store_true',
        help='Only scan PDFs for meta statistics on its attributes.',
    )
    parser.add_argument(
        '-mo',
        '--meta_only',
        action='store_true',
        help='Only parse PDFs for meta data',
    )
    parser.add_argument(
        '-eq',
        '--equation',
        action='store_true',
        help='Include equations into the text categories',
    )
    parser.add_argument(
        '-tab',
        '--table',
        action='store_true',
        help='Include table visualizations (will be stored).',
    )
    parser.add_argument(
        '-fig',
        '--figure',
        action='store_true',
        help='Include figure  (will be stored).',
    )
    parser.add_argument(
        '-sec',
        '--secondary_meta',
        action='store_true',
        help='Include secondary meta data (footnote, headers).',
    )
    parser.add_argument(
        '-a',
        '--accelerate',
        action='store_true',
        help='If true, accelerate inference by packing non-meta text patches.',
    )

    # batch sizes (tuned for single-GPU performance)
    parser.add_argument(
        '--batch_yolo',
        type=int,
        default=128,
        help='Main batch size for detection/# of images loaded per batch.',
    )
    parser.add_argument(
        '--batch_vit',
        type=int,
        default=512,
        help='Batch size N for number of pre-processed patches for ViT pseudo-OCR inference.',
    )
    parser.add_argument(
        '--batch_cls',
        type=int,
        default=512,
        help='Batch size K for subsequent text processing.',
    )

    # finetuning parameters
    parser.add_argument(
        '--bbox_offset',
        type=int,
        default=2,
        help='Number of pixels along which',
    )
    parser.add_argument(
        '--dtype',
        type=torch.dtype,
        default=torch.float16,
        help='Dtype of ViT OCR model.',
    )

    # model weights
    parser.add_argument(
        '--detection_weights',
        type=str,
        default='./yolov5/runs/train/best_SPv05_run/weights/best.pt',
        help='Weights to layout detection model.',
    )
    parser.add_argument(
        '--text_cls_weights',
        type=str,
        default='./text_classifier/meta_text_classifier',
        help='Model weights for (meta) text classifier.',
    )

    # parse arguments
    args = parser.parse_args()
    return args


class OreoParserSettings(BaseParser):
    """Configuration for the Oreo parser."""

    input_dir: Path
    output_dir: Path
    file_type: str = 'pdf'
    detect_only: bool = False
    meta_only: bool = False
    equation: bool = False
    table: bool = False
    figure: bool = False
    secondary_meta: bool = False
    accelerate: bool = False
    batch_yolo: int = 128
    batch_vit: int = 512
    batch_cls: int = 512
    bbox_offset: int = 2
    dtype: torch.dtype = torch.float16
    detection_weights_path: Path = Path(
        './yolov5/runs/train/best_SPv05_run/weights/best.pt'
    )
    text_cls_weights_path: Path = Path(
        './text_classifier/meta_text_classifier'
    )


class OreoParser(BaseParser):
    """The Oreo PDF parser."""

    def __init__(self) -> None:
        """Initialize the Oreo parser."""
        import torch
        from pylatexenc.latex2text import LatexNodes2Text
        from tensor_utils import get_relevant_text_classes
        from tensor_utils import get_relevant_visual_classes
        from texify2.texify.model.model import load_model
        from texify2.texify.model.processor import load_processor
        from transformers import AutoModelForSequenceClassification
        from transformers import AutoTokenizer

        # Set device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # load models
        # - (1.) detection: Yolov5
        detect_model = torch.hub.load(
            'ultralytics/yolov5',
            'custom',
            path=args.detection_weights_path,
            force_reload=True,
        )
        detect_model.to(device)
        detect_model.eval()

        # - (2.) text classifier for meta data
        txt_cls_model = AutoModelForSequenceClassification.from_pretrained(
            args.txt_cls_weights_path
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
        LaTex2Text = LatexNodes2Text()

        # identify relevant classes and group by treatment
        rel_txt_classes = get_relevant_text_classes(
            'pdf',
            args.meta_only,
            args.equation_flag,
            args.table_flag,
            args.fig_flag,
            args.secondary_meta,
        )
        rel_meta_txt_classes = get_relevant_text_classes('pdf', meta_only=True)
        rel_visual_classes = get_relevant_visual_classes(
            'pdf', table_flag=args.table_flag, fig_flag=args.fig_flag
        )

        unpackable_classes = {}

        # determine unpackable_classes
        if args.accelerate_flag:
            # only exclude `meta` cats
            unpackable_classes = rel_meta_txt_classes
        rel_txt_classes.update(rel_meta_txt_classes)

        # Set attributes
        self.device = device
        self.detect_model = detect_model
        self.txt_cls_model = txt_cls_model
        self.tokenizer = tokenizer
        self.ocr_model = ocr_model
        self.ocr_processor = ocr_processor
        self.rel_txt_classes = rel_txt_classes
        self.rel_meta_txt_classes = rel_meta_txt_classes
        self.rel_visual_classes = rel_visual_classes
        self.unpackable_classes = unpackable_classes
        self.LaTex2Text = LaTex2Text

    @torch.no_grad()
    def parse(self, pdf_files: list[str]) -> list[dict[str, Any]]:
        """Parse a PDF file and extract markdown.

        Parameters
        ----------
        pdf_path : str
            Path to the PDF file to convert.

        Returns:
        -------
        tuple[str, dict[str, str]] | None
            The extracted markdown and metadata or None if an error occurred.
        """
        from tensor_utils import assign_text_inferred_meta_classes
        from tensor_utils import custom_collate
        from tensor_utils import format_documents
        from tensor_utils import get_packed_patch_tensor
        from tensor_utils import PDFDataset
        from tensor_utils import pre_processing
        from tensor_utils import store_visuals
        from tensor_utils import update_main_content_dict
        from texify2.texify.inference import accelerated_batch_inference
        from torch.utils.data import DataLoader

        # load dataset
        dataset = PDFDataset(pdf_paths=pdf_files, meta_only=args.meta_only)

        # Create a DataLoader for batching and shuffling
        data_loader = DataLoader(
            dataset=dataset,
            batch_size=args.batch_yolo,
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
            i_tab, i_fig, prev_file_id = 0, 0, -1
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
                results,
                file_ids,
                self.rel_txt_classes.values(),
                iou_thres=0.001,
            )

            # metadata specific extraction
            (
                pack_patch_tensor,
                idx_quad,
                curr_file_ids,
            ) = get_packed_patch_tensor(
                tensors,
                y,
                self.rel_txt_classes.values(),
                unpackable_class_ids=self.unpackable_classes.values(),
                sep_symbol_flag=False,
                btm_pad=4,
                by=['file_id'],
                offset=args.bbox_offset,
                sep_symbol_tensor=None,
            )

            # store visual patches (tables, figures)
            if self.rel_visual_classes:
                vis_path_dict, i_tab, i_fig, prev_file_id = store_visuals(
                    tensors=tensors,
                    y=y,
                    rel_visual_classes=self.rel_visual_classes,
                    file_paths=file_paths,
                    file_ids=file_ids,
                    output_dir=args.output_dir,
                    i_tab=i_tab,
                    i_fig=i_fig,
                    prev_file_id=prev_file_id,
                )

            # no use for page images pass this point
            tensors = None

            # ViT: pseudo-OCR inference
            text_results = accelerated_batch_inference(
                pack_patch_tensor,
                self.ocr_model,
                self.ocr_processor,
                batch_size=args.batch_vit,
            )

            # re-assess meta text categories
            index_quadruplet = assign_text_inferred_meta_classes(
                self.txt_cls_model,
                tokenizer=self.tokenizer,
                batch_size=args.batch_cls,
                index_quadruplet=idx_quad,
                text_results=text_results,
            )

            # assign decoded text to file docs
            doc_dict = update_main_content_dict(
                doc_dict,
                text_results,
                index_quadruplet,
                curr_file_ids,
                vis_path_dict,
            )

            # Store the file path for each file_id in the batch
            doc_file_paths.update(dict(zip(file_ids, file_paths)))

        # Store the parsed documents
        documents = format_documents(
            doc_dict=doc_dict,
            doc_file_paths=doc_file_paths,
            LaTex2Text=self.LaTex2Text,
        )

        return documents


def main(args: OreoConfig) -> None:
    """Main function for the Oreo parser."""
    ...


if __name__ == '__main__':
    args = OreoConfig(**vars(parse_args()))

    check = (
        args.equation_flag
        or args.table_flag
        or args.fig_flag
        or args.secondary_meta
        or args.meta_only
    )
    if args.detect_only and check:
        raise ValueError(
            'The `--detect_only` flag cannot be used with any other flag.'
        )
    if args.meta_only and check:
        raise ValueError(
            'The `--meta_only` flag cannot be used with any other flag.'
        )

    if args.input_dir.is_dir():
        raise FileNotFoundError(f'Input does not exist: {args.input_dir}')
    if args.output_dir.is_dir():
        raise FileNotFoundError(f'Output does not exist: {args.output_dir}')
    if args.detection_weights_path.is_file():
        raise FileNotFoundError(
            'Path to weights of detection model does not exist: '
            f'{args.detection_weights_path}'
        )
    if args.text_cls_weights_path.is_dir():
        raise FileNotFoundError(
            'Path to weights of text classification model does not exist: '
            f'{args.text_cls_weights_path}'
        )

    main(args)
