from __future__ import annotations

import gc
import json
import os
import re
import sys
from collections import Counter
from collections import defaultdict
from enum import auto
from enum import Enum
from pathlib import Path
from typing import Union

import fitz
import numpy as np
import torch
import torch.nn.functional as F
import yaml
from PIL import Image
from pylatexenc.latex2text import LatexNodes2Text
from torch.utils.data import Dataset
from torchvision import transforms

# import YOLOv5 dependencies
yolo_path = './yolov5'
if yolo_path not in sys.path:
    sys.path.append(yolo_path)
from utils.general import non_max_suppression


class FileType(Enum):
    PDF = auto()
    DOC = auto()
    DOCX = auto()


def infer_file_type(doc_dir: Union[Path, str]) -> str:
    """Scans the files in the input directory and returns the file format of the most frequently observed file.

    Args:
        doc_dir (Path): Directory path containing the files.

    Returns:
        str: The file type or suffix (without `.`) of the most frequently observed file format. Subsequently, only those files will be processed.

    Raises:
        AssertionError: If `doc_dir` is not a valid directory or contains no files.
        ValueError: If the most frequent file type does not match the list of allowed file types, even if allowed file types are present in the directory.
    """
    # to Path
    doc_dir = Path(doc_dir)

    # check dir existance
    assert os.path.isdir(
        doc_dir
    ), f'Directory `doc_dir`={doc_dir} does not exist.'
    assert (
        len(os.listdir(doc_dir)) > 0
    ), 'Directory `doc_dir` exists but is empty.'

    # most frequent file format
    suffix_counts = Counter(
        [Path(f).suffix for f in os.listdir(doc_dir) if '.' in f]
    )
    mode_suffix, mdoe_count = suffix_counts.most_common(1)[0]
    mode_suffix = mode_suffix.split('.')[1].upper()

    # match
    if mode_suffix in FileType.__members__:
        file_type = FileType[mode_suffix].name.lower()
    else:
        raise ValueError(
            f"The mode file type in the directory '{mode_suffix}' does not match any of the allowed file types: '.pdf', '.doc', '.docx'"
        )

    return file_type


class DocDataset(Dataset):
    def __init__(
        self,
        doc_dir: Union[Path, str],
        meta_only: bool = False,
        target_heigth: int = 1280,
        file_type: str = 'pdf',
    ) -> None:
        assert (
            file_type in ['pdf', 'doc', 'docx', 'auto']
        ), 'DocDataset can handle PDFs (file_type=`pdf`) or Word documents (file_type=`doc(x)`)'
        if file_type == 'auto':
            file_type = infer_file_type(doc_dir)

        if file_type != 'pdf':
            raise NotImplementedError(
                'Only file_type that is supported currently is `pdf`'
            )

        self.doc_file_paths = sorted(
            [
                Path(doc_dir) / f
                for f in os.listdir(doc_dir)
                if f.lower().endswith(f'.{file_type}')
            ]
        )
        self.target_heigth = target_heigth
        self.doc_file_ids = [i for i in range(len(self.doc_file_paths))]
        self.meta_only = meta_only
        self.file_type = file_type

        # image count
        doc_lengths = []
        for doc_path in self.doc_file_paths:
            if file_type == 'pdf':
                doc = fitz.open(doc_path)
                doc_lengths.append(len(doc))
                doc.close()
            else:
                pass

        # cummulative page count across documents
        self.doc_csum = np.cumsum(doc_lengths).astype(int)

        # total page count
        if self.meta_only:
            self.len = len(self.doc_file_paths)
        else:
            self.len = sum(doc_lengths)

        self.current_doc_file_path = None
        self.current_doc_doc = None
        self.current_doc_idx = 0

    def __len__(self) -> int:
        """Conventional dataset: total number of pages in across all documents in the dataset.
        If `meta_only`, length equals the number of documents (as only the first page each document is used).
        """
        return self.len

    def __getitem__(self, idx: int) -> torch.Tensor:
        """A single page (i.e. image) rather than the document is the dataset item.
        Since page counts vary across documents and batch sizes shall be exhausted, the cummulative sum of pages (up to a document) is pre-computed.
        For a given page index, determine the corresponding doc (index) it belongs to.
        Subsequently, compute the relative page index by subtracting the previous page count.

        Leverage `shuffle=False` as this Dataset is used for inference only.
        """
        # identify the document page index `idx` falls into
        # - meta_data only
        if self.meta_only:
            doc_index = idx
            rel_page_idx = 0
        # - entire paper
        else:
            doc_index = np.searchsorted(self.doc_csum, idx, side='right')
            # relative page (shift idx if previous papers)
            rel_page_idx = idx - (
                0 if doc_index == 0 else int(self.doc_csum[doc_index - 1])
            )

        # keep a single document in memory at a time
        if doc_index != self.current_doc_idx or self.current_doc_doc is None:
            # close the previous DOC
            if self.current_doc_doc is not None:
                self.current_doc_doc.close()
            # Update the current doc and read it into memory
            self.current_doc_file_path = self.doc_file_paths[doc_index]
            self.current_doc_file_id = self.doc_file_ids[doc_index]
            self.current_doc_doc = fitz.open(self.current_doc_file_path)
            self.current_doc_idx = doc_index

        # requested page of current document
        page = self.current_doc_doc[rel_page_idx]

        # output tensor representing a page
        output = docpage_to_tensor(
            page, self.target_heigth, fill_value=1
        )  # (C, H, W)

        return (output, self.current_doc_file_id, self.current_doc_file_path)


class DatasetSizeError(Exception):
    """Exception raised when dataset size exceeds maximum limit to perform a certain operation."""

    def __init__(self, n: int):
        message = f'The dataset contains to many pages (N={n}) to store tables or figures. Increase `max_page_to_store_visuals` to a value >{n} and make sure the outputdir has enough storage; or set leave `--table` and `--figure` out.'
        self.message = message
        super().__init__(self.message)


def custom_collate(batch):
    # Transpose the batch (unzip)
    tensors, file_ids, file_paths = zip(*batch)

    # Stack the tensors into a single tensor
    stacked_tensors = torch.stack(tensors)

    # Return the stacked tensor and the list of file strings
    return stacked_tensors, file_ids, file_paths


def pre_processing(
    results: torch.Tensor,
    file_ids: list[int],
    rel_class_ids: list[int],
    iou_thres: float = 0.0001,
    conf_thres: float = 0.6,
    x_delta: int = 200,
    freq_thresh: float = 0.1,
    collapse_agnosticly: bool = True,
) -> torch.Tensor:
    """Emulates the PIL image pre-processing pipeline for texify's OCR model but with torch.tensor (3x improvement).
    Applies non_max_suppression() and get_y_indexed() as tensor-only operations.

    Args:
    - results             : BxNxD-dimensional tensor inferred by Yolov5 (B:b.size, N: # bboxes, D:dim. = #classes + 5 {4 coords+score})
    - file_ids            : List of IDs from the respective dataset
    - rel_class_ids       : List of class label IDs used to infer modes (usually 0: Text, potentially 16: List_element etc.)
    - iou_thres           : Maximum IoU (intersection over union) that is tolerable for bbox predictions
    - conf_thres          : Confidence score must exceed this threshold to be forwarded
    - x_delta             : Coarsity with which robust modes are inferred for items along the x-axis
    - freq_thresh         : Minimum frequency of observations associated to a mode for this mode to be used to identify column indices of items
    - collapse_agnosticly : Handle overlapping bboxes (True: collapse to single bbox of most likely class; False: maintain all bboxes per class)

    Raises:
    -
    """
    # extract bbox predictions
    y = non_max_suppression(
        results,
        iou_thres=iou_thres,
        conf_thres=conf_thres,
        agnostic=collapse_agnosticly,
    )

    # post-process bbox predictions
    y = get_y_indexed(
        y,
        file_ids=file_ids,
        x_delta=x_delta,
        freq_thresh=freq_thresh,
        rel_class_ids=rel_class_ids,
    )

    return y


def get_y_indexed(
    y: torch.Tensor,
    file_ids: list[int],
    rel_class_ids: list[int],
    x_delta: int = 200,
    freq_thresh: float = 0.15,
    y_freq_thresh: float = 0.0,
    y_bin_width: int = 10,
    y_delta: int = 25,
) -> torch.Tensor:
    """Derive row and column indices of each patch. Row indices are derived by y_mid (the lower Y_mid, the higher the patch is located on the page)
    and column index by robust mode association.

    Args:
    - y             : 2D tensor where each row is a patch/bbox and the columns are (x_min, y_min, x_max, y_max, conf, cls label, page idx)

    - rel_class_ids : List of relevant class label IDs used for robust mode estimation
    - x_delta       : coarsity by which X_midpoints are rounded
    - freq_thresh   : Frequency threshold at which a column is defined

    - y_freq_thresh : Relative frequency required to be considered for mode estimation. Prevents "outlier" boxes to distort otherwise standard text flow.
    - y_bin_width   : Binning size.

    Returns:
    - torch.Tensor
    """
    # - columns indices
    xmin_column = 0
    ymin_column = 1
    xmax_column = 2
    ymax_column = 3
    score_column = 4
    cls_column = 5
    page_idx_column = 6
    midpoint_x_column = 7
    midpoint_y_column = 8
    col_idx_column = 9
    row_idx_column = 10
    width_column = 11
    height_column = 12
    order_idx_column = 13
    file_idx_column = 14
    idx_column = 15

    # - add page index & stack
    y_indexed = [
        torch.cat(
            (
                page_tensor,
                torch.full(
                    (page_tensor.size(0), 1), i, device=page_tensor.device
                ),
            ),
            dim=1,
        )
        for i, page_tensor in enumerate(y)
    ]

    y_batch = torch.cat(y_indexed, dim=0)

    # subset to localizing condition column classes
    current_device = y_batch.device
    loc_cond_col = torch.zeros(y_batch.size(0), dtype=torch.bool).to(
        current_device
    )

    # define the condition (class-conditioned, x-axis modes)
    for cls_label in rel_class_ids:
        loc_cond_col = loc_cond_col | torch.isclose(
            y_batch[:, cls_column], torch.tensor(1.0 * cls_label)
        )

    # - compute midpoints (along x-axis)
    x_Mid = 0.5 * (y_batch[:, xmin_column] + y_batch[:, xmax_column])
    rounding_values = torch.arange(100, 1300, x_delta)
    x_Mid_rounded = torch.round(x_Mid / x_delta) * x_delta
    x_Mid_rounded = torch.clamp(x_Mid_rounded, min=x_delta, max=1200).reshape(
        x_Mid_rounded.size()[0], 1
    )

    # - compute y minpoints (along y-axis)
    y_Min = 0.5 * (y_batch[:, ymin_column] + y_batch[:, ymax_column])  # NEW
    rounding_values = torch.arange(100, 1300, y_delta)
    y_Min_rounded = torch.round(y_Min / y_delta) * y_delta
    y_Mid_rounded = torch.clamp(y_Min_rounded, min=y_delta, max=1300).reshape(
        y_Min_rounded.size()[0], 1
    )

    # - augment w/ midpoints
    tensor_aug = torch.cat(
        (y_batch, x_Mid_rounded.reshape(x_Mid_rounded.size()[0], 1)), dim=1
    )

    # - compute midpoints (along y-axis)
    y_Mid = 0.5 * (y_batch[:, ymin_column] + y_batch[:, ymax_column])
    tensor_aug = torch.cat(
        (tensor_aug, y_Mid.reshape(x_Mid.size()[0], 1)), dim=1
    )

    # empty mode column: col_index, row_idx, width, height, patch_order_idx, file_idx, element idx
    zeros_columns = torch.zeros(
        tensor_aug.size(0), 7, device=tensor_aug.device
    )
    tensor_aug = torch.cat((tensor_aug, zeros_columns), dim=1)

    # for each page (if localizing class): mode estimation/assignment
    for page_idx in torch.unique(
        y_batch[loc_cond_col, page_idx_column], return_counts=False
    ):
        # condition column (txt element & page idx)
        idx_cond_col = torch.isclose(y_batch[:, page_idx_column], page_idx)

        # x-axis mode estimation
        modes, counts = torch.unique(
            x_Mid_rounded[idx_cond_col & loc_cond_col], return_counts=True
        )
        freqs = (
            counts.float() / len(x_Mid_rounded[idx_cond_col])
            if len(x_Mid_rounded[idx_cond_col]) > 0
            else 1
        )
        robust_modes = modes[freqs > freq_thresh]
        robust_modes = (
            robust_modes if robust_modes.size()[0] > 0 else torch.tensor(0)
        )

        # column_index
        # = = = = = = =
        # - compute mode distances
        midpoint_mode_distances = torch.abs(
            tensor_aug[:, midpoint_x_column].unsqueeze(1) - robust_modes
        )
        # - closest mode index
        min_indices = torch.argmin(midpoint_mode_distances, dim=1)
        # - column_index
        tensor_aug[idx_cond_col, col_idx_column] = min_indices[
            idx_cond_col
        ].float()

        # row_index
        # = = = = =
        # - round y_min values
        y_bins = torch.arange(0, 1280, y_bin_width).to(tensor_aug.device)
        y_min_rounded = torch.bucketize(
            tensor_aug[idx_cond_col, ymin_column], y_bins
        ).to(torch.float)

        # y-axis mode estimation
        modes, counts = torch.unique(
            y_Mid_rounded[idx_cond_col], return_counts=True
        )
        freqs = (
            counts.float() / len(x_Mid_rounded[idx_cond_col])
            if len(y_Mid_rounded[idx_cond_col]) > 0
            else 1
        )
        y_robust_modes = modes[freqs > y_freq_thresh]
        y_robust_modes = (
            y_robust_modes if y_robust_modes.size()[0] > 0 else torch.tensor(0)
        )

        # column_index
        # - compute mode distances
        y_min_mode_distances = torch.abs(
            tensor_aug[:, ymin_column].unsqueeze(1) - y_robust_modes
        )
        # - closest mode index
        y_min_indices = torch.argmin(y_min_mode_distances, dim=1)
        # - column_index
        tensor_aug[idx_cond_col, row_idx_column] = y_min_indices[
            idx_cond_col
        ].float()

        # lexsort : infer order of patches
        sorted_indices = torch_lexsort(
            tensor_aug[idx_cond_col, :],
            keys=[ymin_column, row_idx_column, col_idx_column],
        )
        ranks = torch.empty_like(sorted_indices, dtype=torch.float)
        ranks[sorted_indices] = 1.0 * torch.arange(len(sorted_indices)).to(
            tensor_aug.device
        )
        tensor_aug[idx_cond_col, order_idx_column] = 1.0 * ranks

        # DEBUG: looks good
        # print('Ranks : ', ranks)

    # assign `width` and `height` columns
    tensor_aug[:, width_column] = tensor_aug[:, 2] - tensor_aug[:, 0]
    tensor_aug[:, height_column] = tensor_aug[:, 3] - tensor_aug[:, 1]

    # map file_idx (via page_idx and file_ids list)
    tensor_aug[:, file_idx_column] = torch.Tensor(file_ids).to(
        tensor_aug.device
    )[tensor_aug.to(torch.int)[:, page_idx_column]]

    # assign elementwise idx (last column)
    tensor_aug[:, idx_column] = (
        torch.arange(tensor_aug.size()[0])
        .to(torch.float)
        .to(tensor_aug.device)
    )

    return tensor_aug


def subset_y_by_class(
    y_batch: torch.Tensor, rel_class_ids: list[int]
) -> torch.Tensor:
    """Given a tensor of bbox predictions w/ columns to store metadata on patch location and inferred label. Subsets those "rows" that coincide to `text` predictions.

    Assumes 5th column of `y_batch` to hold class label information as per Yolov5 standard.

    Args:
    - y_batch        : 2D tensor of stacked results (n,7) with page index added, 7 columns: x_min, y_min, x_max, y_max, conf, cls_label, page_idx.
    - rel_class_ids  : list of integers representing the relevant predicted classes per bounding box (0: Text, 1: Title etc.) which are to be selected

    Returns:
    - subset_tensor  : subset of torch tensor (n_sub, 7) where each bbox belongs to one of the relevant classes

    Raises:
    - AssertionError : Number of relevant classes is not in {0,...,21}, i.e. number of SPv05 classes.
    """
    assert (
        min(rel_class_ids) >= 0 and max(rel_class_ids) <= 21
    ), 'Class labels should be '

    # constants
    cls_column = 5

    # copy data & device
    data_tensor = torch.Tensor(y_batch)
    current_device = data_tensor.device

    # column w/ class label
    cls_column = data_tensor[:, cls_column]

    # condtion
    condition_tensor = torch.zeros(data_tensor.size(0), dtype=torch.bool).to(
        current_device
    )

    # Define the condition
    for cls_label in rel_class_ids:
        condition_tensor = condition_tensor | torch.isclose(
            cls_column, torch.tensor(1.0 * cls_label)
        )

    # Subset the tensor based on the condition
    subset_tensor = data_tensor[condition_tensor, :]

    return subset_tensor


def docpage_to_tensor(
    page: fitz.Page,
    target_height: int = 1280,
    target_width: int = 960,
    fill_value: int = 0,
) -> torch.Tensor:
    """Converts a document-style, fitz instance `Page` into a 3-dimensional torch tensor (C x H x W)
    Maintains height and width to be a multiple of 64 to not cause misalignment with Yolov5 inference.

    Args:
    - page           : Page to be converted to a tensor (representing an image of the document page)
    - target_height  : Height

    Returns:
    - x              : output tensor of dimension C x des. Height x des. Width (representing the image)

    Raises:
    - AssertionError : Target height of the page images should lie in [640,1280], i.e. from small to large image size.
    """
    assert (
        640 <= target_height <= 1280
    ), 'Target height should be in [640, 1280]'

    # re-scale (maintain multiple of 64 along width and height)
    scale_factor_y = target_height / page.mediabox_size.y
    scale_factor_x = (
        (scale_factor_y * page.mediabox_size.x // 64) * 64
    ) / page.mediabox_size.x
    mat = fitz.Matrix(scale_factor_x, scale_factor_y)

    # pix
    pix = page.get_pixmap(matrix=mat)
    h, w = pix.height, pix.width

    # data
    pixmap_data = np.frombuffer(pix.samples, dtype=np.uint8)
    pixmap_data = pixmap_data.reshape((h, w, len(pixmap_data) // (w * h)))

    # tensor
    x = transforms.ToTensor()(pixmap_data)

    # size
    if x.size() != torch.Size([3, target_height, target_width]):
        x = pad_zeros(
            x,
            target_height=target_height,
            target_width=target_width,
            fill_value=fill_value,
        )
        x = center_crop(
            x, target_height=target_height, target_width=target_width
        )

    return x


class ResizeAndCenterCropTensor:
    def __init__(self, reference_height=1280, reference_width=960):
        self.reference_height = reference_height
        self.reference_width = reference_width

    def __call__(self, img_tensor):
        _, height, width = img_tensor.shape

        # Resize
        if height > self.reference_height or width > self.reference_width:
            img_tensor = torch.nn.functional.interpolate(
                img_tensor.unsqueeze(0),
                size=(self.reference_height, self.reference_width),
                mode='bilinear',
                align_corners=False,
            ).squeeze(0)

        # Center crop
        _, height, width = img_tensor.shape
        start_h = (height - self.reference_height) // 2
        start_w = (width - self.reference_width) // 2

        img_tensor = img_tensor[
            :,
            start_h : start_h + self.reference_height,
            start_w : start_w + self.reference_width,
        ]

        return img_tensor


def pad_zeros(
    x: torch.Tensor,
    target_height: int = 1280,
    target_width: int = 960,
    fill_value: int = 0,
) -> torch.Tensor:
    """Pads a given torch tensor with 0s such that it meets (C x H x W) condition.
    Leaves channel (1st entry untouched)

    Args:
    - x             : 3-dimensional input tensor of dimension with (channel x height x width) that is to be padded
    - target_heigth : Desired height of the tensor (i.e. x.size(1))
    - target_width  : Desired width of the tensor (i.e. x.size(2))
    - fill_value    : Fill-value

    Returns:
    - output tensor : Zero-padded in the bottom rows and rightmost columns

    Raises:
    """
    # height
    if target_height > x.size(1):
        zero_cols = fill_value * torch.ones(
            (x.size(0), target_height - x.size(1), x.size(2)), device=x.device
        )
        x = torch.cat((x, zero_cols), dim=1)
    # width
    if target_width > x.size(2):
        zero_rows = fill_value * torch.ones(
            (x.size(0), x.size(1), target_width - x.size(2)), device=x.device
        )
        x = torch.cat((x, zero_rows), dim=2)

    return x


def center_crop(
    x: torch.Tensor, target_height: int = 1280, target_width: int = 960
) -> torch.Tensor:
    """If the tensor is larger along width or height, center-crops the tensor to match desired heigth/width.

    Args:
    - x             : 3-dimensional input tensor of dimension with (channel x height x width) that is to be padded
    - target_heigth : Desired height of the tensor (i.e. x.size(1))
    - target_width  : Desired width of the tensor (i.e. x.size(2))

    Returns:
    - output        : Zero-padded in the bottom rows and rightmost columns
    """
    h_delta, w_delta = 0, 0

    # delta
    if x.size(1) > target_height:
        h_delta = (x.size(1) - target_height) // 2
    if x.size(2) > target_width:
        w_delta = (x.size(2) - target_width) // 2

    # subset
    x = x[
        :,
        h_delta : (target_height + h_delta),
        w_delta : (target_width + w_delta),
    ]

    return x


def pad_patch(patch: torch.Tensor, patch_value: int = 255) -> torch.Tensor:
    """Pads "0"s to a non-square patch

    Args:
    - patch_value  : Value that is to be inserted for padding (255: white on RGB scale)

    Returns:
    - padded_patch : Padded tensor post upscaling
    """
    target_length = max(patch.size()[1:])

    # padding
    pad_height = max(0, target_length - patch.size(1))
    pad_width = max(0, target_length - patch.size(2))

    pad_top = max(pad_height // 2, 0)
    pad_bottom = max(pad_height - pad_top, 0)
    pad_left = max(pad_width // 2, 0)
    pad_right = max(pad_width - pad_left, 0)

    padded_patch = F.pad(
        patch,
        (pad_left, pad_right, pad_top, pad_bottom),
        mode='constant',
        value=patch_value,
    )

    return padded_patch


def resize_patch(
    patch: torch.Tensor, max_length: int = 420, mode: str = 'bilinear'
) -> torch.Tensor:
    """Scale patch to (3x420x420). the texify default input size

    Args:
    - patch      : 3D tensor representing an image patch
    - max_length : Maximum length a patch is allowed to be scaled to
    - mode       : Mode by which upscaled pixels are interpolated (e.g. `nearest`, `bilinear` is Marker's PIL equivalent)
    """
    # patch = patch.to(torch.float)
    height, width = patch.size()[1:]

    # rescale
    width_scale = 1.0 * max_length / width
    height_scale = 1.0 * max_length / height
    scale = min(width_scale, height_scale)

    # maybe here
    new_width = max_length if width_scale == scale else round(width * scale)
    new_height = max_length if height_scale == scale else round(height * scale)

    # PIL-like resizing/resampling
    t_interpol = F.interpolate(
        patch.unsqueeze(0),
        size=(new_height, new_width),
        mode=mode,
        align_corners=False,
        antialias=True,
    )[0, :, :]  # .to(torch.uint8)

    return t_interpol.round().to(torch.int)


def normalize_pad(
    t: torch.Tensor,
    mean: list[float] = [0.485, 0.456, 0.406],
    std: list[float] = [0.229, 0.224, 0.225],
):
    """Normalizes z_i = (x_i - mu_i) / sigma_i along each channel i in {0,1,2}

    Args:
    - t            : Tensor of size BxCxHxW with unnormalized (i.e. native) float pixel values scaled to [0.0, 1.0]
    - mean         : vector of length 3 (one value per channel) by which every pixel is shifted
    - std          : vector of length 3 (one value per channel) by by which every shifted pixel is divided

    Returns:
    - normalized_t : Tensor of size BxCxHxW for which most pixel fall within [-2.575, +2.575]

    Raises:
    - ValueError   : Mismatch of dimension between arguments. Input tensor `t` does not adhere to Bx3xHxW or mean/std are not of length 3.
    """
    normalize_transform = transforms.Normalize(mean=mean, std=std)

    return normalize_transform(t)


def tens_proc(
    t: torch.Tensor,
    max_length: int = 420,
    mode: str = 'bilinear',
    patch_value: float = 1.0,
):
    """Processing stages of torch.Tensor that emulates the PIL.Image transformation pipeline of VariableDonutImageProcessor's `process_inner()`

    Note:
    The pipeline matches around 95% of the pixels exactly with a remainder of 2.5% of pixels varying by either -1 or +1 pixel in {0,1,,...,255} scale - likely due to
    rounding.

    Args:
    - t           : Input torch tensor BxCxHxW
    - max_length  : Maximum length a patch is allowed to be scaled to
    - mode        : Mode by which upscaled pixels are interpolated (e.g. `nearest`, `bilinear` is Marker's PIL equivalent)
    - patch_value : The padding value that is to be inserted to fill out space (1.0 or 255 is white)

    Raises:
    -
    """
    # dtype float & resize
    t = t.to(torch.float)
    t = resize_patch(t, max_length=max_length, mode=mode)

    # scale pixel values to [0.0, 1.0]
    t = t.to(torch.float) / 255.0

    # rescale (to [0.0, 1.0])
    t = pad_patch(t, patch_value)

    # normalize along channels
    t = normalize_pad(t)

    return t


def get_patch_batch(
    tensors: torch.Tensor,
    y: torch.Tensor,
    max_width: int = 420,
    max_height: int = 420,
    dtype=torch.float16,
) -> torch.Tensor:
    """Given a batch of images (BxCxHxW) and a tensor of meta information on patches, return the batch of (similarily sized) patches after pre-processing the
    images in a PIL-like fashion.

    Args:
    - tensors     : Tensor of page images, i.e. of size Bx3x1280x960
    - y           : 2D tensor that stores the patches' metadata (e.g. location, inferred class label and derived attributes relevant for subsequent OCR and patch order reconstruction)
    - max_width   : Width to which a patch is scaled if it's width exceeds this value
    - max_height  : Height -||- height exceeds this value
    - dtype       : Datatype to which tensor of patches is set (float16 is

    Returns:
    - patch_batch : Tensor of size N x 3 x H_m x W_m where `N` is the number of patches given by `N`, 3 channels are maintained and H_m=W_m=420.

    """
    # Number of patches can only be inferred from meta table (as one page can have 0 or many patches)
    N = y.size()[0]

    # convert to target dtype
    tensors = tensors.to(dtype=dtype)

    # placeholder batch
    patch_batch = torch.zeros(
        (N, 3, max_height, max_width),
        device=tensors.device,
        dtype=tensors.dtype,
    )

    # subset relevant meta data
    y_rel = y.round().to(int)

    # extract each patch from page
    for i in range(N):
        x_min, y_min, x_max, y_max = y_rel[i, :4]
        b = y_rel[i, 6]

        # extract patch
        patch = tensors[b, :, y_min:y_max, x_min:x_max]

        # process patch
        patch = tens_proc(
            t=patch * 255.0, max_length=420, mode='bilinear', patch_value=1.0
        )

        # assign
        patch_batch[i, :, :, :] = patch

    return patch_batch


def vectorized_patch_batch(
    tensors: torch.Tensor,
    y: torch.Tensor,
    max_width: int = 420,
    max_height: int = 420,
    mean: list[float] = [0.485, 0.456, 0.406],
    std: list[float] = [0.229, 0.224, 0.225],
) -> torch.Tensor:
    """HELPER Function that was merely setup to assess the quality of the PIL-like processing pipeline.

    Given a batch of images (BxCxHxW) and a tensor of meta information on patches, return the batch of patch tensors
    """
    # No of patches
    y = y[:, [0, 1, 2, 3, 6]].round().to(int)
    N = y.size()[0]

    # transform
    tensors = tensors.to(torch.float) * 255.0

    # placeholder batch
    patch_batch = torch.zeros((N, 3, max_height, max_width))

    for i in range(N):
        x_min, y_min, x_max, y_max, b = y[i]
        # extract patch
        patch = tensors[b, :, y_min:y_max, x_min:x_max]  # challenge

        # resize
        t = resize_patch(patch, max_length=420, mode='bilinear')  # challenge

        # assign
        patch_batch[i, :, :, :] = pad_patch(t, 255.0)

    # rescale & normalize
    patch_batch = patch_batch / 255.0
    patch_batch = normalize_pad(patch_batch)

    return patch_batch


def get_patch_list(
    tensors: torch.Tensor,
    y: torch.Tensor,
    max_width: int = 420,
    max_height: int = 420,
    mean: list[float] = [0.485, 0.456, 0.406],
    std: list[float] = [0.229, 0.224, 0.225],
    offset: int = 0,
) -> tuple[torch.Tensor, list[int]]:
    """Given a batch of images (BxCxHxW) and a tensor of meta information on patches, return the batch of patch tensors.

    Args:
    - tensors     : Tensor of page images, i.e. of size Bx3x1280x960
    - y           : 2D tensor that stores the patches' metadata (e.g. location, inferred class label and derived attributes relevant for subsequent OCR and patch order reconstruction)
    - max_width   : Width to which a patch is scaled if it's width exceeds this value
    - max_height  : Height -||- height exceeds this value
    - dtype       : Datatype to which tensor of patches is set (float16 is
    - mean        : vector of length 3 (one value per channel) by which every pixel is shifted
    - std         : vector of length 3 (one value per channel) by by which every shifted pixel is divided
    - offset      : Number of pixels by which the bounding box is extended in both directions along x- and y-axis less tight fit of text and better OCR performance

    Returns:
    - patch_list  : List of variable sized patches as extracted from the tensor images

    """
    # contsant
    order_idx_columns = 13

    # No of patches
    y_sub = y[:, [0, 1, 2, 3, 6]].round().to(int)
    N = y.size()[0]

    # transform
    tensors = tensors.to(torch.float) * 255.0

    # placeholder batch
    patch_list = []
    for i in range(N):
        x_min, y_min, x_max, y_max, b = y_sub[i]
        _, _, page_h, page_w = tensors.size()
        # apply offset (artificially increase bbox)
        if offset > 0:
            x_min, y_min = max(0, x_min - offset), max(0, y_min - offset)
            x_max, y_max = (
                min(page_w, x_max + offset),
                min(page_h, y_max + offset),
            )
        # extract patch
        patch = tensors[b, :, y_min:y_max, x_min:x_max]  # challenge

        # append list
        patch_list.append(patch / 255.0)

    return patch_list


def patch_list_to_tensor(
    grouped_patch_list: list[torch.Tensor],
    dtype: torch.dtype,
    max_width: int = 420,
    max_height: int = 420,
    mean: list[float] = [0.485, 0.456, 0.406],
    std: list[float] = [0.229, 0.224, 0.225],
    offset: int = 0,
) -> tuple[torch.Tensor, list[int]]:
    """Given an list of variably-sized, raw patches, turns them into a single torch tensor of dimension (N_eff,C,H,W)

    Args:
    - grouped_patch_list : List of sublists where each sublist refers to patches that can be packed into one joint patch
    - dtype              : Target datatype to which the output tensor is cast
    - max_width          : Width to which a patch is scaled if it's width exceeds this value
    - max_height         : Height -||- height exceeds this value
    - mean               : vector of length 3 (one value per channel) by which every pixel is shifted
    - std                : vector of length 3 (one value per channel) by by which every shifted pixel is divided
    - offset             : Number of pixels by which the bounding box is extended in both directions along x- and y-axis less tight fit of text and better OCR performance

    Returns:
    - packed_batch       : Tensor of patches of size N_eff x 3 x 420 x 420 where each patch potentially contains several re-scaled patches

    """
    # No of patches
    N_eff = len(grouped_patch_list)

    # transform
    packed_batch = (
        torch.zeros((N_eff, 3, max_height, max_width)).to(
            grouped_patch_list[0].device
        )
        * 255.0
    )

    # placeholder batch
    patch_list = []
    for i, patch in enumerate(grouped_patch_list):
        # scale up
        patch *= 255.0

        # resize
        t = resize_patch(
            patch, max_length=max_width, mode='bilinear'
        )  # challenge

        # assign
        packed_batch[i, :, :, :] = pad_patch(t, 255.0)

    # rescale & normalize
    packed_batch = packed_batch / 255.0
    packed_batch = normalize_pad(packed_batch)

    # dtype
    packed_batch = packed_batch.to(dtype)

    return packed_batch


def lower_bound_patch_batch(
    tensors: torch.Tensor,
    y: torch.Tensor,
    max_width: int = 420,
    max_height: int = 420,
    mean: list[float] = [0.485, 0.456, 0.406],
    std: list[float] = [0.229, 0.224, 0.225],
) -> torch.Tensor:
    """HELPER FUNCTION ONLY to determine runtim of loops over torch.Tensors and normalization. Provides a lower bound for runtime of other functions.

    Given a batch of images (BxCxHxW) and a tensor of meta information on patches, return the batch of patch tensors
    """
    # No of patches
    y = y[:, [0, 1, 2, 3, 6]].round().to(int)
    tensors = tensors.to(torch.float) * 255.0
    N = tensors.size()[0]

    # placeholder batch
    patch_batch = torch.zeros((N, 3, max_height, max_width))

    for i in range(N):
        j = i + 1

    # rescale & normalize
    patch_batch = patch_batch / 255.0
    patch_batch = normalize_pad(patch_batch)

    return patch_batch


def merge_patches_into_row(
    row_patch_list: list[torch.Tensor],
    btm_pad: int,
    sep_flag: bool = False,
    alpha: float = 0.5,
) -> torch.Tensor:
    """Merges a list of patches tensors into a tensor assumed to be a row. Output tensor is of size C x H_max x W_row
    where H_max is the maximum patch height (inferred from the list) and W_row is the sum of all patch widths

    Args:
    - row_patch_list   : List of patches that is to be arranged in the same row (i.e. next to one another)
    - btm_pad          : Number of white pixels added to the bottom of the row to increase distance to adjacent rows. If chosen to small, causes hallucinations in OCR if chosen to large causes subsequent rows to be ignored at decoding stage.
    - sep_flag         : Flag indicating if a separating line is to be added between rows. (Empirical performance underwhelming when included.)
    - alpha            : How pronounced the grey bar is (if included)

    Returns:
    - row_tensor       : Tensor representing a row of patches.

    """
    row_height = max([patch.size()[1] for patch in row_patch_list])
    # row_tensor = torch.cat([F.pad(patch, (0, 0, 0, row_height-patch.size()[1]), value=1.0) for patch in row_patch_list], dim=2) # bottom-padding
    row_tensor = torch.cat(
        [
            F.pad(patch, (0, 0, row_height - patch.size()[1], 0), value=1.0)
            for patch in row_patch_list
        ],
        dim=2,
    )  # top-padding

    # padding at the bottom
    if btm_pad > 0:
        c, _, w = row_tensor.size()
        # add padding at top and btm
        pad_lines = torch.ones((c, btm_pad // 2, w)).to(row_tensor.device)
        row_tensor = torch.cat((pad_lines, row_tensor, pad_lines), dim=1)
        # add black line at the bottom
        if sep_flag:
            black_line = alpha * torch.ones((c, 1, w))
            row_tensor = torch.cat((row_tensor, black_line), dim=1)

    return row_tensor


def merge_rows_into_patch(
    list_of_row_tensors: list[torch.Tensor],
) -> torch.Tensor:
    """Merges a list of patches tensors (each representing a "row" in a patch.
    Pads rows with 1.0 (white pixels) before concatenating row-by-row

    Args:
    - row_patch_list   : List of patches that is to be arranged in the same row (i.e. next to one another)
    - btm_pad          : Number of white pixels added to the bottom of the row to increase distance to adjacent rows. If chosen to small, causes hallucinations in OCR if chosen to large causes subsequent rows to be ignored at decoding stage.
    - sep_flag         : Flag indicating if a separating line is to be added between rows. (Empirical performance underwhelming when included.)
    - alpha            : How pronounced the grey bar is (if included)

    Returns:
    - row_tensor       : Tensor representing a row of patches.
    """
    patch_width = max(
        [row_tensor.size()[2] for row_tensor in list_of_row_tensors]
    )

    patch_tensor = torch.cat(
        [
            F.pad(
                row_tensor, (0, patch_width - row_tensor.size()[2]), value=1.0
            )
            for row_tensor in list_of_row_tensors
        ],
        dim=1,
    )

    return patch_tensor


def grouped_patch_list(
    patch_list: list,
    y: torch.Tensor,
    unpackable_class_ids: list[int],
    by: list[str] = ['file_id', 'cls'],
) -> list[list[torch.Tensor]]:
    """Given a list of variably-sized patches, groups them by paper and cls type. Each group represented by a sublist in the returned list.
    def get_packed_patch_list
    Args:
    - patch_list           : List of (variably-sized) patches
    - y                    : 2D torch tensor storing the meta data
    - unpackable_class_ids : List of class IDs for which respective patches are not grouped but rather decoded individually
    - by                   : Str indicating by which criteria grouping is applied (class_label, file_id etc.) Corresponds to `inner merge`

    Returns:
    - grouped_lists      : List of sublists

    Raises:
    - AssertionError     :
    """
    # check input
    assert (
        len(y.size()) == 2 and y.size()[1] == 16
    ), 'Tensor of patch meta data must be 2-dimensional with 16 columns.'
    assert len(patch_list) == len(
        y
    ), 'List of raw patches and length of metadata tensor must coincide.'
    assert set(
        by
    ).issubset(
        {'file_id', 'cls', 'page_id'}
    ), 'Can only group by `file_id` (within pdf file), `cls` (same class label id), or `page_id` (within same pdf page).'

    # constants
    cls_column = 5
    page_idx_column = 6
    file_idx_column = 14

    # set of potential keys
    keys_dict = {
        'cls': cls_column,
        'page_id': page_idx_column,
        'file_id': file_idx_column,
    }
    keys = [keys_dict[k] for k in by]

    # processing
    index_set = torch.arange(len(y), device=y.device, dtype=torch.int)
    unpack_mask = torch.any(
        y[:, cls_column][:, None]
        == torch.Tensor(list(unpackable_class_ids)).to(y.device),
        dim=1,
    )
    unpack_indices = torch.arange(len(y))[unpack_mask.detach().cpu()].tolist()

    # add isolated identifier column (to keep unpackable patches isolated)
    zeros_col = torch.zeros(y.size()[0], 1, dtype=y.dtype, device=y.device)
    y_aug = torch.cat((y, zeros_col), dim=1).to(torch.int)
    unpack_col = y_aug.size()[1] - 1
    y_aug[unpack_indices, unpack_col] = (
        torch.arange(
            len(unpack_indices), device=y_aug.device, dtype=y_aug.dtype
        )
        + 1
    )
    keys_aug = keys + [unpack_col]

    # cumsum -> groups different streaks of `0`s together (when interrupted by non-zero, i.e. non-packable entry) add those entries with *100
    y_aug[:, unpack_col] = torch.cumsum(y_aug[:, unpack_col], dim=0) + (
        1000 * y_aug[:, unpack_col]
    )

    # unique entries
    _, index_groups = torch.unique(
        y_aug[:, keys_aug], return_inverse=True, dim=0
    )
    unique_vals = torch.unique(index_groups).detach().tolist()

    # subset index list
    subset_indices = [
        index_set[(index_groups == v)].tolist() for v in unique_vals
    ]
    subset_indices = sorted(subset_indices, key=lambda x: x[0])

    # subset list
    patch_sublists = [
        [patch_list[i] for i in sub_ind] for sub_ind in subset_indices
    ]

    return patch_sublists, subset_indices


def grouped_patch_list_LEGACY(
    patch_list: list, y: torch.Tensor, by=['file_id', 'cls']
) -> list[list[torch.Tensor]]:
    """Given a list of variably-sized patches, groups them by paper and cls type. Each group represented by a sublist in the returned list.
    def get_packed_patch_list
    Args:
    - patch_list         : List of (variably-sized) patches
    - y                  : 2D torch tensor storing the meta data
    - unpackable_classes : List of class IDs for which respective patches are not grouped but rather decoded individually
    - by                 : Str indicating by which criteria grouping is applied (e.g. class_label, file_id, or page_id). Corresponds to `inner merge`

    Returns:
    - grouped_lists      : List of sublists

    Raises:
    - AssertionError     :
    """
    # check input
    assert (
        len(y.size()) == 2 and y.size()[1] == 16
    ), 'Tensor of patch meta data must be 2-dimensional with 16 columns.'
    assert len(patch_list) == len(
        y
    ), 'List of raw patches and length of metadata tensor must coincide.'
    assert set(
        by
    ).issubset(
        {'file_id', 'cls', 'page_id'}
    ), 'Can only group by `file_id` (within pdf file), `cls` (same class label id), or `page_id` (within same pdf page).'

    # constants
    cls_column = 5
    page_idx_column = 6
    file_idx_column = 14

    # set of potential keys
    keys_dict = {
        'cls': cls_column,
        'page_id': page_idx_column,
        'file_id': file_idx_column,
    }
    keys = [keys_dict[k] for k in by]

    # processing
    index_set = torch.arange(len(y)).to(y.device).to(torch.int)
    y_sub = y[:, keys].to(torch.int)
    _, index_groups = torch.unique(y_sub, return_inverse=True, dim=0)
    unique_vals = torch.unique(index_groups).detach().tolist()

    # subset list
    subset_indices = [
        index_set[index_groups == v].tolist() for v in unique_vals
    ]
    patch_sublists = [
        [patch_list[i] for i in sub_ind] for sub_ind in subset_indices
    ]

    return patch_sublists, subset_indices


def get_packed_patch_list(
    patch_list: list[torch.Tensor],
    sep_tensor: torch.Tensor,
    return_indices: bool = True,
    sep_flag: bool = False,
    sep_symbol_flag: bool = False,
    btm_pad: int = 6,
    max_width: int = 420,
    max_height: int = 420,
    alpha: float = 0.5,
) -> list[torch.Tensor]:
    """Merges the patches (CxHxW patches) as tightly as possible into a list of packed patches

    Args:
    - patch_list      : (Flat) list of variably-sized patch images each being a 3xH_{i}xW_{i} image
    - sep_tensor      : Separator image stores as a tensor to be inserted into the patch at the end or each row (if sep_flag=True)
    - return_indices  : Flag indicating if the indices of the respective patches are to be returned (for debgging)
    - sep_flag        : Flag indicating if a separation line is to be included in between lines
    - sep_symbol_flag : Flag indicating if a  `sep_tensor` is to be included to allow unpacking a packed patch
    - btm_pad         : Number of white pixel lines added after each row to spread out rows in each patch
    - max_width       : Width to which a patch is scaled if it's width exceeds this value
    - max_height      : Height -||- height exceeds this value
    - alpha           : How pronounced the grey bar is (if included)

    Returns:
    - out_patches     : Tensor of packed patches N_eff x B x 420 x 420 where N_eff is the number of packed patches from N (unpacked), variably-sized patches

    Raises:

    """
    # init lists
    out_patches = []
    out_indices = []
    list_of_rows = []
    list_of_rows_indices = []
    current_row = []
    current_row_indices = []

    # init constants
    max_row_height = 0
    current_row_width = 0
    current_patch_heigth = 0

    # separator tensor dimensions
    if sep_tensor is None:
        sep_symbol_flag = False
    else:
        _, h_sep, w_sep = sep_tensor.size()

    # loop patches (& patch indices)
    for j, patch in enumerate(patch_list):
        # patch dimension
        _, h, w = patch.size()

        # fit in row?
        if current_row_width + w < max_width:
            # append row
            current_row.append(patch)
            if return_indices:
                current_row_indices.append(j)
            # update row dimensions
            current_row_width += w
            max_row_height = max(h, max_row_height)
            # - separator added (within row)
            if sep_symbol_flag:
                # append row
                current_row.append(sep_tensor)
                # update row dimensions
                current_row_width += w_sep
                max_row_height = max(h_sep, max_row_height)
        else:
            # current row done either way
            if len(current_row) > 0:
                list_of_rows.append(
                    merge_patches_into_row(
                        current_row,
                        btm_pad=btm_pad,
                        sep_flag=sep_flag,
                        alpha=alpha,
                    )
                )
                list_of_rows_indices += current_row_indices

            # new patch?
            if current_patch_heigth + h > max_height:
                # store previous patch
                if len(list_of_rows) > 0:
                    out_patches.append(merge_rows_into_patch(list_of_rows))
                if return_indices:
                    out_indices.append(list_of_rows_indices)

                # init new patch/row
                list_of_rows = []
                if return_indices:
                    list_of_rows_indices = []
                current_patch_heigth = h
            else:
                # same patch, new row
                current_patch_heigth += h
                max_row_height = h
                current_row_width = w
            # new row w/ or w/o new patch
            current_row = [patch]
            if return_indices:
                current_row_indices = [j]
            # sperator symbol
            if sep_symbol_flag:
                current_row.append(sep_tensor)
                current_row_width += w_sep
                max_row_height = max(h_sep, max_row_height)

    # after loop: clean-up excesss patches - append last row
    if current_row:
        # list existant?
        if list_of_rows:
            list_of_rows.append(
                merge_patches_into_row(
                    current_row,
                    btm_pad=btm_pad,
                    sep_flag=sep_flag,
                    alpha=alpha,
                )
            )
            # log patch index
            if return_indices:
                list_of_rows_indices += current_row_indices
        else:
            list_of_rows = [
                merge_patches_into_row(
                    current_row,
                    btm_pad=btm_pad,
                    sep_flag=sep_flag,
                    alpha=alpha,
                )
            ]
            list_of_rows_indices = current_row_indices

        # patch existant
        if out_patches:
            out_patches.append(merge_rows_into_patch(list_of_rows))
            out_indices.append(list_of_rows_indices)
        else:
            out_patches = [merge_rows_into_patch(list_of_rows)]
            out_indices = [list_of_rows_indices]
    # return
    if return_indices:
        return out_patches, out_indices

    return out_patches


def lexsort(keys, dim=-1) -> list[int]:
    """Emulates np.lexsort to sort a 2-dimensional array according to columns indexed by `keys`
    Source: https://discuss.pytorch.org/t/numpy-lexsort-equivalent-in-pytorch/47850/4

    Args:
    - keys  : torch.Tensor to be sorted. Keys in reverse order of importance, i.e. primary key in keys[-1], least significant key in keys[0]

    Returns:
    - idx   : Index set of sorted indices

    Raises:
    -
    """
    if keys.ndim < 2:
        raise ValueError(
            f'keys must be at least 2 dimensional, but {keys.ndim=}.'
        )
    if len(keys) == 0:
        raise ValueError(f'Must have at least 1 key, but {len(keys)=}.')

    idx = keys[0].argsort(dim=dim, stable=True)
    for k in keys[1:]:
        idx = idx.gather(dim, k.gather(dim, idx).argsort(dim=dim, stable=True))

    return idx


def torch_lexsort(T: torch.Tensor, keys: list[int]):
    """Wraps the torch emulation of np.lexsort allowing to sort a tensor for multiple keys indexed by columns idx

    Args:
    - T.    : Torch tensor that is to be sorted. Rows are sorted according to columns with primary key keys[-1], secondary key keys[-2] etc.
    - keys  : Integers representing the column index. Keys in reverse order of importance, i.e. primary key in keys[-1], least significant key in keys[0]

    Returns:
    - T     : Row-sorted torch tensor (according to columns)

    Raises:
    - AssertionError:
    """
    assert len(T.size()) == 2, 'T should be a 2D tensor.'
    assert max(keys) < T.size()[1], 'keys cannot exceed column index of T'

    sort_indices = lexsort(T.t()[keys], dim=-1)

    return sort_indices


def post_process_text(text_raw_results: list[str]) -> list[str]:
    """Post-processing (unpacking, Tex-conversion of text list)"""
    raise NotImplementedError('No code yet.')

    pass


def get_separator_tensor(
    img_path: Path = Path(
        '/home/siebenschuh/Projects/N-O-REO/sep_image/square.png'
    ),
) -> torch.Tensor:
    """Loads the separator symbol from an image and turns it into a tensor"""
    img_path = Path(img_path)
    assert os.path.isfile(
        img_path
    ), f'Path to separator symbol image {img_path} does not exist:'
    assert (
        img_path.suffix == '.png'
    ), 'Image path exists but expected it to be a .png'

    to_tensor = transforms.ToTensor()
    seperator_img = Image.open(img_path).convert('RGB')
    sep_tensor = to_tensor(seperator_img)

    return sep_tensor


def restate_global_patch_indices(
    packed_indices: list[list[list[int]]],
) -> list[list[list[int]]]:
    """Consumes the list of patch indices and re-instates a global batch index (rather than j=0 for each group within the batch) by
    adding the previous groups largest index added to the current indices.
    Global indices are required to look-up meta data for each patch (in a grouped, packed patch).

    Args:
    - List of indices  :  List of list of list. 1st list (groups of patches that can be merged theoretically), 2nd list (actual packed patches)

    Returns:
    - List of indices  : Same structure as input, indices just shifted.

    Raises:
    - AssertionError : Misfit of input and output

    """
    new_packed_indices = []
    j_acc = 0

    # loop sub_indices
    for j, p_patch in enumerate(packed_indices):
        if j > 0:
            j_acc += (
                max(
                    [1]
                    if torch.is_tensor(packed_indices[j - 1])
                    else max(packed_indices[j - 1])
                )
                + 1
            )
        new_packed_indices.append(
            [[j + j_acc for j in p_row] for p_row in p_patch]
        )

    assert len(packed_indices) == len(
        new_packed_indices
    ), 'Length should coicide'
    assert len(packed_indices[0]) == len(
        new_packed_indices[0]
    ), 'Inner list lengths do not coincide.'

    return new_packed_indices


def load_spv05_categories(
    spv05_category_file_path: Path = Path('./meta/spv05_categories.yaml'),
):
    """Load SPv05 category file that includes two dictionaries `categories` and `groups` that define the text/non-text categories

    Args:
    - spv05_category_file_path : File path tot the yaml

    Returns:
    - Tuple of dictionary  :  categories (class name : class ID) and groups (group name : list of class names)

    Raises:
    - AssertionError  :  Yaml file does not exist or does not have the correct format
    """
    # read yaml
    with open(spv05_category_file_path) as file:
        spv05_meta_file = yaml.safe_load(file)

    # categories
    assert all(
        [k in spv05_meta_file.keys() for k in ['names', 'groups']]
    ), "File exists but doesn't have key `names` and `groups` in it."
    inv_categories = spv05_meta_file['names']

    groups = spv05_meta_file['groups']
    categories = {v: k for k, v in inv_categories.items()}

    assert list(categories.values()) == list(
        range(21)
    ), 'SPv05 has 21 classes that are assumed to be represented.'

    return categories, groups


def get_relevant_text_classes(
    file_type: str = 'pdf',
    meta_only: bool = False,
    equation_flag: bool = False,
    table_flag: bool = False,
    fig_flag: bool = False,
    secondary_meta: bool = False,
    table_only: bool = False,
    fig_only: bool = False,
    spv05_category_file_path: Path = Path('./meta/spv05_categories.yaml'),
):
    """Returns class IDs for the SPv05 dataset that are relevant for the input

    Args:
    - file_type      : Document file type that will be handled that determines the category type

    Returns:
    - rel_classes    : List of relevant classes

    Raises:
    - AssertionError : When category file is not found or doesn't have the expected format
    """
    # load category ID
    if file_type == 'pdf':
        dset_category_file_path = spv05_category_file_path
        categories, groups = load_spv05_categories(dset_category_file_path)
    else:
        dset_category_file_path = None
        categories, groups = None, None
        raise NotImplementedError('Only available for PDFs at this point')

    # parse
    if meta_only:
        main_classes = groups['prim_meta']
    else:
        main_classes = groups['main_text'] + groups['prim_meta']

        if equation_flag:
            main_classes += groups['quant_text']
        if secondary_meta:
            main_classes += groups['sec_meta']
        # rel. text
        if table_flag:
            main_classes += groups['table_text']
        if fig_flag:
            main_classes += groups['figure_text']

    # class name -> class id
    rel_classes = {r: categories[r] for r in main_classes}
    rel_classes = dict(sorted(rel_classes.items()))

    return rel_classes


def get_relevant_visual_classes(
    file_type: str = 'pdf',
    table_flag: bool = False,
    fig_flag: bool = False,
    spv05_category_file_path: Path = Path('./meta/spv05_categories.yaml'),
):
    """TODO 2: Finish implementation / logic to track names of columns (tabs and figures are treated differently)

    Returns class IDs for the SPv05 dataset that are relevant for the input

    Args:
    - file_type      : Document file type (which also implies the dataset, e.g. `SPv05` for `PDF`) that will be handled that determines the category type

    Returns:
    - rel_classes    : List of relevant classes

    Raises:
    - AssertionError : When category file is not found or doesn't have the expected format
    """
    # load category ID
    if file_type == 'pdf':
        dset_category_file_path = spv05_category_file_path
        categories, groups = load_spv05_categories(dset_category_file_path)
    else:
        raise NotImplementedError('Only available for PDFs at this point')

    # visual classes
    visual_classes = []
    visual_names = ['table', 'figure']
    if table_flag:
        visual_classes += groups['table_visual']
    if fig_flag:
        visual_classes += groups['figure_visual']

    # class name -> class id
    rel_class = {r: categories[r] for r in visual_classes}
    rel_class = dict(sorted(rel_class.items()))

    return rel_class


def get_packed_patch_tensor(
    tensors: torch.Tensor,
    y: torch.Tensor,
    rel_class_ids: list[int],
    unpackable_class_ids: list[
        int
    ] = None,  # sharp corner : lists as arguments in Python "is a bitch"
    by: list[str] = None,  # sharp corner
    offset: int = 2,
    btm_pad: int = 6,
    max_width: int = 420,
    max_height: int = 420,
    sep_flag: bool = False,
    sep_symbol_flag: bool = False,
    sep_symbol_tensor: torch.Tensor = None,
    dtype: torch.dtype = torch.float16,
):
    """Given a list of variably-sized patches and meta data 2D torch tensor, returns a torch tensor of dimensions (BxCxHxW)

    Args:
    - tensors              : Tensor of page images usually of dimension (Bx3x1280x960)
    - y                    : 2D torch tensor that stores meta data for each patch (row)
    - rel_class_ids        : ...
    - unpackable_class_ids : class ids that are not to be packed but decoded individually for increased accuracy
    - by                   : columns by which patches are *not* to be aggregated (file, page, class). (default: `file_id`)
    - offset               : number of pixels by which a bounding box is extended along each direction of the y- and x-axis (to extend bbox)
    - sep_symbol_tensor    : tensor that is inserted as a separation symbol (if one is used). Default: none as newline is sufficient for separation
    - dtype                : datatype to which the tensor is converted (float16 is the ViT default datatype)

    Returns:
    - packed_patches_tensor : torch Tensor with similarily sized patches (3x420x420)

    Raises:
    - AssertionError     : X
    """
    assert (
        len(tensors.size()) == 4
    ), '`tensors` must be 4-dimensional (BxCxHxW).'
    assert len(y.size()) == 2, '`y` must be a 2-dimensional tensor.'
    assert by is not None, 'Argument `by` is None but should be a list of str'

    # constants
    file_idx_column, page_idx_column, order_idx_column, cls_column = (
        14,
        6,
        13,
        5,
    )

    # subset y given the classes of interest
    y_subset = subset_y_by_class(y_batch=y, rel_class_ids=rel_class_ids)

    # sort y_subset by patch order; infer patch order from meta table --> induces "right" order for decoding for the rest of the code
    patch_ids_in_order = torch_lexsort(
        y_subset[
            :, [file_idx_column, page_idx_column, order_idx_column, cls_column]
        ],
        keys=[2, 1, 0],
    )

    # sort
    y_subset = y_subset[patch_ids_in_order, :]

    # get patch list
    patch_list = get_patch_list(tensors, y_subset, offset=offset)

    # pack patches
    if sorted(rel_class_ids) != sorted(unpackable_class_ids):
        # DEBUG
        # print('unpackable_class_ids: ', unpackable_class_ids)

        # group patches
        grouped_patches, group_indices = grouped_patch_list(
            patch_list,
            y_subset,
            by=by,
            unpackable_class_ids=unpackable_class_ids,
        )

        # packing patches & log indices
        packed_patches_and_indices = [
            get_packed_patch_list(
                groups,
                sep_tensor=sep_symbol_tensor,
                btm_pad=btm_pad,
                return_indices=True,
                sep_flag=sep_flag,
                sep_symbol_flag=sep_symbol_flag,
            )
            for groups in grouped_patches
        ]

        # single list of patches
        packed_patches = [pair[0] for pair in packed_patches_and_indices]
        packed_indices = restate_global_patch_indices(
            [pair[1] for pair in packed_patches_and_indices]
        )

        # flatten patch & index lists
        flat_patches = [item for sublist in packed_patches for item in sublist]
        flat_indices = [idx for sublist in packed_indices for idx in sublist]
        flat_indices = [
            f for f in flat_indices if len(f) > 0
        ]  # del. empty list elements

        index_quadruplet = y_subset[[f[0] for f in flat_indices], :][
            :, [file_idx_column, page_idx_column, order_idx_column, cls_column]
        ].to(torch.int)

        # packed_patch_list -> tensor
        packed_patches_tensor = patch_list_to_tensor(flat_patches, dtype=dtype)
    else:
        packed_patches_tensor = patch_list_to_tensor(patch_list, dtype=dtype)
        index_quadruplet = y_subset[
            :, [file_idx_column, page_idx_column, order_idx_column, cls_column]
        ].to(torch.int)

    # current page ids
    curr_file_ids = torch.unique(index_quadruplet[:, 0])

    return packed_patches_tensor, index_quadruplet, curr_file_ids


def clear_cuda_cache() -> None:
    gc.collect()
    with torch.no_grad():
        torch.cuda.empty_cache()
        torch.clear_autocast_cache()


def update_main_content_dict(
    doc_dict: defaultdict(list),
    text_results: list[str],
    index_quadruplet: torch.Tensor,
    curr_file_ids: torch.Tensor,
    vis_path_dict: defaultdict(list),
    dset_name: str = 'SPv05',
):
    """Given a list of text patch predictions `text_results`, assigns the text elements to the respective documents (via file_id).

    Args:
    - main_doc_dict    : Dictionary (key: doc file id, values: list of sorted decoded text patches)
    - text_results     : list of sorted text segments, decoded from (sorted) list of packed patches
    - index_quadruplet : 2D tensor that stores columns file_id, page_id, order_id, and inferred class_id for each patch (row)
    - curr_file_ids    : file ids that are currently present in batch (if a previous file id is not presented, it has been fully processed -> can be stored)

    Returns:
    - main_doc_dict    : pre-exisitng files appended with current text patches

    Raises:
    - AssertionError   : Lengths of data strcutures does not coincide (i.e. meta data not commensurate to data entries)
    """
    assert len(index_quadruplet) == len(
        text_results
    ), 'Meta data entries and list of text elements must coincide.'

    # constant
    file_id_column = 0
    cls_column = 3
    vis_cat_map = {14: 'Table', 15: 'Figure'}

    # dataset
    dset_name = 'SPv05'
    if dset_name == 'SPv05':
        # meta class IDs and label names (source this from categories.yaml)
        meta_class_id_tensor = torch.tensor(
            [1, 3, 6, 10, 11, 12],
            device=index_quadruplet.device,
            dtype=torch.int,
        )
        meta_class_lab_list = [
            'Title',
            'Equations',
            'Keywords',
            'Author',
            'Institution',
            'Date',
        ]
        meta_added_to_text = ['Title']
    else:
        raise NotImplementedError(
            "Text-based classification only applicable to meta classes of SPv05. Register another dataset's meta class id's here."
        )

    # index
    file_index_set = torch.arange(
        len(index_quadruplet),
        device=index_quadruplet.device,
        dtype=index_quadruplet.dtype,
    )

    # meta classes
    all_index_dict = {}
    for cls_meta_id, cls_meta_label in zip(
        meta_class_id_tensor, meta_class_lab_list
    ):
        all_index_dict[cls_meta_label] = torch.isclose(
            index_quadruplet[:, cls_column], cls_meta_id
        )
    # text (remaining categories)
    tensors_excl_meta = torch.stack(
        [
            all_index_dict[lab]
            for lab in list(
                set(meta_class_lab_list).difference(meta_added_to_text)
            )
        ]
    )
    all_index_dict['Text'] = torch.ones_like(
        index_quadruplet[:, cls_column]
    ).bool() & (~torch.any(tensors_excl_meta, dim=0))

    # assign infered text `text_results` to respective files
    curr_file_ids_on_cpu = curr_file_ids.tolist()
    for file_id, file_id_cpu in zip(curr_file_ids, curr_file_ids_on_cpu):
        # subset text per document
        file_id_indices = torch.isclose(
            index_quadruplet[:, file_id_column], file_id
        )
        file_patch_indices = file_index_set[file_id_indices]
        file_text_results = [text_results[i] for i in file_patch_indices]
        # loop groups
        for cls_label in all_index_dict.keys():
            cls_meta_patch_indices = (
                file_id_indices & all_index_dict[cls_label]
            )
            file_meta_index_subset = file_index_set[
                cls_meta_patch_indices
            ].tolist()
            doc_dict[cls_label][file_id_cpu].extend(
                [text_results[i] for i in file_meta_index_subset]
            )

            # visual assets
            if vis_path_dict:
                for vis_cls_label in vis_path_dict.keys():
                    if file_id_cpu in vis_path_dict[vis_cls_label].keys():
                        doc_dict[vis_cat_map[vis_cls_label]][
                            file_id_cpu
                        ].extend(vis_path_dict[vis_cls_label][file_id_cpu])

    return doc_dict


def store_completed_docs(
    main_doc_dict: defaultdict(list),
    curr_file_ids: torch.Tensor,
    doc_file_paths: list[Path],
    store_dir: Path,
    store_all_now: bool = False,
    output_style: str = 'text',
    file_format: str = 'jsonl',
):
    """Given a dictionary of documents (key: file id, value: text list), stores the respective documents into local files
    if they are completed (i.e. their file ids have stopped to appear in the current batch)

    Args:
    - main_doc_dict    : Dictionary (key: doc file id, values: list of sorted decoded text patches)
    - curr_file_ids    : file ids that are currently present in batch (if a previous file id is not presented, it has been fully processed -> can be stored)
    - doc_file_paths   : List of source paths (sorted by file id) from which the document content was extracted
    - store_dir        : Directory to which the textual output files are stored
    - store_all_now    : If true, entire content of `main_doc_dict` is stored regardless what file ids are observed (used in last batch to "empty out" all docs)
    - output_style     : Representation format (text or tex/latex)
    - file_format      : filetype to which text (regardless of style) is to be stored.

    Returns:
    - None

    Raises:

    """
    # check dir existance
    assert os.path.isdir(
        store_dir
    ), f'The directory to which the text output files are stored `store_dir`={store_dir} does not exist.'

    # file ids to store (all remaining or either completed)
    if store_all_now:
        completed_file_ids = list(main_doc_dict.keys())
    else:
        curr_file_ids_on_cpu = curr_file_ids.detach().cpu().tolist()
        completed_file_ids = list(
            set(main_doc_dict.keys()).difference(set(curr_file_ids_on_cpu))
        )

    assert max(completed_file_ids) < len(
        doc_file_paths
    ), 'File ids extend range of `doc_file_paths`'

    # loop completed file ids
    if len(completed_file_ids) > 0:
        for compl_file_id in completed_file_ids:
            file_text = main_doc_dict.pop(compl_file_id, None)

            # FOR NOW: naïve post-processing of text
            processed_file_text = ''.join(file_text)

            # TODO: actual post-processing
            # processed_file_text = post_process_text(, output_type=['tex', 'markup', 'txt'], spellcheck=[True,False])

            # save TODO. Uses "dataset" below
            file_store_path = (
                store_dir / f'{doc_file_paths[compl_file_id].stem}.txt'
            )
            with open(file_store_path, 'w') as file:
                file.write(processed_file_text)
    pass


def assign_text_inferred_meta_classes(
    txt_cls_model,
    tokenizer,
    batch_size: int,
    index_quadruplet: torch.Tensor,
    text_results: list[str],
    dset_name: str = 'SPv05',
) -> torch.Tensor:
    """Re-assess the detection model's label predictions for meta classes with a text classifier's predictions
    based on pseudo OCR-inferred text.

    Args:
    - txt_cls_model     : Text classification model (DistilBertForSequenceClassification) to infer meta category from ViT-inferred text
    - tokenizer         : Tokenizer for model
    - batch_size        : Batch size for inference
    - index_quadruplet  : 2D tensor that tracks file_id, page_id, order_id, & (inferred) class label id for each patch (row)
    - text_results      : list of str
    - dset_name         :

    Returns:
    - index_quadruplet  : 2D tensor w/ modified cls_label column for patches that are inferred to be meta categories

    Raises:
    -

    """
    # constant
    cls_column = 3

    # dataset
    if dset_name == 'SPv05':
        class_id_mapping_tensor = torch.tensor(
            [1, 6, 10, 11, 12], device=index_quadruplet.device, dtype=torch.int
        )

    else:
        raise NotImplementedError(
            "Text-based classification only applicable to meta classes of SPv05. Register another dataset's meta class id's here."
        )

    # subset to meta classes
    mask = torch.isin(index_quadruplet[:, cls_column], class_id_mapping_tensor)
    meta_indices = torch.arange(
        len(index_quadruplet), device=index_quadruplet.device
    )[mask]
    meta_text_results = [text_results[i] for i in meta_indices.tolist()]

    # batch
    n_meta = len(meta_text_results)
    n_eff = n_meta // batch_size

    # predict meta categories based on text
    text_results = []
    for j in range(n_eff + 1):
        # subset indices
        j_min, j_max = j * batch_size, min((j + 1) * batch_size, n_meta)

        # run inference
        inputs = tokenizer(
            meta_text_results[j_min:j_max], return_tensors='pt', padding=True
        ).to(index_quadruplet.device)
        logits = txt_cls_model(**inputs).logits
        pred_cls_id = logits.argmax(dim=1)
        pred_vals = class_id_mapping_tensor[pred_cls_id]
        index_quadruplet[meta_indices[j_min:j_max], cls_column] = pred_vals

    return index_quadruplet


def extract_file_specific_doc_dict(
    doc_dict, file_id: int, LaTex2Text: LatexNodes2Text
):
    """Extract doc-specific dict (each category is a key) and the file_id key is dropped.

    Args:
    - doc_dict   : `defaultdict(lambda: defaultdict(list))` holding a batch of documents with primary keys `categories` (`Title`, `Text` etc.)
                    and secondary key `file_id`
    - file_id    :  Integer ID for the specific document that is to be extracted

    Returns:
    - file_dict  :

    Raises:
    -
    """
    # pattern
    pattern = re.compile(r'(\n\s*)+')

    # DEBUG
    # print(doc_dict.keys())

    file_dict = {}
    for key in doc_dict.keys():
        if file_id in doc_dict[key]:
            # post-process
            if key in ['Equations', 'Table', 'Figure']:
                file_dict[key] = doc_dict[key][file_id]  # append list
            else:
                extracted_text = '\n'.join(doc_dict[key][file_id])
                if len(extracted_text) > 0:
                    try:
                        proc_text = LaTex2Text.latex_to_text(extracted_text)
                        file_dict[key] = re.sub(pattern, '\n', proc_text)
                    except:
                        print(extracted_text)
    return file_dict


def store_completed_docs(
    doc_dict: defaultdict(lambda: defaultdict(list)),
    curr_file_ids: torch.Tensor,
    doc_file_paths: list[Path],
    store_dir: Path,
    LaTex2Text: LatexNodes2Text,
    store_all_now: bool = False,
    output_style: str = 'text',
    file_format: str = 'jsonl',
):
    """Given a dictionary of documents (key: file id, value: text list), stores the respective documents into local files
    if they are completed (i.e. their file ids have stopped to appear in the current batch)

    Args:
    - main_doc_dict    : Dictionary (key: doc file id, values: list of sorted decoded text patches)
    - curr_file_ids    : file ids that are currently present in batch (if a previous file id is not presented, it has been fully processed -> can be stored)
    - doc_file_paths   : List of source paths (sorted by file id) from which the document content was extracted
    - store_dir        : Directory to which the textual output files are stored
    - store_all_now    : If true, entire content of `main_doc_dict` is stored regardless what file ids are observed (used in last batch to "empty out" all docs)
    - output_style     : Representation format (text or tex/latex)
    - file_format      : filetype to which text (regardless of style) is to be stored.

    Returns:
    - None

    Raises:

    """
    # check dir existance
    assert os.path.isdir(
        store_dir
    ), f'The directory to which the text output files are stored `store_dir`={store_dir} does not exist.'

    # file ids to store (all remaining or either completed)
    if store_all_now:
        completed_file_ids = list(doc_dict['Text'].keys())
    else:
        curr_file_ids_on_cpu = curr_file_ids.tolist()
        completed_file_ids = list(
            set(doc_dict['Text'].keys()).difference(set(curr_file_ids_on_cpu))
        )

    if len(completed_file_ids) > 0:
        assert max(completed_file_ids) < len(
            doc_file_paths
        ), 'File ids extend range of `doc_file_paths`'

    # loop completed file ids
    if len(completed_file_ids) > 0:
        for compl_file_id in completed_file_ids:
            # extract
            one_doc_dict = extract_file_specific_doc_dict(
                doc_dict, compl_file_id, LaTex2Text
            )

            # save TODO. Uses "dataset" below
            file_store_path = (
                store_dir / f'{doc_file_paths[compl_file_id].stem}.json'
            )
            with open(file_store_path, 'w') as f:
                json.dump(one_doc_dict, f)
    pass


def store_visuals(
    tensors: torch.Tensor,
    y: torch.Tensor,
    rel_visual_classes: dict[str, int],
    file_paths: list[str],
    file_ids: list[int],
    output_dir: Path,
    i_tab: int,
    i_fig: int,
    prev_file_id: int,
) -> None:
    """Extracts and stores visuals (e.g. figures and/or tables) into the respective subdirectory in `output_dir`

    Args:
    - tensors            : Tensor of batched patch images of dimension BxCxHxW with C,H,W=3,1280,960 usually
    - y                  : 2D tensor of patch meta data (location, inferred class label, score, file_id etc.)
    - rel_visual_classes : Dictionary with relevant visual class ids/names (Table and or Figure) that are to be stored
    - file_paths         : List of file paths of the particular batch relating to `tensors` and `y`
    - file_ids           :    -||-       IDs               -||-
    - output_dir         : location where patches are to be stored (be default, in subdirectories `Table` and `Figure`)
    - i_tab              : current patch index of table  (of the current file id), allows ordering of table images across batches
    - i_fig              :            -||-        figure         -||-
    - prev_file_id       : previous batch's last file_id (required to properly increment and index table/figure patches)

    Returns:
    - (vis_path_dict, i_tab, i_fig, last_file_id) : Tuple of vis path dictionary, table index, figure index, and last file_id

    Raises:

    """
    # CONSTANTS
    xmin_column = 0
    ymin_column = 1
    xmax_column = 2
    ymax_column = 3
    cls_column = 5
    page_idx_column = 6
    file_idx_column = 14

    # store in dict
    i_dict = {14: i_tab, 15: i_fig}

    # Path conversion
    output_dir = Path(output_dir)
    assert os.path.isdir(
        output_dir
    ), 'Target directory `output_dir` must exist.'

    # class id -> class name
    inv_rel_visual_classes = {v: k for k, v in rel_visual_classes.items()}

    # list
    visual_class_ids = list(rel_visual_classes.values())
    y_sub_visual = (
        subset_y_by_class(y_batch=y, rel_class_ids=visual_class_ids)
        .round()
        .to(torch.int)
    )

    # clip
    _, C, H, W = tensors.size()
    y_sub_visual[:, ymin_column] = y_sub_visual[:, ymin_column].clamp(
        min=0, max=H
    )
    y_sub_visual[:, ymax_column] = y_sub_visual[:, ymax_column].clamp(
        min=0, max=H
    )
    y_sub_visual[:, xmin_column] = y_sub_visual[:, xmin_column].clamp(
        min=0, max=W
    )
    y_sub_visual[:, xmax_column] = y_sub_visual[:, xmax_column].clamp(
        min=0, max=W
    )

    # mapping file_id -> filename
    file_id_name_mapping = {
        f_id: f_name for (f_id, f_name) in zip(file_ids, file_paths)
    }

    # create dir if necessary
    for vis_name, vis_class_id in rel_visual_classes.items():
        store_path = output_dir / vis_name
        os.makedirs(store_path, exist_ok=True)

    # to list
    cls_id_list = y_sub_visual[:, cls_column].tolist()
    file_id_list = y_sub_visual[:, file_idx_column].tolist()

    # filepath list
    vis_path_dict = defaultdict(lambda: defaultdict(list))

    # loop visual patches
    for i, (cls_id, file_id) in enumerate(zip(cls_id_list, file_id_list)):
        # localize patch
        x_min, y_min, x_max, y_max = y_sub_visual[i, : (ymax_column + 1)]
        b = y_sub_visual[i, page_idx_column]

        # extract patch
        img_patch = transforms.ToPILImage()(
            tensors[b, :, y_min:y_max, x_min:x_max]
        )

        # derive file path (incl. sub-directory)
        patch_file_path = (
            output_dir
            / inv_rel_visual_classes[cls_id]
            / f'{inv_rel_visual_classes[cls_id]}_{file_id_name_mapping[file_id].stem}_{i_dict[cls_id]}.png'
        )
        i_dict[cls_id] += 1

        # append path
        vis_path_dict[cls_id][file_id].append(str(patch_file_path.absolute()))

        # store
        try:
            img_patch.save(patch_file_path)
        except:
            print('x_min, y_min, x_max, y_max : ', x_min, y_min, x_max, y_max)
            print('tensors.size() : ', tensors.size())

    last_file_id = file_id

    # DEBUG
    print('i_dict[14], i_dict[15] : ', i_dict[14], i_dict[15])

    return vis_path_dict, i_dict[14], i_dict[15], last_file_id
