"""Utility functions for preparing visual prompt support samples."""

from __future__ import annotations

from typing import Tuple

import torch
from torch import Tensor
from torch.nn import functional as F


def extract_support_crops(image: Tensor, bboxes: Tensor, output_size: int) -> Tuple[Tensor, Tensor]:
    """Extract object crops from an image using bounding boxes and resize them to a fixed resolution.

    Args:
        image (Tensor): Image tensor in ``(C, H, W)`` format.
        bboxes (Tensor): Bounding boxes in normalized ``xywh`` format with shape ``(N, 4)``.
        output_size (int): Target height and width for each cropped support image.

    Returns:
        Tuple[Tensor, Tensor]:
            - Cropped support tensors stacked along the first dimension with shape
              ``(N, C, output_size, output_size)``.
            - Indices mapping each crop to the corresponding entry in ``bboxes``.
    """

    if image.ndim != 3:
        raise ValueError("Expected image tensor with shape (C, H, W).")
    if bboxes.ndim != 2 or bboxes.shape[-1] != 4:
        raise ValueError("Bounding boxes must have shape (N, 4) in normalized xywh format.")

    num_boxes = bboxes.shape[0]
    device = image.device
    dtype = torch.float32
    channels, height, width = image.shape

    if num_boxes == 0:
        empty_crops = torch.zeros((0, channels, output_size, output_size), dtype=dtype, device=device)
        empty_indices = torch.zeros((0,), dtype=torch.long, device=device)
        return empty_crops, empty_indices

    boxes = bboxes.to(dtype=torch.float32).clone()
    boxes[:, 0] *= width
    boxes[:, 1] *= height
    boxes[:, 2] *= width
    boxes[:, 3] *= height

    half_w = boxes[:, 2] / 2
    half_h = boxes[:, 3] / 2
    x1 = (boxes[:, 0] - half_w).clamp(0, width - 1)
    y1 = (boxes[:, 1] - half_h).clamp(0, height - 1)
    x2 = (boxes[:, 0] + half_w).clamp(0, width)
    y2 = (boxes[:, 1] + half_h).clamp(0, height)

    x1 = torch.floor(x1).to(dtype=torch.long)
    y1 = torch.floor(y1).to(dtype=torch.long)
    x2 = torch.ceil(x2).to(dtype=torch.long)
    y2 = torch.ceil(y2).to(dtype=torch.long)

    x2 = torch.maximum(x2, x1 + 1)
    y2 = torch.maximum(y2, y1 + 1)

    image_float = image.to(dtype=dtype)
    crops = []
    for left, top, right, bottom in zip(x1, y1, x2, y2):
        crop = image_float[:, top:bottom, left:right]
        if crop.numel() == 0:
            crop = image_float.new_zeros((channels, output_size, output_size))
        else:
            crop = F.interpolate(
                crop.unsqueeze(0), size=(output_size, output_size), mode="bilinear", align_corners=False
            ).squeeze(0)
        crops.append(crop)

    support_crops = torch.stack(crops, dim=0)
    indices = torch.arange(num_boxes, device=device, dtype=torch.long)
    return support_crops, indices
