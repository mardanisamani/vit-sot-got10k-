"""
Bounding Box Operations
========================
All boxes are in [x1, y1, x2, y2] (XYXY) format unless noted otherwise.
"""

import torch
import numpy as np
from typing import Union

Boxes = Union[torch.Tensor, np.ndarray]


def box_iou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    """
    Compute pairwise IoU between two sets of boxes.

    Args:
        boxes1: (N, 4) tensor of [x1, y1, x2, y2]
        boxes2: (M, 4) tensor of [x1, y1, x2, y2]

    Returns:
        iou: (N, M) IoU matrix
    """
    area1 = box_area(boxes1)  # (N,)
    area2 = box_area(boxes2)  # (M,)

    # Intersection
    lt = torch.max(boxes1[:, None, :2], boxes2[None, :, :2])  # (N, M, 2)
    rb = torch.min(boxes1[:, None, 2:], boxes2[None, :, 2:])  # (N, M, 2)
    wh = (rb - lt).clamp(min=0)  # (N, M, 2)
    inter = wh[:, :, 0] * wh[:, :, 1]  # (N, M)

    union = area1[:, None] + area2[None, :] - inter
    return inter / (union + 1e-6)


def box_area(boxes: torch.Tensor) -> torch.Tensor:
    """Compute areas of boxes. (N, 4) → (N,)"""
    return (boxes[:, 2] - boxes[:, 0]).clamp(0) * (boxes[:, 3] - boxes[:, 1]).clamp(0)


def box_cxcywh_to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
    """Convert [cx, cy, w, h] → [x1, y1, x2, y2]."""
    cx, cy, w, h = boxes.unbind(-1)
    return torch.stack([cx - w/2, cy - h/2, cx + w/2, cy + h/2], dim=-1)


def box_xyxy_to_cxcywh(boxes: torch.Tensor) -> torch.Tensor:
    """Convert [x1, y1, x2, y2] → [cx, cy, w, h]."""
    x1, y1, x2, y2 = boxes.unbind(-1)
    return torch.stack([(x1+x2)/2, (y1+y2)/2, x2-x1, y2-y1], dim=-1)


def scale_box_to_image(
    box_norm: np.ndarray,
    search_crop_box: np.ndarray,  # [x1, y1, x2, y2] of search crop in image
) -> np.ndarray:
    """
    Map normalized box in search crop back to absolute image coordinates.

    Args:
        box_norm:       [x1, y1, x2, y2] normalized to [0, 1] in search crop.
        search_crop_box: [x1, y1, x2, y2] of the search crop in the full image.

    Returns:
        box_abs: [x1, y1, x2, y2] in absolute image coordinates.
    """
    cx1, cy1, cx2, cy2 = search_crop_box
    cw = cx2 - cx1
    ch = cy2 - cy1

    x1 = cx1 + box_norm[0] * cw
    y1 = cy1 + box_norm[1] * ch
    x2 = cx1 + box_norm[2] * cw
    y2 = cy1 + box_norm[3] * ch
    return np.array([x1, y1, x2, y2], dtype=np.float32)


def clip_boxes(boxes: np.ndarray, img_size: tuple) -> np.ndarray:
    """Clip boxes to image boundaries. img_size = (W, H)."""
    w, h = img_size
    boxes[..., 0] = np.clip(boxes[..., 0], 0, w)
    boxes[..., 1] = np.clip(boxes[..., 1], 0, h)
    boxes[..., 2] = np.clip(boxes[..., 2], 0, w)
    boxes[..., 3] = np.clip(boxes[..., 3], 0, h)
    return boxes
