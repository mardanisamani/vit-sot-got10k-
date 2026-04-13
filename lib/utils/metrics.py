"""
Tracking Evaluation Metrics
=============================
Standard metrics used in single object tracking:

1. Success (AUC):
   - Plot overlap threshold τ ∈ [0, 1] vs. percentage of frames with IoU ≥ τ
   - Report the Area Under Curve (AUC)
   - GOT-10k uses AO (Average Overlap) as the primary metric

2. Precision:
   - Plot distance threshold (pixels) vs. percentage of frames within that
   - Report at threshold 20 pixels (P₂₀)

3. GOT-10k official metrics:
   - AO:  Average Overlap (mean IoU across all frames)
   - SR₀.₅: Success Rate at IoU ≥ 0.5
   - SR₀.₇₅: Success Rate at IoU ≥ 0.75
"""

import numpy as np
from typing import List, Tuple


def compute_iou(pred: np.ndarray, gt: np.ndarray) -> float:
    """
    Compute IoU between predicted and ground-truth boxes.

    Args:
        pred: [x1, y1, x2, y2]
        gt:   [x1, y1, x2, y2]

    Returns:
        IoU in [0, 1]
    """
    ix1 = max(pred[0], gt[0])
    iy1 = max(pred[1], gt[1])
    ix2 = min(pred[2], gt[2])
    iy2 = min(pred[3], gt[3])

    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    if inter == 0:
        return 0.0

    area_pred = max(0, pred[2] - pred[0]) * max(0, pred[3] - pred[1])
    area_gt = max(0, gt[2] - gt[0]) * max(0, gt[3] - gt[1])
    union = area_pred + area_gt - inter

    return inter / (union + 1e-6)


def compute_center_error(pred: np.ndarray, gt: np.ndarray) -> float:
    """Center point distance (pixels) between pred and gt boxes."""
    pred_cx = (pred[0] + pred[2]) / 2
    pred_cy = (pred[1] + pred[3]) / 2
    gt_cx = (gt[0] + gt[2]) / 2
    gt_cy = (gt[1] + gt[3]) / 2
    return np.sqrt((pred_cx - gt_cx) ** 2 + (pred_cy - gt_cy) ** 2)


def evaluate_sequence(
    pred_boxes: List[np.ndarray],
    gt_boxes: List[np.ndarray],
) -> dict:
    """
    Evaluate a single sequence.

    Args:
        pred_boxes: List of (N,) [x1,y1,x2,y2] predictions (N frames).
        gt_boxes:   List of (N,) [x1,y1,x2,y2] ground truths.

    Returns:
        dict with AO, SR50, SR75, precision metrics.
    """
    assert len(pred_boxes) == len(gt_boxes)

    ious = [compute_iou(p, g) for p, g in zip(pred_boxes, gt_boxes)]
    errors = [compute_center_error(p, g) for p, g in zip(pred_boxes, gt_boxes)]

    ious = np.array(ious)
    errors = np.array(errors)

    # GOT-10k primary metrics
    ao = ious.mean()
    sr50 = (ious >= 0.5).mean()
    sr75 = (ious >= 0.75).mean()

    # Success plot: AUC
    thresholds = np.linspace(0, 1, 101)
    success_rates = [(ious >= t).mean() for t in thresholds]
    auc = np.mean(success_rates)

    # Precision plot: percentage within 20px
    prec20 = (errors <= 20).mean()

    return {
        "AO": float(ao),
        "SR50": float(sr50),
        "SR75": float(sr75),
        "AUC": float(auc),
        "Precision@20": float(prec20),
        "mean_iou": float(ao),
        "mean_center_error": float(errors.mean()),
    }


def summarize_results(seq_results: List[dict]) -> dict:
    """Average metrics across all sequences."""
    if not seq_results:
        return {}
    keys = seq_results[0].keys()
    return {k: float(np.mean([r[k] for r in seq_results])) for k in keys}
