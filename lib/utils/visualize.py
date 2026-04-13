"""
Visualization Utilities for Tracking
======================================
Helper functions to:
  - Draw bounding boxes on frames
  - Plot success/precision curves
  - Visualize attention maps from ViT
  - Create tracking demo videos
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image, ImageDraw, ImageFont
from typing import List, Optional, Tuple
import os


def draw_box(
    image: np.ndarray,
    box: np.ndarray,         # [x1, y1, x2, y2]
    color: tuple = (0, 255, 0),
    thickness: int = 2,
    label: str = None,
) -> np.ndarray:
    """Draw a bounding box on a numpy image array (H, W, 3)."""
    img = image.copy()
    x1, y1, x2, y2 = [int(v) for v in box]

    try:
        import cv2
        cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
        if label:
            cv2.putText(img, label, (x1, y1 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    except ImportError:
        # Fallback: PIL drawing
        pil = Image.fromarray(img)
        draw = ImageDraw.Draw(pil)
        draw.rectangle([x1, y1, x2, y2], outline=color, width=thickness)
        img = np.array(pil)

    return img


def visualize_prediction(
    template: np.ndarray,    # (H, W, 3) uint8
    search: np.ndarray,      # (H, W, 3) uint8
    pred_box: np.ndarray,    # [x1, y1, x2, y2] normalized [0,1]
    gt_box: np.ndarray = None,  # [x1, y1, x2, y2] normalized
    title: str = "Tracking Prediction",
    save_path: str = None,
) -> plt.Figure:
    """
    Side-by-side visualization of template and search with boxes.

    Args:
        template:  Template crop as uint8 numpy array.
        search:    Search crop as uint8 numpy array.
        pred_box:  Predicted box in search crop, normalized [0,1].
        gt_box:    Ground-truth box (optional).
        title:     Figure title.
        save_path: If provided, save figure to this path.
    """
    H_s, W_s = search.shape[:2]

    # Denormalize box to pixel coords
    def denorm(box, H, W):
        return [box[0]*W, box[1]*H, box[2]*W, box[3]*H]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(title, fontsize=14)

    # Template
    axes[0].imshow(template)
    axes[0].set_title("Template (Frame 1)")
    axes[0].axis("off")

    # Search with predicted box
    axes[1].imshow(search)
    axes[1].set_title("Search Region")
    axes[1].axis("off")

    pred_px = denorm(pred_box, H_s, W_s)
    rect_pred = patches.Rectangle(
        (pred_px[0], pred_px[1]),
        pred_px[2] - pred_px[0], pred_px[3] - pred_px[1],
        linewidth=2, edgecolor="lime", facecolor="none",
        label="Prediction"
    )
    axes[1].add_patch(rect_pred)

    if gt_box is not None:
        gt_px = denorm(gt_box, H_s, W_s)
        rect_gt = patches.Rectangle(
            (gt_px[0], gt_px[1]),
            gt_px[2] - gt_px[0], gt_px[3] - gt_px[1],
            linewidth=2, edgecolor="red", facecolor="none",
            label="Ground Truth"
        )
        axes[1].add_patch(rect_gt)

    axes[1].legend(loc="upper right", fontsize=9)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_success_curve(
    success_rates: np.ndarray,   # (101,) values at thresholds 0..1
    auc: float,
    label: str = "OSTrack",
    color: str = "tab:blue",
    ax: plt.Axes = None,
    save_path: str = None,
) -> plt.Figure:
    """Plot success curve (overlap threshold vs. success rate)."""
    thresholds = np.linspace(0, 1, len(success_rates))

    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 5))
    else:
        fig = ax.get_figure()

    ax.plot(thresholds, success_rates, color=color,
            label=f"{label} [AUC={auc:.3f}]", linewidth=2)
    ax.set_xlabel("Overlap Threshold", fontsize=12)
    ax.set_ylabel("Success Rate", fontsize=12)
    ax.set_title("Success Plot (GOT-10k)", fontsize=13)
    ax.legend(fontsize=11)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.3)

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def visualize_attention(
    attn_weights: np.ndarray,    # (num_heads, Nz+Nx, Nz+Nx)
    search_size: int = 256,
    patch_size: int = 16,
    template_size: int = 128,
    head_idx: int = 0,
    save_path: str = None,
) -> plt.Figure:
    """
    Visualize ViT attention: how much each search token attends to others.

    Args:
        attn_weights: Attention matrix from a ViT layer.
        search_size:  Size of the search crop.
        patch_size:   Patch size used in ViT.
        template_size: Size of the template crop.
        head_idx:     Which attention head to visualize.
    """
    Nz = (template_size // patch_size) ** 2
    Nx = (search_size // patch_size) ** 2
    Hx = Wx = search_size // patch_size

    attn = attn_weights[head_idx]  # (Nz+Nx, Nz+Nx)

    # Attention from search tokens to all tokens
    x_to_all = attn[Nz:, :]        # (Nx, Nz+Nx)
    x_to_z = x_to_all[:, :Nz]      # (Nx, Nz) — search attends to template
    x_to_x = x_to_all[:, Nz:]      # (Nx, Nx) — search self-attention

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Search→Template attention map
    s2t = x_to_z.sum(axis=-1).reshape(Hx, Wx)
    axes[0].imshow(s2t, cmap="hot", interpolation="nearest")
    axes[0].set_title(f"Search→Template Attention (head {head_idx})")
    axes[0].axis("off")

    # Search self-attention map
    s2s = x_to_x.mean(axis=0).reshape(Hx, Wx)
    axes[1].imshow(s2s, cmap="hot", interpolation="nearest")
    axes[1].set_title(f"Search Self-Attention (head {head_idx})")
    axes[1].axis("off")

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig
