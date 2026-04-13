"""
OSTrack: One-Stream Tracking with ViT
======================================
Paper: "One-Stream Exploration-Exploitation Network for Single Object
        Tracking" — Ye et al., ECCV 2022
        https://arxiv.org/abs/2203.11991

This module assembles the full OSTrack model:
    backbone (ViT) + head (CornerHead)

The forward pass handles both:
  - Training mode: (template, search, gt_boxes) → losses
  - Inference mode: (template, search) → predicted boxes
"""

import torch
import torch.nn as nn
from .vit_backbone import build_vit_backbone
from .head import build_head, TrackingLoss


class OSTrack(nn.Module):
    """
    OSTrack: One-stream ViT-based single object tracker.

    Design principles:
    1. One-stream: template and search are processed together through ViT.
       All 12 transformer layers see both template and search tokens.
    2. No cross-attention module needed: standard self-attention handles it.
    3. Simple head: reshape search tokens → 2D map → predict corners.

    Comparison with two-stream approaches (SiamRPN, SiamFC):
    ┌────────────────┬────────────────┬────────────────────┐
    │                │  Two-stream    │   One-stream (ours) │
    ├────────────────┼────────────────┼────────────────────┤
    │ Template/Search│ Separate       │ Jointly in ViT     │
    │ Interaction    │ Late fusion    │ From layer 1       │
    │ Backbone       │ CNN (typically)│ ViT                │
    │ Complexity     │ High (x-corr)  │ Low (simple head)  │
    │ Performance    │ Good           │ State-of-the-art   │
    └────────────────┴────────────────┴────────────────────┘
    """

    def __init__(self, backbone: nn.Module, head: nn.Module):
        super().__init__()
        self.backbone = backbone
        self.head = head

    def forward(
        self,
        template: torch.Tensor,          # (B, 3, Hz, Wz)
        search: torch.Tensor,             # (B, 3, Hx, Wx)
        gt_boxes: torch.Tensor = None,   # (B, 4) [x1,y1,x2,y2] normalized
    ):
        """
        Forward pass.

        Training:  pass gt_boxes → returns loss dict
        Inference: gt_boxes=None → returns prediction dict

        Args:
            template: Template crop from the first frame.
            search:   Search crop from the current frame.
            gt_boxes: Ground-truth boxes (training only).

        Returns:
            Training:  {"total": scalar, "l1": scalar, "giou": scalar, ...}
            Inference: {"pred_boxes": (B,4), "score_map": (B,1,H,W)}
        """
        # ── Backbone: joint processing of template + search ──────────────────
        z_tokens, x_tokens = self.backbone(template, search)
        # z_tokens: (B, Nz, D) — template features after ViT
        # x_tokens: (B, Nx, D) — search features after ViT

        # ── Head: predict bounding box from search tokens ─────────────────────
        predictions = self.head(x_tokens)
        # predictions: {"pred_boxes": (B,4), "score_map": (B,1,H,W)}

        if gt_boxes is not None:
            # Training: compute losses
            criterion = TrackingLoss()
            losses = criterion(
                predictions["pred_boxes"],
                gt_boxes,
                predictions.get("score_map"),
                None,  # target_mask (Gaussian heatmap) — extend as needed
            )
            return losses

        return predictions

    def track_init(self, template_img: torch.Tensor, template_box: torch.Tensor):
        """
        Initialize tracker with first-frame template.

        Stores the template features for reuse across all subsequent frames.
        This avoids recomputing template patch embeddings each frame.

        Args:
            template_img: (1, 3, Hz, Wz) first-frame crop.
            template_box: (1, 4) ground-truth box in the first frame.
        """
        self.eval()
        with torch.no_grad():
            self._template_img = template_img
            self._template_box = template_box
            # Pre-embed the template (reused every frame)
            self._z_embed = self.backbone.patch_embed_image(template_img)
            self._z_embed = self._z_embed + self.backbone.pos_embed_z

    def track_step(self, search_img: torch.Tensor) -> dict:
        """
        Track in the current frame using stored template.

        Args:
            search_img: (1, 3, Hx, Wx) current-frame search crop.

        Returns:
            dict with "pred_boxes" and "score_map".
        """
        self.eval()
        with torch.no_grad():
            # Embed search region
            x_embed = self.backbone.patch_embed_image(search_img)
            x_embed = x_embed + self.backbone.pos_embed_x

            # Concatenate with stored template tokens
            tokens = torch.cat([self._z_embed, x_embed], dim=1)

            # Run transformer + norm
            tokens = self.backbone.blocks(tokens)
            tokens = self.backbone.norm(tokens)

            # Split and predict
            x_tokens = tokens[:, self.backbone.num_z_patches:]
            return self.head(x_tokens)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def build_ostrack(cfg: dict) -> OSTrack:
    """Build OSTrack from a config dictionary."""
    backbone = build_vit_backbone(cfg)
    head = build_head(cfg)
    return OSTrack(backbone, head)


def build_ostrack_small(pretrained: bool = False) -> OSTrack:
    """Convenience: OSTrack with ViT-Small (faster, good for experiments)."""
    cfg = {
        "backbone": {
            "pretrained_model": "vit_small_patch16_224",
            "pretrained": pretrained,
            "patch_size": 16,
            "embed_dim": 384,
            "depth": 12,
            "num_heads": 6,
            "mlp_ratio": 4.0,
            "drop_path_rate": 0.1,
        },
        "head": {
            "type": "corner",
            "hidden_dim": 128,
            "num_layers": 3,
        },
        "template_size": 128,
        "search_size": 256,
    }
    return build_ostrack(cfg)


def build_ostrack_base(pretrained: bool = False) -> OSTrack:
    """Convenience: OSTrack with ViT-Base (standard, best performance)."""
    cfg = {
        "backbone": {
            "pretrained_model": "vit_base_patch16_224",
            "pretrained": pretrained,
            "patch_size": 16,
            "embed_dim": 768,
            "depth": 12,
            "num_heads": 12,
            "mlp_ratio": 4.0,
            "drop_path_rate": 0.1,
        },
        "head": {
            "type": "corner",
            "hidden_dim": 256,
            "num_layers": 3,
        },
        "template_size": 128,
        "search_size": 256,
    }
    return build_ostrack(cfg)
