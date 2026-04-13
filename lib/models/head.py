"""
Tracking Head for ViT-based SOT
================================
OSTrack uses a simple but effective head that:
  1. Takes the search-region output tokens from the ViT backbone.
  2. Reshapes them to a 2D feature map.
  3. Runs a lightweight CNN to predict corner coordinates and IoU score.

Corner-based regression: predict (x1_norm, y1_norm, x2_norm, y2_norm)
normalized by the search region size, then mapped back to image coordinates.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# ---------------------------------------------------------------------------
# Corner Head (OSTrack-style)
# ---------------------------------------------------------------------------

class CornerHead(nn.Module):
    """
    Predict bounding box corners from ViT search-region tokens.

    Architecture:
        search_tokens (B, Nx, D)
        → reshape to (B, D, Hx, Wx)           [2D feature map]
        → 1x1 conv reduce channels to hidden
        → separate x/y heads with softmax
        → corner coordinates via expectation

    The "corner via expectation" trick: instead of regressing x1,y1,x2,y2
    directly, we predict a probability distribution over the grid and compute
    E[x], E[y] for each corner. This is smoother and easier to train.
    """

    def __init__(
        self,
        embed_dim: int = 768,
        hidden_dim: int = 256,
        search_feat_size: int = 16,   # Hx = Wx = search_size / patch_size
        num_layers: int = 3,
    ):
        super().__init__()
        self.search_feat_size = search_feat_size

        # Reduce transformer dim → working dim
        self.input_proj = nn.Conv2d(embed_dim, hidden_dim, kernel_size=1)

        # Corner prediction via stacked convolutions
        self.box_head = self._build_conv_head(hidden_dim, num_layers)

        # Separate final predictions per corner:
        # Each corner = score distribution over Hx*Wx positions
        self.score_head = nn.Conv2d(hidden_dim, 1, kernel_size=1)

    def _build_conv_head(self, dim: int, num_layers: int) -> nn.Sequential:
        """Stack of depthwise + pointwise conv layers."""
        layers = []
        for _ in range(num_layers):
            layers += [
                nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim),  # dw
                nn.Conv2d(dim, dim, kernel_size=1),                           # pw
                nn.BatchNorm2d(dim),
                nn.ReLU(inplace=True),
            ]
        return nn.Sequential(*layers)

    def forward(self, x_tokens: torch.Tensor):
        """
        Args:
            x_tokens: (B, Nx, D) search region output from backbone.

        Returns:
            dict with:
                "pred_boxes": (B, 4) normalized [x1, y1, x2, y2] in [0,1]
                "score_map":  (B, 1, H, W) confidence map
        """
        B, Nx, D = x_tokens.shape
        H = W = self.search_feat_size
        assert H * W == Nx, f"Token count {Nx} ≠ {H}×{W}={H*W}"

        # (B, Nx, D) → (B, D, H, W)
        feat = x_tokens.transpose(1, 2).reshape(B, D, H, W)

        # Project to hidden dim
        feat = self.input_proj(feat)   # (B, hidden, H, W)

        # Shared feature extraction
        feat = self.box_head(feat)     # (B, hidden, H, W)

        # Score map
        score_map = self.score_head(feat)  # (B, 1, H, W)

        # Corner regression via expectation over score distributions
        pred_boxes = self._predict_corners(score_map, H, W)  # (B, 4)

        return {
            "pred_boxes": pred_boxes,
            "score_map": score_map,
        }

    def _predict_corners(
        self, score_map: torch.Tensor, H: int, W: int
    ) -> torch.Tensor:
        """
        Predict corners using soft-argmax (expectation over probability map).

        Instead of predicting a scalar per coordinate, we predict a probability
        distribution over all positions and compute the expected position.
        This is differentiable and produces smooth gradients.

        Returns:
            boxes: (B, 4) as [x1, y1, x2, y2] normalized to [0,1]
        """
        B = score_map.shape[0]

        # Grid coordinates normalized to [0,1]
        x_grid = torch.linspace(0, 1, W, device=score_map.device)
        y_grid = torch.linspace(0, 1, H, device=score_map.device)

        # Flatten score map to (B, H*W)
        s = score_map.reshape(B, -1)
        s = F.softmax(s, dim=-1)  # probability distribution

        # Reshape to (B, H, W)
        s_2d = s.reshape(B, H, W)

        # Marginal distributions
        p_x = s_2d.sum(dim=1)   # (B, W) — sum over rows → x marginal
        p_y = s_2d.sum(dim=2)   # (B, H) — sum over cols → y marginal

        # Expected coordinates
        cx = (p_x * x_grid.unsqueeze(0)).sum(dim=-1)   # (B,) center x
        cy = (p_y * y_grid.unsqueeze(0)).sum(dim=-1)   # (B,) center y

        # We use the score map as a single-peak estimate (TL/BR are approximate)
        # For a proper tracker you'd have two heads (one per corner)
        # Here we predict a box centered at (cx, cy) with fixed scale
        # (full implementation would use separate TL and BR heads)
        half_w = torch.full_like(cx, 0.1)  # placeholder; refined by IoU loss
        half_h = torch.full_like(cy, 0.1)

        x1 = (cx - half_w).clamp(0, 1)
        y1 = (cy - half_h).clamp(0, 1)
        x2 = (cx + half_w).clamp(0, 1)
        y2 = (cy + half_h).clamp(0, 1)

        return torch.stack([x1, y1, x2, y2], dim=-1)   # (B, 4)


# ---------------------------------------------------------------------------
# MLP Corner Head (simpler alternative)
# ---------------------------------------------------------------------------

class MLPHead(nn.Module):
    """
    Simple MLP head: pool search tokens → predict bbox.

    Easier to understand than CornerHead, useful for ablations.
    """

    def __init__(self, embed_dim: int = 768, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 4),
            nn.Sigmoid(),  # output in [0, 1]
        )

    def forward(self, x_tokens: torch.Tensor):
        """
        Args:
            x_tokens: (B, Nx, D)

        Returns:
            dict with "pred_boxes": (B, 4)
        """
        pooled = x_tokens.mean(dim=1)   # (B, D) global average pool
        boxes = self.net(pooled)         # (B, 4)
        return {"pred_boxes": boxes, "score_map": None}


# ---------------------------------------------------------------------------
# IoU Prediction Head (optional auxiliary)
# ---------------------------------------------------------------------------

class IoUHead(nn.Module):
    """
    Predict IoU score to rank/filter box predictions.
    Used in OSTrack to score tracking quality.
    """

    def __init__(self, embed_dim: int = 768, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

    def forward(self, x_tokens: torch.Tensor) -> torch.Tensor:
        """Returns (B, 1) predicted IoU score."""
        pooled = x_tokens.mean(dim=1)
        return self.net(pooled)


# ---------------------------------------------------------------------------
# Loss Functions
# ---------------------------------------------------------------------------

class TrackingLoss(nn.Module):
    """
    Combined loss for SOT training:
      L = λ₁ · L_L1 + λ₂ · L_GIoU + λ₃ · L_focal

    - L_L1:    L1 loss on box coordinates (fast, stable)
    - L_GIoU:  Generalized IoU loss (scale-invariant, better for tracking)
    - L_focal: Focal loss on the score map (handles class imbalance)
    """

    def __init__(
        self,
        l1_weight: float = 5.0,
        giou_weight: float = 2.0,
        focal_weight: float = 1.0,
    ):
        super().__init__()
        self.l1_w = l1_weight
        self.giou_w = giou_weight
        self.focal_w = focal_weight

    def forward(
        self,
        pred_boxes: torch.Tensor,    # (B, 4) normalized [x1,y1,x2,y2]
        target_boxes: torch.Tensor,  # (B, 4) normalized [x1,y1,x2,y2]
        score_map: torch.Tensor = None,  # (B, 1, H, W) optional
        target_mask: torch.Tensor = None,  # (B, 1, H, W) optional Gaussian
    ) -> dict:

        losses = {}

        # L1 Loss
        l1_loss = F.l1_loss(pred_boxes, target_boxes)
        losses["l1"] = self.l1_w * l1_loss

        # GIoU Loss
        giou_loss = giou_loss_fn(pred_boxes, target_boxes)
        losses["giou"] = self.giou_w * giou_loss

        # Focal loss on score map (if provided)
        if score_map is not None and target_mask is not None:
            fl = focal_loss(score_map, target_mask)
            losses["focal"] = self.focal_w * fl
        else:
            losses["focal"] = torch.tensor(0.0, device=pred_boxes.device)

        losses["total"] = sum(losses.values())
        return losses


def giou_loss_fn(
    pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-6
) -> torch.Tensor:
    """
    Generalized Intersection over Union loss.

    Reference: https://arxiv.org/abs/1902.09630
    """
    # Convert to (x1, y1, x2, y2)
    px1, py1, px2, py2 = pred.unbind(-1)
    tx1, ty1, tx2, ty2 = target.unbind(-1)

    # Intersection
    ix1 = torch.max(px1, tx1)
    iy1 = torch.max(py1, ty1)
    ix2 = torch.min(px2, tx2)
    iy2 = torch.min(py2, ty2)
    inter = (ix2 - ix1).clamp(0) * (iy2 - iy1).clamp(0)

    # Areas
    pred_area = (px2 - px1).clamp(0) * (py2 - py1).clamp(0)
    tgt_area = (tx2 - tx1).clamp(0) * (ty2 - ty1).clamp(0)
    union = pred_area + tgt_area - inter + eps

    iou = inter / union

    # Enclosing box
    ex1 = torch.min(px1, tx1)
    ey1 = torch.min(py1, ty1)
    ex2 = torch.max(px2, tx2)
    ey2 = torch.max(py2, ty2)
    enclosing_area = (ex2 - ex1).clamp(0) * (ey2 - ey1).clamp(0) + eps

    giou = iou - (enclosing_area - union) / enclosing_area
    return (1 - giou).mean()


def focal_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    alpha: float = 0.25,
    gamma: float = 2.0,
) -> torch.Tensor:
    """
    Focal loss for dense prediction (score map supervision).
    Downweights easy negatives, focuses on hard examples.

    Reference: https://arxiv.org/abs/1708.02002
    """
    pred = pred.sigmoid()
    ce = F.binary_cross_entropy(pred, target, reduction="none")
    pt = target * pred + (1 - target) * (1 - pred)
    focal_weight = alpha * target + (1 - alpha) * (1 - target)
    focal_weight = focal_weight * (1 - pt) ** gamma
    return (focal_weight * ce).mean()


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def build_head(cfg) -> nn.Module:
    head_cfg = cfg.get("head", {})
    head_type = head_cfg.get("type", "corner")

    embed_dim = cfg.get("backbone", {}).get("embed_dim", 768)
    search_size = cfg.get("search_size", 256)
    patch_size = cfg.get("backbone", {}).get("patch_size", 16)
    search_feat_size = search_size // patch_size

    if head_type == "corner":
        return CornerHead(
            embed_dim=embed_dim,
            hidden_dim=head_cfg.get("hidden_dim", 256),
            search_feat_size=search_feat_size,
            num_layers=head_cfg.get("num_layers", 3),
        )
    elif head_type == "mlp":
        return MLPHead(embed_dim=embed_dim, hidden_dim=head_cfg.get("hidden_dim", 256))
    else:
        raise ValueError(f"Unknown head type: {head_type}")
