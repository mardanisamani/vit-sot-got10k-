"""
ViT Backbone for Single Object Tracking (OSTrack-style)
========================================================
In OSTrack, the template (z) and search (x) patches are concatenated
into a single token sequence and processed together through ViT layers.
This "one-stream" design allows natural cross-attention between z and x.

Key insight: Unlike two-stream trackers (SiamRPN, SiamFC) that process
template and search separately then fuse, OSTrack processes them jointly
from the very first layer — richer interaction, simpler architecture.
"""

import math
import torch
import torch.nn as nn
from functools import partial

try:
    import timm
    from timm.models.vision_transformer import VisionTransformer, Block
    HAS_TIMM = True
except ImportError:
    HAS_TIMM = False


# ---------------------------------------------------------------------------
# Positional Encoding Utils
# ---------------------------------------------------------------------------

def get_2d_sincos_pos_embed(embed_dim: int, grid_h: int, grid_w: int) -> torch.Tensor:
    """
    Generate 2D sine-cosine positional embeddings.

    Args:
        embed_dim: Dimension of the embedding.
        grid_h: Height of the grid (number of patches in height).
        grid_w: Width of the grid (number of patches in width).

    Returns:
        pos_embed: (grid_h * grid_w, embed_dim) positional embeddings.
    """
    assert embed_dim % 4 == 0, "embed_dim must be divisible by 4"
    half_dim = embed_dim // 2

    # Height encoding
    grid_h_idx = torch.arange(grid_h, dtype=torch.float32)
    grid_w_idx = torch.arange(grid_w, dtype=torch.float32)
    grid = torch.meshgrid(grid_h_idx, grid_w_idx, indexing="ij")
    grid = torch.stack(grid, dim=0)  # (2, H, W)

    omega = torch.arange(half_dim // 2, dtype=torch.float32)
    omega = 1.0 / (10000 ** (2 * omega / half_dim))

    # Outer product: positions × frequencies
    pos_h = grid[0].reshape(-1)[:, None] * omega[None, :]  # (H*W, D/4)
    pos_w = grid[1].reshape(-1)[:, None] * omega[None, :]  # (H*W, D/4)

    emb = torch.cat([
        torch.sin(pos_h), torch.cos(pos_h),
        torch.sin(pos_w), torch.cos(pos_w),
    ], dim=-1)  # (H*W, D)
    return emb


# ---------------------------------------------------------------------------
# Core ViT Backbone with Tracking Support
# ---------------------------------------------------------------------------

class VisionTransformerTrack(nn.Module):
    """
    ViT backbone adapted for one-stream tracking (OSTrack-style).

    The model concatenates [template_tokens | search_tokens] and processes
    them jointly through all transformer layers.

    Template crop:  128×128  →  8×8  = 64  patches  (patch_size=16)
    Search crop:    256×256  →  16×16 = 256 patches  (patch_size=16)

    Total sequence length = 64 + 256 = 320 tokens (+ optional CLS tokens)
    """

    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        in_chans: int = 3,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.1,
        norm_layer=None,
        # Tracking-specific
        template_size: int = 128,
        search_size: int = 256,
    ):
        super().__init__()
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)

        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.template_size = template_size
        self.search_size = search_size

        # Number of patches for each input
        self.num_z_patches = (template_size // patch_size) ** 2   # e.g. 64
        self.num_x_patches = (search_size // patch_size) ** 2      # e.g. 256

        # ── Patch Embedding ──────────────────────────────────────────────────
        self.patch_embed = nn.Conv2d(
            in_chans, embed_dim,
            kernel_size=patch_size, stride=patch_size
        )

        # CLS tokens (optional, not used in head but kept for compatibility)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        # ── Positional Embeddings ─────────────────────────────────────────────
        # Separate learnable pos-embeds for template and search
        self.pos_embed_z = nn.Parameter(
            torch.zeros(1, self.num_z_patches, embed_dim)
        )
        self.pos_embed_x = nn.Parameter(
            torch.zeros(1, self.num_x_patches, embed_dim)
        )

        # ── Transformer Blocks ────────────────────────────────────────────────
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        if HAS_TIMM:
            self.blocks = nn.Sequential(*[
                Block(
                    dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias, drop=drop_rate, attn_drop=attn_drop_rate,
                    drop_path=dpr[i], norm_layer=norm_layer
                )
                for i in range(depth)
            ])
        else:
            self.blocks = nn.Sequential(*[
                TransformerBlock(
                    dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias, drop=drop_rate, attn_drop=attn_drop_rate,
                    drop_path=dpr[i], norm_layer=norm_layer
                )
                for i in range(depth)
            ])

        self.norm = norm_layer(embed_dim)

        # Weight initialization
        self._init_weights()

    def _init_weights(self):
        """Initialize weights following ViT paper (He et al., 2021)."""
        # Initialize patch embed like a conv layer
        nn.init.normal_(self.cls_token, std=0.02)

        # Sincos pos embed as initialization
        z_h = z_w = self.template_size // self.patch_size
        x_h = x_w = self.search_size // self.patch_size

        pos_z = get_2d_sincos_pos_embed(self.embed_dim, z_h, z_w)
        pos_x = get_2d_sincos_pos_embed(self.embed_dim, x_h, x_w)

        with torch.no_grad():
            self.pos_embed_z.copy_(pos_z.unsqueeze(0))
            self.pos_embed_x.copy_(pos_x.unsqueeze(0))

    def patch_embed_image(self, x: torch.Tensor) -> torch.Tensor:
        """Convert image to patch embeddings. (B, C, H, W) → (B, N, D)"""
        x = self.patch_embed(x)          # (B, D, H/P, W/P)
        x = x.flatten(2).transpose(1, 2)  # (B, N, D)
        return x

    def forward(
        self,
        template: torch.Tensor,   # (B, 3, Hz, Wz)  e.g. (B, 3, 128, 128)
        search: torch.Tensor,      # (B, 3, Hx, Wx)  e.g. (B, 3, 256, 256)
    ):
        """
        Forward pass through the one-stream ViT tracker.

        Args:
            template: Template image crop (first frame, ground truth bbox).
            search: Search image crop (current frame to localize object).

        Returns:
            z_tokens: Template tokens after transformer (B, Nz, D)
            x_tokens: Search tokens after transformer (B, Nx, D)
        """
        B = template.shape[0]

        # ── Embed patches ────────────────────────────────────────────────────
        z_tokens = self.patch_embed_image(template)   # (B, Nz, D)
        x_tokens = self.patch_embed_image(search)     # (B, Nx, D)

        # ── Add positional embeddings ─────────────────────────────────────────
        z_tokens = z_tokens + self.pos_embed_z
        x_tokens = x_tokens + self.pos_embed_x

        # ── Concatenate: [template | search] ─────────────────────────────────
        # This is the key design choice of one-stream trackers:
        # template tokens can attend to search tokens (and vice versa)
        # right from the first layer.
        tokens = torch.cat([z_tokens, x_tokens], dim=1)  # (B, Nz+Nx, D)

        # ── Transformer ───────────────────────────────────────────────────────
        tokens = self.blocks(tokens)   # (B, Nz+Nx, D)
        tokens = self.norm(tokens)

        # ── Split back into template and search tokens ────────────────────────
        z_out = tokens[:, :self.num_z_patches]   # (B, Nz, D)
        x_out = tokens[:, self.num_z_patches:]   # (B, Nx, D)

        return z_out, x_out

    @classmethod
    def from_pretrained(cls, model_name: str = "vit_base_patch16_224", **kwargs):
        """
        Load a pretrained ViT backbone from timm and adapt it for tracking.

        The ImageNet-pretrained patch embeddings are directly reusable —
        only the positional embeddings need to be adapted (different crop sizes).

        Args:
            model_name: timm model name.
            **kwargs: Extra arguments passed to __init__.

        Returns:
            model: VisionTransformerTrack instance with pretrained weights.
        """
        if not HAS_TIMM:
            raise ImportError("timm is required. Run: pip install timm")

        # Build model with our tracking config
        model = cls(**kwargs)

        # Load pretrained ViT
        pretrained = timm.create_model(model_name, pretrained=True)

        # Transfer weights (patch_embed and transformer blocks)
        state_dict = pretrained.state_dict()
        model_dict = model.state_dict()

        # Map timm keys → our keys
        transfer = {}
        for k, v in state_dict.items():
            if k.startswith("patch_embed.proj"):
                new_k = k.replace("patch_embed.proj", "patch_embed")
                if new_k in model_dict and model_dict[new_k].shape == v.shape:
                    transfer[new_k] = v
            elif k.startswith("blocks"):
                if k in model_dict and model_dict[k].shape == v.shape:
                    transfer[k] = v
            elif k in ("norm.weight", "norm.bias"):
                if k in model_dict:
                    transfer[k] = v

        model.load_state_dict(transfer, strict=False)
        print(f"[VisionTransformerTrack] Loaded {len(transfer)}/{len(model_dict)} keys from {model_name}")
        return model


# ---------------------------------------------------------------------------
# Fallback: Minimal Transformer Block (no timm dependency)
# ---------------------------------------------------------------------------

class TransformerBlock(nn.Module):
    """Minimal transformer block (used when timm is unavailable)."""

    def __init__(
        self, dim, num_heads, mlp_ratio=4.0, qkv_bias=True,
        drop=0.0, attn_drop=0.0, drop_path=0.0, norm_layer=nn.LayerNorm
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = nn.MultiheadAttention(
            dim, num_heads, dropout=attn_drop, bias=qkv_bias, batch_first=True
        )
        self.norm2 = norm_layer(dim)
        mlp_hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(mlp_hidden, dim),
            nn.Dropout(drop),
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x):
        normed = self.norm1(x)
        attn_out, _ = self.attn(normed, normed, normed)
        x = x + self.drop_path(attn_out)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class DropPath(nn.Module):
    """Stochastic depth (drop path) regularization."""

    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if not self.training or self.drop_prob == 0.0:
            return x
        keep = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask = torch.bernoulli(torch.full(shape, keep, device=x.device)) / keep
        return x * mask


# ---------------------------------------------------------------------------
# Factory helpers
# ---------------------------------------------------------------------------

def build_vit_backbone(cfg) -> VisionTransformerTrack:
    """Build ViT backbone from config dict."""
    backbone_cfg = cfg.get("backbone", {})
    model_name = backbone_cfg.get("pretrained_model", "vit_base_patch16_224")
    pretrained = backbone_cfg.get("pretrained", True)

    kwargs = dict(
        img_size=backbone_cfg.get("img_size", 224),
        patch_size=backbone_cfg.get("patch_size", 16),
        embed_dim=backbone_cfg.get("embed_dim", 768),
        depth=backbone_cfg.get("depth", 12),
        num_heads=backbone_cfg.get("num_heads", 12),
        mlp_ratio=backbone_cfg.get("mlp_ratio", 4.0),
        drop_path_rate=backbone_cfg.get("drop_path_rate", 0.1),
        template_size=cfg.get("template_size", 128),
        search_size=cfg.get("search_size", 256),
    )

    if pretrained and HAS_TIMM:
        return VisionTransformerTrack.from_pretrained(model_name, **kwargs)
    return VisionTransformerTrack(**kwargs)
