"""
OSTrack Training Script — GOT-10k
===================================
Training pipeline for OSTrack on GOT-10k.

Usage:
    # Train with default config (ViT-Small for faster experiments)
    python train.py --config configs/ostrack_small.yaml

    # Train with ViT-Base (better performance, needs more GPU memory)
    python train.py --config configs/ostrack_base.yaml

    # Resume from checkpoint
    python train.py --config configs/ostrack_small.yaml --resume checkpoints/epoch_10.pth

    # Multi-GPU (DataParallel)
    python train.py --config configs/ostrack_base.yaml --gpus 0,1,2,3

    # Quick sanity check (1 epoch, small batch)
    python train.py --config configs/ostrack_small.yaml --debug

Recommended hardware: 1× A100 (80GB) or 4× V100 (32GB) for ViT-Base
                      1× RTX 3090/4090 for ViT-Small
"""

import os
import sys
import time
import yaml
import argparse
import logging
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from lib.models import build_ostrack
from lib.datasets import build_got10k_loader
from lib.models.head import TrackingLoss

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
    ]
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Argument Parsing
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Train OSTrack on GOT-10k")
    parser.add_argument("--config", type=str, default="configs/ostrack_small.yaml",
                        help="Path to YAML config file")
    parser.add_argument("--resume", type=str, default=None,
                        help="Checkpoint path to resume training")
    parser.add_argument("--gpus", type=str, default="0",
                        help="Comma-separated GPU IDs to use (e.g. '0,1,2,3')")
    parser.add_argument("--debug", action="store_true",
                        help="Debug mode: 1 epoch, reduced dataset")
    parser.add_argument("--output_dir", type=str, default="output",
                        help="Directory to save checkpoints and logs")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Training Loop
# ---------------------------------------------------------------------------

def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: optim.Optimizer,
    scaler: GradScaler,
    epoch: int,
    cfg: dict,
    device: torch.device,
    debug: bool = False,
) -> dict:
    """Train for one epoch, return loss statistics."""
    model.train()
    criterion = TrackingLoss(
        l1_weight=cfg.get("loss", {}).get("l1_weight", 5.0),
        giou_weight=cfg.get("loss", {}).get("giou_weight", 2.0),
        focal_weight=cfg.get("loss", {}).get("focal_weight", 1.0),
    )

    stats = {"total": 0.0, "l1": 0.0, "giou": 0.0, "focal": 0.0}
    n_batches = 0

    for batch_idx, batch in enumerate(loader):
        if debug and batch_idx >= 10:
            break

        template = batch["template"].to(device, non_blocking=True)
        search = batch["search"].to(device, non_blocking=True)
        gt_boxes = batch["gt_boxes"].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        # Mixed precision forward pass
        with autocast(enabled=cfg.get("use_amp", True)):
            # Get predictions from model
            z_tokens, x_tokens = model.backbone(template, search)
            predictions = model.head(x_tokens)

            losses = criterion(
                predictions["pred_boxes"],
                gt_boxes,
                predictions.get("score_map"),
            )

        # Backward
        scaler.scale(losses["total"]).backward()

        # Gradient clipping (important for transformer training)
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.1)

        scaler.step(optimizer)
        scaler.update()

        # Accumulate stats
        for k in stats:
            if k in losses:
                stats[k] += losses[k].item()
        n_batches += 1

        if batch_idx % cfg.get("log_interval", 50) == 0:
            logger.info(
                f"Epoch {epoch} [{batch_idx}/{len(loader)}] "
                f"Loss: {losses['total'].item():.4f} "
                f"(L1: {losses['l1'].item():.4f}, "
                f"GIoU: {losses['giou'].item():.4f})"
            )

    return {k: v / max(n_batches, 1) for k, v in stats.items()}


@torch.no_grad()
def validate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    debug: bool = False,
) -> dict:
    """Validate model, return average losses."""
    model.eval()
    criterion = TrackingLoss()

    stats = {"total": 0.0, "l1": 0.0, "giou": 0.0}
    n_batches = 0

    for batch_idx, batch in enumerate(loader):
        if debug and batch_idx >= 5:
            break

        template = batch["template"].to(device)
        search = batch["search"].to(device)
        gt_boxes = batch["gt_boxes"].to(device)

        z_tokens, x_tokens = model.backbone(template, search)
        predictions = model.head(x_tokens)
        losses = criterion(predictions["pred_boxes"], gt_boxes)

        for k in stats:
            if k in losses:
                stats[k] += losses[k].item()
        n_batches += 1

    return {k: v / max(n_batches, 1) for k, v in stats.items()}


# ---------------------------------------------------------------------------
# Optimizer and Scheduler
# ---------------------------------------------------------------------------

def build_optimizer(model: nn.Module, cfg: dict) -> optim.Optimizer:
    """
    Build AdamW optimizer with layer-wise learning rate decay.

    Deeper ViT layers receive smaller learning rates (LLR decay).
    This is crucial for fine-tuning large pretrained transformers.

    Typical settings:
      - Backbone LR: 4e-5 (fine-tuned slowly from ImageNet weights)
      - Head LR: 4e-4 (trained faster, random init)
      - LLR decay: 0.75 (multiply LR by 0.75 per ViT layer from top)
    """
    opt_cfg = cfg.get("optimizer", {})
    lr = opt_cfg.get("lr", 4e-4)
    backbone_lr_factor = opt_cfg.get("backbone_lr_factor", 0.1)
    weight_decay = opt_cfg.get("weight_decay", 1e-4)

    # Separate backbone and head parameters
    backbone_params = list(model.backbone.named_parameters())
    head_params = list(model.head.named_parameters())

    param_groups = [
        {
            "params": [p for _, p in backbone_params if p.requires_grad],
            "lr": lr * backbone_lr_factor,
            "name": "backbone",
        },
        {
            "params": [p for _, p in head_params if p.requires_grad],
            "lr": lr,
            "name": "head",
        },
    ]

    optimizer = optim.AdamW(
        param_groups,
        weight_decay=weight_decay,
        betas=(0.9, 0.999),
    )
    return optimizer


def build_scheduler(optimizer: optim.Optimizer, cfg: dict, num_epochs: int):
    """Cosine annealing with linear warmup."""
    sched_cfg = cfg.get("scheduler", {})
    warmup_epochs = sched_cfg.get("warmup_epochs", 2)

    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return epoch / warmup_epochs
        # Cosine decay after warmup
        progress = (epoch - warmup_epochs) / max(1, num_epochs - warmup_epochs)
        return 0.5 * (1 + math.cos(math.pi * progress))

    import math
    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# ---------------------------------------------------------------------------
# Checkpoint Utilities
# ---------------------------------------------------------------------------

def save_checkpoint(
    model: nn.Module,
    optimizer: optim.Optimizer,
    scheduler,
    epoch: int,
    val_loss: float,
    output_dir: str,
    cfg: dict,
    best: bool = False,
):
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    state = {
        "epoch": epoch,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "val_loss": val_loss,
        "cfg": cfg,
    }
    fname = "best.pth" if best else f"epoch_{epoch:03d}.pth"
    path = os.path.join(output_dir, fname)
    torch.save(state, path)
    logger.info(f"Saved checkpoint: {path}")


def load_checkpoint(
    model: nn.Module,
    optimizer: optim.Optimizer,
    scheduler,
    resume_path: str,
    device: torch.device,
) -> int:
    """Load checkpoint, return starting epoch."""
    state = torch.load(resume_path, map_location=device)
    model.load_state_dict(state["model"])
    optimizer.load_state_dict(state["optimizer"])
    scheduler.load_state_dict(state["scheduler"])
    logger.info(f"Resumed from {resume_path} (epoch {state['epoch']})")
    return state["epoch"] + 1


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    # Load config
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    if args.debug:
        cfg["training"]["num_epochs"] = 1
        cfg["data"]["batch_size"] = 4
        cfg["data"]["num_workers"] = 0
        logger.info("Debug mode: 1 epoch, batch_size=4")

    # Setup devices
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Setup output directory
    output_dir = os.path.join(args.output_dir, Path(args.config).stem)
    os.makedirs(output_dir, exist_ok=True)

    # Log file
    logging.getLogger().addHandler(
        logging.FileHandler(os.path.join(output_dir, "train.log"))
    )

    # Build model
    logger.info("Building model...")
    model = build_ostrack(cfg)

    # Multi-GPU
    if torch.cuda.device_count() > 1 and "," in args.gpus:
        logger.info(f"Using {torch.cuda.device_count()} GPUs")
        model = nn.DataParallel(model)

    model = model.to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Trainable parameters: {n_params:,}")

    # Build data loaders
    logger.info("Loading GOT-10k dataset...")
    train_loader, val_loader = build_got10k_loader(cfg)

    # Optimizer & Scheduler
    num_epochs = cfg.get("training", {}).get("num_epochs", 300)
    base_model = model.module if hasattr(model, "module") else model
    optimizer = build_optimizer(base_model, cfg)
    scheduler = build_scheduler(optimizer, cfg, num_epochs)

    # AMP Scaler
    scaler = GradScaler(enabled=cfg.get("use_amp", True) and device.type == "cuda")

    # Resume
    start_epoch = 0
    if args.resume:
        start_epoch = load_checkpoint(model, optimizer, scheduler, args.resume, device)

    best_val_loss = float("inf")

    # Training loop
    logger.info(f"Starting training for {num_epochs} epochs...")
    for epoch in range(start_epoch, num_epochs):
        t0 = time.time()

        # Train
        train_stats = train_one_epoch(
            model, train_loader, optimizer, scaler, epoch, cfg, device, args.debug
        )

        # Validate
        val_stats = validate(model, val_loader, device, args.debug)

        # Step scheduler
        scheduler.step()

        epoch_time = time.time() - t0
        logger.info(
            f"Epoch {epoch}/{num_epochs} | "
            f"Train Loss: {train_stats['total']:.4f} | "
            f"Val Loss: {val_stats['total']:.4f} | "
            f"LR: {optimizer.param_groups[0]['lr']:.6f} | "
            f"Time: {epoch_time:.1f}s"
        )

        # Save checkpoint
        is_best = val_stats["total"] < best_val_loss
        if is_best:
            best_val_loss = val_stats["total"]
            save_checkpoint(
                base_model, optimizer, scheduler, epoch,
                val_stats["total"], output_dir, cfg, best=True
            )

        if (epoch + 1) % cfg.get("training", {}).get("save_interval", 10) == 0:
            save_checkpoint(
                base_model, optimizer, scheduler, epoch,
                val_stats["total"], output_dir, cfg
            )

    logger.info(f"Training complete. Best val loss: {best_val_loss:.4f}")
    logger.info(f"Checkpoints saved to: {output_dir}")


if __name__ == "__main__":
    main()
