"""
OSTrack Evaluation Script — GOT-10k
======================================
Run tracking on GOT-10k test/val sequences and compute AO, SR50, SR75.

Usage:
    # Evaluate on GOT-10k val set
    python evaluate.py --checkpoint checkpoints/best.pth --split val

    # Evaluate on GOT-10k test (generates submission files)
    python evaluate.py --checkpoint checkpoints/best.pth --split test --save_results

    # Evaluate a single sequence
    python evaluate.py --checkpoint checkpoints/best.pth --seq GOT-10k_Val_000001

    # Generate visualization videos
    python evaluate.py --checkpoint checkpoints/best.pth --split val --visualize
"""

import os
import sys
import yaml
import argparse
import time
import logging
from pathlib import Path
from typing import List

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))

from lib.models import build_ostrack
from lib.tracking import OSTracker
from lib.datasets.got10k import GOT10kSequence, xywh_to_xyxy
from lib.utils.metrics import evaluate_sequence, summarize_results

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Argument Parsing
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate OSTrack on GOT-10k")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to model checkpoint (.pth)")
    parser.add_argument("--config", type=str, default=None,
                        help="Config YAML (auto-loaded from checkpoint if not given)")
    parser.add_argument("--data_root", type=str, default="data/got10k",
                        help="GOT-10k dataset root directory")
    parser.add_argument("--split", type=str, default="val",
                        choices=["train", "val", "test"],
                        help="Dataset split to evaluate on")
    parser.add_argument("--seq", type=str, default=None,
                        help="Evaluate a single sequence by name")
    parser.add_argument("--save_results", action="store_true",
                        help="Save per-frame predictions to disk (for GOT-10k submission)")
    parser.add_argument("--visualize", action="store_true",
                        help="Generate tracking visualization videos")
    parser.add_argument("--output_dir", type=str, default="output/eval",
                        help="Output directory for results and visualizations")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device: 'cuda' or 'cpu'")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Evaluation on a single sequence
# ---------------------------------------------------------------------------

def evaluate_single_sequence(
    tracker: OSTracker,
    seq: GOT10kSequence,
    save_path: str = None,
    visualize: bool = False,
) -> dict:
    """
    Run tracker on a single sequence and compute metrics.

    Returns dict with AO, SR50, SR75, Precision@20.
    """
    n_frames = len(seq)

    # Initialize with first frame
    init_frame = seq.get_frame(0)
    init_box = seq.get_box_xyxy(0)
    tracker.init(init_frame, init_box)

    pred_boxes = [init_box]
    times = [0.0]

    # Track remaining frames
    for i in range(1, n_frames):
        frame = seq.get_frame(i)
        t0 = time.time()
        pred = tracker.track(frame)
        times.append(time.time() - t0)
        pred_boxes.append(pred)

    # Ground truth boxes
    gt_boxes = [seq.get_box_xyxy(i) for i in range(n_frames)]

    # Compute metrics
    metrics = evaluate_sequence(pred_boxes, gt_boxes)
    fps = 1.0 / (np.mean(times[1:]) + 1e-9) if len(times) > 1 else 0

    metrics["fps"] = fps
    metrics["n_frames"] = n_frames
    metrics["seq_name"] = seq.name

    # Save predictions
    if save_path:
        os.makedirs(save_path, exist_ok=True)
        pred_file = os.path.join(save_path, f"{seq.name}.txt")
        with open(pred_file, "w") as f:
            for box in pred_boxes:
                # GOT-10k format: x, y, w, h
                x1, y1, x2, y2 = box
                f.write(f"{x1:.4f},{y1:.4f},{x2-x1:.4f},{y2-y1:.4f}\n")

    # Visualize
    if visualize:
        _visualize_sequence(seq, pred_boxes, gt_boxes, save_path)

    return metrics


def _visualize_sequence(
    seq: GOT10kSequence,
    pred_boxes: list,
    gt_boxes: list,
    save_dir: str,
    max_frames: int = 100,
):
    """Save tracking visualization as PNG frames."""
    try:
        import cv2
    except ImportError:
        logger.warning("OpenCV not available, skipping visualization.")
        return

    vis_dir = os.path.join(save_dir, seq.name, "vis")
    os.makedirs(vis_dir, exist_ok=True)

    for i, (frame_path, pred, gt) in enumerate(
        zip(seq.frames[:max_frames], pred_boxes[:max_frames], gt_boxes[:max_frames])
    ):
        frame = cv2.imread(frame_path)
        if frame is None:
            continue

        # Draw GT (red) and prediction (green)
        def draw_box_cv(img, box, color, label=""):
            x1, y1, x2, y2 = [int(v) for v in box]
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            if label:
                cv2.putText(img, label, (x1, y1 - 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            return img

        frame = draw_box_cv(frame, gt, (0, 0, 255), "GT")
        frame = draw_box_cv(frame, pred, (0, 255, 0), "Pred")
        cv2.putText(frame, f"Frame {i}", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        cv2.imwrite(os.path.join(vis_dir, f"{i:05d}.png"), frame)

    logger.info(f"Saved visualization to {vis_dir}")


# ---------------------------------------------------------------------------
# Main evaluation loop
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    # Load config from checkpoint if not provided
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    cfg = args.config
    if cfg is None:
        if "cfg" in ckpt:
            cfg = ckpt["cfg"]
        else:
            # Default config
            cfg = {
                "backbone": {"patch_size": 16, "embed_dim": 384, "depth": 12,
                             "num_heads": 6, "pretrained": False},
                "head": {"type": "corner", "hidden_dim": 128},
                "template_size": 128,
                "search_size": 256,
            }
    elif isinstance(cfg, str):
        with open(cfg, "r") as f:
            cfg = yaml.safe_load(f)

    # Build model
    logger.info("Building model...")
    cfg["backbone"]["pretrained"] = False  # Load from checkpoint
    model = build_ostrack(cfg)
    model.load_state_dict(ckpt["model"], strict=False)
    logger.info(f"Loaded weights from {args.checkpoint} (epoch {ckpt.get('epoch', '?')})")

    # Build tracker
    tracker = OSTracker(
        model=model,
        device=args.device,
        template_size=cfg.get("template_size", 128),
        search_size=cfg.get("search_size", 256),
    )

    # Find sequences
    data_root = Path(args.data_root)
    split_dir = data_root / args.split

    if args.seq:
        seq_dirs = [split_dir / args.seq]
    else:
        seq_dirs = sorted(
            d for d in split_dir.iterdir()
            if d.is_dir() and (d / "groundtruth.txt").exists()
        )

    logger.info(f"Evaluating {len(seq_dirs)} sequences on {args.split} split...")

    # Evaluate
    all_results = []
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for seq_dir in seq_dirs:
        try:
            seq = GOT10kSequence(str(seq_dir))
        except Exception as e:
            logger.warning(f"Skipping {seq_dir.name}: {e}")
            continue

        save_path = str(output_dir / "predictions") if args.save_results else None

        result = evaluate_single_sequence(
            tracker, seq,
            save_path=save_path,
            visualize=args.visualize,
        )
        all_results.append(result)

        logger.info(
            f"{seq.name:40s} | "
            f"AO: {result['AO']:.3f} | "
            f"SR50: {result['SR50']:.3f} | "
            f"SR75: {result['SR75']:.3f} | "
            f"FPS: {result['fps']:.1f}"
        )

    # Summarize
    if all_results:
        summary = summarize_results(all_results)
        print("\n" + "=" * 60)
        print(f"  GOT-10k {args.split.upper()} Evaluation Results")
        print("=" * 60)
        print(f"  Sequences evaluated: {len(all_results)}")
        print(f"  AO   (Average Overlap):          {summary['AO']:.4f}")
        print(f"  SR50 (Success Rate @ IoU≥0.5):   {summary['SR50']:.4f}")
        print(f"  SR75 (Success Rate @ IoU≥0.75):  {summary['SR75']:.4f}")
        print(f"  AUC  (Area Under Curve):          {summary['AUC']:.4f}")
        print(f"  Prec (Center Error ≤ 20px):       {summary['Precision@20']:.4f}")
        print(f"  FPS  (Average):                   {summary['fps']:.1f}")
        print("=" * 60)

        # Save summary
        result_file = output_dir / f"results_{args.split}.yaml"
        import yaml
        with open(result_file, "w") as f:
            yaml.dump({"summary": summary, "per_sequence": all_results}, f)
        logger.info(f"Results saved to {result_file}")


if __name__ == "__main__":
    main()
