"""
OSTrack Demo — Track Objects in Your Own Video
================================================
Quick demo: run tracking on a video file or webcam.

Usage:
    # Track in a video file (annotate with bbox in first frame)
    python demo.py --checkpoint checkpoints/best.pth --video my_video.mp4

    # Track from webcam
    python demo.py --checkpoint checkpoints/best.pth --webcam

    # Track with a provided initial bbox
    python demo.py --checkpoint checkpoints/best.pth --video video.mp4 \
                   --init_box 100,150,300,400   # x1,y1,x2,y2

    # Visualize attention maps during tracking
    python demo.py --checkpoint checkpoints/best.pth --video video.mp4 \
                   --show_attention
"""

import os
import sys
import argparse
import yaml
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))

from lib.models import build_ostrack
from lib.tracking import OSTracker


def parse_args():
    parser = argparse.ArgumentParser(description="OSTrack Demo")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--video", type=str, default=None,
                        help="Path to input video file")
    parser.add_argument("--webcam", action="store_true",
                        help="Use webcam as input")
    parser.add_argument("--init_box", type=str, default=None,
                        help="Initial bbox: x1,y1,x2,y2")
    parser.add_argument("--output", type=str, default="demo_output.mp4",
                        help="Output video path")
    parser.add_argument("--device", type=str, default="cuda")
    return parser.parse_args()


def select_bbox_interactively(frame: np.ndarray) -> np.ndarray:
    """Let user draw a bounding box using OpenCV."""
    import cv2
    print("Draw a bounding box around the object to track.")
    print("Press ENTER or SPACE to confirm, press C to cancel.")
    box = cv2.selectROI("Select Object", frame, fromCenter=False, showCrosshair=True)
    cv2.destroyWindow("Select Object")
    if box[2] == 0 or box[3] == 0:
        raise ValueError("No valid bounding box selected.")
    x, y, w, h = box
    return np.array([x, y, x + w, y + h], dtype=np.float32)


def main():
    import cv2

    args = parse_args()

    # Load config + checkpoint
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    cfg = ckpt.get("cfg", {
        "backbone": {"patch_size": 16, "embed_dim": 384, "depth": 12,
                     "num_heads": 6, "pretrained": False},
        "head": {"type": "corner", "hidden_dim": 128},
        "template_size": 128, "search_size": 256,
    })
    cfg["backbone"]["pretrained"] = False

    # Build tracker
    model = build_ostrack(cfg)
    model.load_state_dict(ckpt["model"], strict=False)
    tracker = OSTracker(model, device=args.device)

    # Open video source
    if args.webcam:
        cap = cv2.VideoCapture(0)
    else:
        cap = cv2.VideoCapture(args.video)

    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video source.")

    # Read first frame
    ret, frame = cap.read()
    if not ret:
        raise RuntimeError("Cannot read first frame.")
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Get initial bbox
    if args.init_box:
        init_box = np.array([float(v) for v in args.init_box.split(",")],
                            dtype=np.float32)
    else:
        init_box = select_bbox_interactively(frame)

    # Initialize tracker
    tracker.init(Image.fromarray(frame_rgb), init_box)

    # Setup video writer
    h, w = frame.shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(args.output, fourcc, 30, (w, h))

    # Draw first frame
    x1, y1, x2, y2 = [int(v) for v in init_box]
    frame_out = frame.copy()
    cv2.rectangle(frame_out, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(frame_out, "INIT", (x1, y1-8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
    writer.write(frame_out)
    cv2.imshow("OSTrack Demo", frame_out)

    # Track
    frame_idx = 1
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pred_box = tracker.track(Image.fromarray(frame_rgb))

        # Draw prediction
        px1, py1, px2, py2 = [int(v) for v in pred_box]
        frame_out = frame.copy()
        cv2.rectangle(frame_out, (px1, py1), (px2, py2), (0, 255, 0), 2)
        cv2.putText(frame_out, f"Frame {frame_idx}", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        writer.write(frame_out)
        cv2.imshow("OSTrack Demo", frame_out)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q") or key == 27:
            break

        frame_idx += 1

    cap.release()
    writer.release()
    cv2.destroyAllWindows()
    print(f"\nTracking complete. Output saved to: {args.output}")


if __name__ == "__main__":
    main()
