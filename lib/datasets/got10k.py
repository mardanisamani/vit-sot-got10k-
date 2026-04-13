"""
GOT-10k Dataset for Single Object Tracking
============================================
GOT-10k is a large-scale high-diversity tracking benchmark:
  - 10,000 training sequences, 180 test sequences, 180 validation sequences
  - 563 object classes (over 87 motion patterns)
  - Official train/test protocol: train ONLY on GOT-10k train split
  - Download: http://got-10k.aitestunion.com/

Dataset structure expected on disk:
    got10k/
    ├── train/
    │   ├── GOT-10k_Train_000001/
    │   │   ├── 00000001.jpg
    │   │   ├── 00000002.jpg
    │   │   ├── ...
    │   │   ├── groundtruth.txt      # (x, y, w, h) per line
    │   │   └── meta_info.ini
    │   └── ...
    ├── val/
    └── test/

Bounding box format in GOT-10k: [x, y, w, h] (XYWH, top-left origin)
We convert to [x1, y1, x2, y2] (XYXY) internally.
"""

import os
import csv
import math
import random
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset

from .transforms import TrackingTransforms


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def xywh_to_xyxy(box: np.ndarray) -> np.ndarray:
    """Convert [x, y, w, h] → [x1, y1, x2, y2]."""
    x, y, w, h = box
    return np.array([x, y, x + w, y + h], dtype=np.float32)


def xyxy_to_xywh(box: np.ndarray) -> np.ndarray:
    """Convert [x1, y1, x2, y2] → [x, y, w, h]."""
    x1, y1, x2, y2 = box
    return np.array([x1, y1, x2 - x1, y2 - y1], dtype=np.float32)


def crop_and_resize(
    image: Image.Image,
    bbox: np.ndarray,        # [x1, y1, x2, y2]
    crop_size: int,
    context_factor: float = 2.0,
) -> Tuple[Image.Image, np.ndarray]:
    """
    Crop a search/template region around a bounding box with context padding.

    The crop is a square region centered on the bbox with size:
        s = sqrt((w + context_w/2) * (h + context_h/2)) * context_factor

    This is the standard crop strategy from SiamFC / OSTrack.

    Args:
        image:         PIL Image.
        bbox:          [x1, y1, x2, y2] bounding box.
        crop_size:     Target size of the square output crop.
        context_factor: How much context to include around the bbox.

    Returns:
        crop:      Cropped and resized PIL Image.
        new_bbox:  Bbox coordinates relative to the new crop, normalized [0,1].
    """
    im_w, im_h = image.size
    x1, y1, x2, y2 = bbox

    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    obj_w, obj_h = x2 - x1, y2 - y1

    # Context area (SiamFC formula)
    context = (obj_w + obj_h) / 2 * context_factor
    crop_sz = math.sqrt((obj_w + context) * (obj_h + context))

    # Crop boundaries (may extend outside image → pad)
    c_x1 = cx - crop_sz / 2
    c_y1 = cy - crop_sz / 2
    c_x2 = cx + crop_sz / 2
    c_y2 = cy + crop_sz / 2

    # Pad if needed
    pad_left = max(0, -c_x1)
    pad_top = max(0, -c_y1)
    pad_right = max(0, c_x2 - im_w)
    pad_bottom = max(0, c_y2 - im_h)

    c_x1_clipped = max(0, c_x1)
    c_y1_clipped = max(0, c_y1)
    c_x2_clipped = min(im_w, c_x2)
    c_y2_clipped = min(im_h, c_y2)

    crop = image.crop((
        int(c_x1_clipped), int(c_y1_clipped),
        int(c_x2_clipped), int(c_y2_clipped)
    ))

    # Pad with mean color
    if any(p > 0 for p in [pad_left, pad_top, pad_right, pad_bottom]):
        mean_color = tuple(int(v) for v in np.array(image).mean(axis=(0, 1))[:3])
        padded = Image.new("RGB", (
            int(crop.width + pad_left + pad_right),
            int(crop.height + pad_top + pad_bottom)
        ), mean_color)
        padded.paste(crop, (int(pad_left), int(pad_top)))
        crop = padded

    # Resize to target size
    crop = crop.resize((crop_size, crop_size), Image.BILINEAR)
    scale = crop_size / crop_sz

    # Transform bbox into crop coordinates, normalize to [0,1]
    new_x1 = (x1 - c_x1) * scale / crop_size
    new_y1 = (y1 - c_y1) * scale / crop_size
    new_x2 = (x2 - c_x1) * scale / crop_size
    new_y2 = (y2 - c_y1) * scale / crop_size

    new_bbox = np.clip(
        np.array([new_x1, new_y1, new_x2, new_y2], dtype=np.float32),
        0.0, 1.0
    )

    return crop, new_bbox


# ---------------------------------------------------------------------------
# Sequence Reader
# ---------------------------------------------------------------------------

class GOT10kSequence:
    """Reads a single GOT-10k sequence from disk."""

    def __init__(self, seq_dir: str):
        self.seq_dir = Path(seq_dir)
        self.name = self.seq_dir.name

        # Load groundtruth
        gt_file = self.seq_dir / "groundtruth.txt"
        self.boxes = self._load_gt(gt_file)  # list of np.array [x,y,w,h]

        # Collect frames (sorted)
        self.frames = sorted(
            str(p) for p in self.seq_dir.glob("*.jpg")
        )
        assert len(self.frames) == len(self.boxes), (
            f"Frame/GT mismatch in {self.name}: "
            f"{len(self.frames)} frames, {len(self.boxes)} boxes"
        )

    def _load_gt(self, gt_file: Path) -> List[np.ndarray]:
        boxes = []
        with open(gt_file, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                vals = [float(v) for v in line.replace("\t", ",").split(",")]
                boxes.append(np.array(vals[:4], dtype=np.float32))
        return boxes

    def __len__(self):
        return len(self.frames)

    def get_frame(self, idx: int) -> Image.Image:
        return Image.open(self.frames[idx]).convert("RGB")

    def get_box_xyxy(self, idx: int) -> np.ndarray:
        """Get box at index in [x1, y1, x2, y2] format."""
        return xywh_to_xyxy(self.boxes[idx])


# ---------------------------------------------------------------------------
# Training Dataset
# ---------------------------------------------------------------------------

class GOT10kDataset(Dataset):
    """
    GOT-10k training dataset.

    Each sample is a (template, search, gt_box) triplet:
      - template:  crop from frame t_z (typically frame 1)
      - search:    crop from frame t_x (sampled within [1, 100] of t_z)
      - gt_box:    ground-truth box in the search crop, normalized [0,1]

    The gap between template and search frames introduces temporal variation
    for robustness, but shouldn't be too large to keep relevance.
    """

    def __init__(
        self,
        root: str,
        split: str = "train",
        template_size: int = 128,
        search_size: int = 256,
        max_frame_gap: int = 100,
        transforms=None,
    ):
        """
        Args:
            root:          Path to GOT-10k root directory.
            split:         "train", "val", or "test".
            template_size: Size of the template crop.
            search_size:   Size of the search crop.
            max_frame_gap: Maximum frame index gap between template and search.
            transforms:    Optional data augmentation transforms.
        """
        self.root = Path(root)
        self.split = split
        self.template_size = template_size
        self.search_size = search_size
        self.max_frame_gap = max_frame_gap
        self.transforms = transforms or TrackingTransforms(split=split)

        self.sequences = self._load_sequences()
        print(f"[GOT10kDataset] Loaded {len(self.sequences)} sequences "
              f"from {split} split.")

    def _load_sequences(self) -> List[GOT10kSequence]:
        split_dir = self.root / self.split
        if not split_dir.exists():
            raise FileNotFoundError(
                f"GOT-10k {self.split} split not found at {split_dir}\n"
                f"Download from: http://got-10k.aitestunion.com/downloads"
            )

        seqs = []
        for seq_dir in sorted(split_dir.iterdir()):
            if not seq_dir.is_dir():
                continue
            gt_file = seq_dir / "groundtruth.txt"
            if not gt_file.exists():
                continue
            try:
                seq = GOT10kSequence(str(seq_dir))
                if len(seq) >= 2:
                    seqs.append(seq)
            except Exception as e:
                print(f"[GOT10kDataset] Skipping {seq_dir.name}: {e}")
        return seqs

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx: int) -> dict:
        seq = self.sequences[idx]
        n = len(seq)

        # Sample template frame (prefer first frame for template)
        if self.split == "train":
            t_z = random.randint(0, max(0, n - 2))
        else:
            t_z = 0  # always use first frame for val/test

        # Sample search frame with bounded gap
        gap = random.randint(1, min(self.max_frame_gap, n - 1 - t_z))
        t_x = t_z + gap

        # Load frames
        frame_z = seq.get_frame(t_z)
        frame_x = seq.get_frame(t_x)

        box_z = seq.get_box_xyxy(t_z)   # template box [x1,y1,x2,y2]
        box_x = seq.get_box_xyxy(t_x)   # search box [x1,y1,x2,y2]

        # Crop template and search regions
        template_crop, _ = crop_and_resize(
            frame_z, box_z, self.template_size, context_factor=2.0
        )
        search_crop, gt_box_norm = crop_and_resize(
            frame_x, box_x, self.search_size, context_factor=4.0
        )

        # Apply augmentation transforms
        template_tensor = self.transforms(template_crop, is_search=False)
        search_tensor = self.transforms(search_crop, is_search=True)

        return {
            "template": template_tensor,           # (3, Hz, Wz)
            "search": search_tensor,               # (3, Hx, Wx)
            "gt_boxes": torch.from_numpy(gt_box_norm),   # (4,) [x1,y1,x2,y2]
            "seq_name": seq.name,
            "frame_idx": t_x,
        }


# ---------------------------------------------------------------------------
# Test/Evaluation Dataset (one sequence at a time)
# ---------------------------------------------------------------------------

class GOT10kTestSequence(Dataset):
    """
    Wraps a single GOT-10k sequence for frame-by-frame evaluation.

    Usage:
        dataset = GOT10kTestSequence(seq_dir)
        frame0, box0 = dataset.get_init_frame()   # initialize tracker
        for i in range(1, len(dataset)):
            frame, _ = dataset[i]                  # track
    """

    def __init__(self, seq_dir: str, search_size: int = 256):
        self.seq = GOT10kSequence(seq_dir)
        self.search_size = search_size
        from .transforms import ToTensor
        self.to_tensor = ToTensor()

    def get_init_info(self):
        """Returns (frame_0_PIL, box_xyxy_0) for tracker initialization."""
        return self.seq.get_frame(0), self.seq.get_box_xyxy(0)

    def __len__(self):
        return len(self.seq)

    def __getitem__(self, idx: int):
        frame = self.seq.get_frame(idx)
        box = self.seq.get_box_xyxy(idx)
        return frame, box


# ---------------------------------------------------------------------------
# DataLoader factory
# ---------------------------------------------------------------------------

def build_got10k_loader(cfg: dict):
    """Build GOT-10k train/val DataLoaders from config."""
    import torch
    from torch.utils.data import DataLoader

    data_cfg = cfg.get("data", {})
    root = data_cfg.get("root", "./data/got10k")

    train_ds = GOT10kDataset(
        root=root,
        split="train",
        template_size=cfg.get("template_size", 128),
        search_size=cfg.get("search_size", 256),
        max_frame_gap=data_cfg.get("max_frame_gap", 100),
    )

    val_ds = GOT10kDataset(
        root=root,
        split="val",
        template_size=cfg.get("template_size", 128),
        search_size=cfg.get("search_size", 256),
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=data_cfg.get("batch_size", 16),
        shuffle=True,
        num_workers=data_cfg.get("num_workers", 4),
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=data_cfg.get("batch_size", 16),
        shuffle=False,
        num_workers=data_cfg.get("num_workers", 4),
        pin_memory=True,
    )

    return train_loader, val_loader
