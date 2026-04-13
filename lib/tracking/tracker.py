"""
Online Tracker (Inference Engine)
===================================
Wraps OSTrack model into a stateful online tracker:
  1. init(frame, bbox)  — set template from first frame
  2. track(frame)       — localize object in new frame, return bbox

Key design decisions:
  - Search region is centered on the PREVIOUS prediction (not image center)
  - Search scale = 4× the template size (wider context for fast motion)
  - Window penalty: cosine window applied to score map to penalize edges
"""

import math
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from typing import Tuple, Optional

from ..models.ostrack import OSTrack
from ..datasets.got10k import crop_and_resize, xywh_to_xyxy, xyxy_to_xywh
from ..utils.box_ops import scale_box_to_image


class OSTracker:
    """
    Stateful online tracker using OSTrack.

    Usage:
        tracker = OSTracker(model, device="cuda")
        tracker.init(frame_pil, init_box_xyxy)
        for frame in video:
            pred_box = tracker.track(frame_pil)
    """

    def __init__(
        self,
        model: OSTrack,
        device: str = "cuda",
        template_size: int = 128,
        search_size: int = 256,
        template_factor: float = 2.0,
        search_factor: float = 4.0,
        window_influence: float = 0.49,
        image_mean: tuple = (0.485, 0.456, 0.406),
        image_std: tuple = (0.229, 0.224, 0.225),
    ):
        """
        Args:
            model:             Trained OSTrack model.
            device:            "cuda" or "cpu".
            template_size:     Template crop size in pixels.
            search_size:       Search crop size in pixels.
            template_factor:   Context factor for template crop.
            search_factor:     Context factor for search crop (larger = wider).
            window_influence:  Weight of cosine window penalty (0 = disabled).
        """
        self.model = model.eval()
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        self.template_size = template_size
        self.search_size = search_size
        self.template_factor = template_factor
        self.search_factor = search_factor
        self.window_influence = window_influence
        self.mean = torch.tensor(image_mean, device=self.device).view(3, 1, 1)
        self.std = torch.tensor(image_std, device=self.device).view(3, 1, 1)

        # Cosine window for score map penalty (precomputed)
        feat_sz = search_size // 16  # patch_size=16
        self._cosine_window = self._make_cosine_window(feat_sz)

        # State
        self._state_box = None  # Current predicted box [x1,y1,x2,y2]
        self._state_frame = None  # Current frame PIL

    def _make_cosine_window(self, size: int) -> np.ndarray:
        """
        2D Hanning window for spatial penalty.

        Penalizes predictions near the edge of the search region,
        encouraging the tracker to predict near the center (where
        the object is likely to be, given inertia).
        """
        win = np.outer(np.hanning(size), np.hanning(size))
        return win / win.sum()

    def _preprocess(self, img: Image.Image) -> torch.Tensor:
        """PIL Image → normalized float tensor (1, 3, H, W)."""
        arr = np.array(img, dtype=np.float32) / 255.0
        t = torch.from_numpy(arr).permute(2, 0, 1).to(self.device)
        t = (t - self.mean) / self.std
        return t.unsqueeze(0)

    def init(self, frame: Image.Image, init_box: np.ndarray):
        """
        Initialize tracker from first frame.

        Args:
            frame:    First frame as PIL Image.
            init_box: Initial bounding box [x1, y1, x2, y2] in pixel coords.
        """
        self._state_box = init_box.copy()
        self._state_frame = frame

        # Crop and store template
        template_crop, _ = crop_and_resize(
            frame, init_box, self.template_size, self.template_factor
        )
        template_tensor = self._preprocess(template_crop)

        # Pre-compute template tokens (reused every frame)
        self.model.track_init(template_tensor, torch.tensor(init_box).unsqueeze(0))

    def track(self, frame: Image.Image) -> np.ndarray:
        """
        Localize object in new frame.

        Args:
            frame: Current frame as PIL Image.

        Returns:
            pred_box: Predicted box [x1, y1, x2, y2] in pixel coords.
        """
        # Compute search crop around previous prediction
        search_crop, _ = crop_and_resize(
            frame, self._state_box, self.search_size, self.search_factor
        )
        search_tensor = self._preprocess(search_crop)

        # Run model
        with torch.no_grad():
            output = self.model.track_step(search_tensor)

        # Get predicted normalized box in search crop
        pred_norm = output["pred_boxes"][0].cpu().numpy()  # [x1,y1,x2,y2] in [0,1]

        # Apply cosine window penalty to score map
        if output.get("score_map") is not None and self.window_influence > 0:
            score = output["score_map"][0, 0].cpu().numpy()
            score = score * (1 - self.window_influence) + \
                    self._cosine_window * self.window_influence

        # Map normalized prediction back to image coordinates
        # First, get the search crop bounds in image space
        img_w, img_h = frame.size
        cx, cy = (self._state_box[0] + self._state_box[2]) / 2, \
                 (self._state_box[1] + self._state_box[3]) / 2
        obj_w = self._state_box[2] - self._state_box[0]
        obj_h = self._state_box[3] - self._state_box[1]
        context = (obj_w + obj_h) / 2 * self.search_factor
        crop_sz = math.sqrt((obj_w + context) * (obj_h + context))

        crop_x1 = cx - crop_sz / 2
        crop_y1 = cy - crop_sz / 2
        crop_x2 = cx + crop_sz / 2
        crop_y2 = cy + crop_sz / 2

        search_crop_box = np.array([crop_x1, crop_y1, crop_x2, crop_y2])
        pred_abs = scale_box_to_image(pred_norm, search_crop_box)

        # Clip to image bounds
        pred_abs[0] = np.clip(pred_abs[0], 0, img_w)
        pred_abs[1] = np.clip(pred_abs[1], 0, img_h)
        pred_abs[2] = np.clip(pred_abs[2], 0, img_w)
        pred_abs[3] = np.clip(pred_abs[3], 0, img_h)

        # Update state
        self._state_box = pred_abs
        self._state_frame = frame

        return pred_abs

    def track_sequence(
        self, frames: list, init_box: np.ndarray
    ) -> Tuple[list, list]:
        """
        Track across a list of PIL frames.

        Args:
            frames:   List of PIL Images.
            init_box: Initial box [x1,y1,x2,y2] for frame 0.

        Returns:
            (predictions, times): Lists of predicted boxes and per-frame times.
        """
        import time
        self.init(frames[0], init_box)
        predictions = [init_box.copy()]
        times = [0.0]

        for frame in frames[1:]:
            t0 = time.time()
            pred = self.track(frame)
            times.append(time.time() - t0)
            predictions.append(pred)

        fps = 1.0 / (np.mean(times[1:]) + 1e-9)
        print(f"[OSTracker] Tracked {len(frames)} frames at {fps:.1f} FPS")
        return predictions, times
