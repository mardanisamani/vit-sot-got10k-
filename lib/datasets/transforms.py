"""
Data Augmentation Transforms for Tracking
==========================================
Tracking augmentation differs from classification:
  - Color jitter is applied INDEPENDENTLY to template and search
    (simulates lighting changes between frames)
  - Geometric augmentation must be CONSISTENT for bounding boxes
  - Flip applied to search independently (template stays fixed)
"""

import random
import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image


class TrackingTransforms:
    """
    Augmentation pipeline for SOT training.

    Template augmentation: mild color jitter only (it's the reference frame)
    Search augmentation:   stronger color jitter + horizontal flip
    """

    def __init__(
        self,
        split: str = "train",
        mean: tuple = (0.485, 0.456, 0.406),
        std: tuple = (0.229, 0.224, 0.225),
        brightness: float = 0.4,
        contrast: float = 0.4,
        saturation: float = 0.4,
        hue: float = 0.1,
        flip_prob: float = 0.5,
    ):
        self.split = split
        self.mean = mean
        self.std = std
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue
        self.flip_prob = flip_prob

    def __call__(self, img: Image.Image, is_search: bool = False) -> torch.Tensor:
        """
        Apply transforms to an image crop.

        Args:
            img:       PIL Image crop.
            is_search: If True, apply full search augmentation.
                       If False, apply mild template augmentation.

        Returns:
            Tensor of shape (3, H, W) normalized to ImageNet stats.
        """
        if self.split == "train":
            img = self._color_jitter(img, strong=is_search)
            if is_search and random.random() < self.flip_prob:
                img = TF.hflip(img)

        # ToTensor + Normalize
        tensor = TF.to_tensor(img)
        tensor = TF.normalize(tensor, mean=self.mean, std=self.std)
        return tensor

    def _color_jitter(self, img: Image.Image, strong: bool = False) -> Image.Image:
        scale = 1.5 if strong else 0.5
        brightness = random.uniform(
            max(0, 1 - self.brightness * scale),
            1 + self.brightness * scale
        )
        contrast = random.uniform(
            max(0, 1 - self.contrast * scale),
            1 + self.contrast * scale
        )
        saturation = random.uniform(
            max(0, 1 - self.saturation * scale),
            1 + self.saturation * scale
        )
        hue = random.uniform(-self.hue * scale, self.hue * scale)

        img = TF.adjust_brightness(img, brightness)
        img = TF.adjust_contrast(img, contrast)
        img = TF.adjust_saturation(img, saturation)
        img = TF.adjust_hue(img, hue)
        return img


class ToTensor:
    """Simple PIL → Tensor with ImageNet normalization."""

    def __init__(
        self,
        mean: tuple = (0.485, 0.456, 0.406),
        std: tuple = (0.229, 0.224, 0.225),
    ):
        self.mean = mean
        self.std = std

    def __call__(self, img: Image.Image) -> torch.Tensor:
        t = TF.to_tensor(img)
        return TF.normalize(t, mean=self.mean, std=self.std)


def denormalize(
    tensor: torch.Tensor,
    mean: tuple = (0.485, 0.456, 0.406),
    std: tuple = (0.229, 0.224, 0.225),
) -> np.ndarray:
    """
    Reverse ImageNet normalization for visualization.

    Args:
        tensor: (3, H, W) or (B, 3, H, W) normalized tensor.

    Returns:
        np.ndarray in [0, 255] uint8.
    """
    mean = torch.tensor(mean, device=tensor.device).view(-1, 1, 1)
    std = torch.tensor(std, device=tensor.device).view(-1, 1, 1)
    if tensor.ndim == 4:
        mean = mean.unsqueeze(0)
        std = std.unsqueeze(0)
    img = tensor * std + mean
    img = (img.clamp(0, 1) * 255).byte()
    return img.permute(1, 2, 0).cpu().numpy() if tensor.ndim == 3 else img
