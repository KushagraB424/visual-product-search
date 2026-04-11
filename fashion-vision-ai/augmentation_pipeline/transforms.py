"""
Standard image transforms applied on top of composite images.
Geometric + photometric + fashion-specific augmentations.
"""

from __future__ import annotations

from typing import Optional, Tuple

import cv2
import numpy as np


class ImageTransforms:
    """Collection of augmentation transforms for synthetic fashion images."""

    @staticmethod
    def random_rotation(
        img: np.ndarray, angle_range: Tuple[float, float] = (-15, 15),
        rng: Optional[np.random.RandomState] = None,
    ) -> np.ndarray:
        rng = rng or np.random.RandomState()
        angle = rng.uniform(*angle_range)
        h, w = img.shape[:2]
        M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
        return cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT_101)

    @staticmethod
    def random_scale(
        img: np.ndarray, scale_range: Tuple[float, float] = (0.8, 1.2),
        rng: Optional[np.random.RandomState] = None,
    ) -> np.ndarray:
        rng = rng or np.random.RandomState()
        scale = rng.uniform(*scale_range)
        h, w = img.shape[:2]
        new_h, new_w = int(h * scale), int(w * scale)
        resized = cv2.resize(img, (new_w, new_h))
        # Pad or crop back to original size
        canvas = np.full_like(img, 114)
        y_off = max(0, (h - new_h) // 2)
        x_off = max(0, (w - new_w) // 2)
        yr = min(new_h, h)
        xr = min(new_w, w)
        canvas[y_off:y_off+yr, x_off:x_off+xr] = resized[:yr, :xr]
        return canvas

    @staticmethod
    def random_flip(
        img: np.ndarray, p: float = 0.5,
        rng: Optional[np.random.RandomState] = None,
    ) -> np.ndarray:
        rng = rng or np.random.RandomState()
        if rng.random() < p:
            return cv2.flip(img, 1)
        return img

    @staticmethod
    def random_brightness_contrast(
        img: np.ndarray,
        brightness_range: Tuple[float, float] = (0.7, 1.3),
        contrast_range: Tuple[float, float] = (0.7, 1.3),
        rng: Optional[np.random.RandomState] = None,
    ) -> np.ndarray:
        rng = rng or np.random.RandomState()
        brightness = rng.uniform(*brightness_range)
        contrast = rng.uniform(*contrast_range)
        img_f = img.astype(np.float32)
        img_f = img_f * contrast + (brightness - 1.0) * 128
        return np.clip(img_f, 0, 255).astype(np.uint8)

    @staticmethod
    def random_hue_shift(
        img: np.ndarray, hue_range: Tuple[int, int] = (-10, 10),
        rng: Optional[np.random.RandomState] = None,
    ) -> np.ndarray:
        rng = rng or np.random.RandomState()
        shift = rng.randint(*hue_range)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        hsv[:, :, 0] = (hsv[:, :, 0].astype(int) + shift) % 180
        return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    @staticmethod
    def random_gaussian_blur(
        img: np.ndarray, p: float = 0.3,
        rng: Optional[np.random.RandomState] = None,
    ) -> np.ndarray:
        rng = rng or np.random.RandomState()
        if rng.random() < p:
            k = rng.choice([3, 5])
            return cv2.GaussianBlur(img, (k, k), 0)
        return img

    @staticmethod
    def random_perspective(
        img: np.ndarray, strength: float = 0.05,
        rng: Optional[np.random.RandomState] = None,
    ) -> np.ndarray:
        rng = rng or np.random.RandomState()
        h, w = img.shape[:2]
        dx = int(w * strength)
        dy = int(h * strength)
        src_pts = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
        dst_pts = np.float32([
            [rng.randint(0, dx + 1), rng.randint(0, dy + 1)],
            [w - rng.randint(0, dx + 1), rng.randint(0, dy + 1)],
            [w - rng.randint(0, dx + 1), h - rng.randint(0, dy + 1)],
            [rng.randint(0, dx + 1), h - rng.randint(0, dy + 1)],
        ])
        M = cv2.getPerspectiveTransform(src_pts, dst_pts)
        return cv2.warpPerspective(img, M, (w, h), borderMode=cv2.BORDER_REFLECT_101)

    # ------------------------------------------------------------------ #

    @classmethod
    def apply_all(
        cls,
        img: np.ndarray,
        rng: Optional[np.random.RandomState] = None,
        rotation_range=(-15, 15),
        scale_range=(0.8, 1.2),
        brightness_range=(0.7, 1.3),
        contrast_range=(0.7, 1.3),
        hue_range=(-10, 10),
    ) -> np.ndarray:
        """Apply all transforms sequentially."""
        rng = rng or np.random.RandomState()
        img = cls.random_flip(img, rng=rng)
        img = cls.random_rotation(img, rotation_range, rng=rng)
        img = cls.random_scale(img, scale_range, rng=rng)
        img = cls.random_brightness_contrast(img, brightness_range, contrast_range, rng=rng)
        img = cls.random_hue_shift(img, hue_range, rng=rng)
        img = cls.random_gaussian_blur(img, rng=rng)
        img = cls.random_perspective(img, rng=rng)
        return img
