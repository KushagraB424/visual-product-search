"""
Background Manager — replaces backgrounds with real-world scenes,
procedural patterns, or solid colour fills.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class BackgroundManager:
    """
    Manages a pool of background images and provides methods to replace
    the background of a foreground BGRA image.
    """

    def __init__(self, background_dir: Optional[Path] = None):
        self.backgrounds: List[np.ndarray] = []
        if background_dir and background_dir.exists():
            self._load_backgrounds(background_dir)

    # ------------------------------------------------------------------ #
    # Loading
    # ------------------------------------------------------------------ #

    def _load_backgrounds(self, bg_dir: Path):
        exts = (".jpg", ".jpeg", ".png", ".webp")
        for p in sorted(bg_dir.iterdir()):
            if p.suffix.lower() in exts:
                img = cv2.imread(str(p))
                if img is not None:
                    self.backgrounds.append(img)
        logger.info("Loaded %d background images", len(self.backgrounds))

    # ------------------------------------------------------------------ #
    # Core method
    # ------------------------------------------------------------------ #

    def replace_background(
        self,
        foreground_bgra: np.ndarray,
        target_size: Tuple[int, int] = (640, 640),
        rng: Optional[np.random.RandomState] = None,
        mode: str = "auto",
    ) -> np.ndarray:
        """
        Place *foreground_bgra* (BGRA) onto a new background.

        Modes:
        - "real"      : pick a random real background from the pool
        - "gradient"  : procedurally generated gradient
        - "clutter"   : random shapes simulating clutter
        - "solid"     : solid random colour
        - "auto"      : pick the best available mode
        """
        if rng is None:
            rng = np.random.RandomState()

        H, W = target_size
        alpha = foreground_bgra[:, :, 3].astype(np.float32) / 255.0

        # Resize foreground to target
        fg = cv2.resize(foreground_bgra, (W, H))
        alpha = cv2.resize(alpha, (W, H))

        # Choose mode
        if mode == "auto":
            if self.backgrounds:
                mode = rng.choice(["real", "gradient", "solid"], p=[0.6, 0.25, 0.15])
            else:
                mode = rng.choice(["gradient", "clutter", "solid"])

        # Generate background
        if mode == "real" and self.backgrounds:
            bg = self._random_real(W, H, rng)
        elif mode == "gradient":
            bg = self._random_gradient(W, H, rng)
        elif mode == "clutter":
            bg = self._random_clutter(W, H, rng)
        else:
            bg = self._random_solid(W, H, rng)

        # Alpha composite
        alpha_3 = np.stack([alpha] * 3, axis=-1)
        composite = (fg[:, :, :3].astype(np.float32) * alpha_3 +
                     bg.astype(np.float32) * (1 - alpha_3))
        return composite.astype(np.uint8)

    # ------------------------------------------------------------------ #
    # Background generators
    # ------------------------------------------------------------------ #

    def _random_real(self, W: int, H: int, rng) -> np.ndarray:
        idx = rng.randint(len(self.backgrounds))
        bg = self.backgrounds[idx]
        return cv2.resize(bg, (W, H))

    @staticmethod
    def _random_gradient(W: int, H: int, rng) -> np.ndarray:
        c1 = rng.randint(0, 256, size=3).tolist()
        c2 = rng.randint(0, 256, size=3).tolist()
        gradient = np.zeros((H, W, 3), dtype=np.uint8)
        for y in range(H):
            t = y / max(H - 1, 1)
            gradient[y, :] = [int(c1[c] * (1 - t) + c2[c] * t) for c in range(3)]
        return gradient

    @staticmethod
    def _random_clutter(W: int, H: int, rng) -> np.ndarray:
        bg = np.full((H, W, 3), rng.randint(30, 80), dtype=np.uint8)
        for _ in range(rng.randint(8, 25)):
            shape = rng.choice(["rect", "circle", "line"])
            color = tuple(rng.randint(0, 256, size=3).tolist())
            if shape == "rect":
                pt1 = (rng.randint(0, W), rng.randint(0, H))
                pt2 = (rng.randint(0, W), rng.randint(0, H))
                cv2.rectangle(bg, pt1, pt2, color, -1)
            elif shape == "circle":
                centre = (rng.randint(0, W), rng.randint(0, H))
                radius = rng.randint(10, max(W, H) // 4)
                cv2.circle(bg, centre, radius, color, -1)
            else:
                pt1 = (rng.randint(0, W), rng.randint(0, H))
                pt2 = (rng.randint(0, W), rng.randint(0, H))
                cv2.line(bg, pt1, pt2, color, rng.randint(1, 8))
        return bg

    @staticmethod
    def _random_solid(W: int, H: int, rng) -> np.ndarray:
        color = rng.randint(0, 256, size=3).tolist()
        return np.full((H, W, 3), color, dtype=np.uint8)
