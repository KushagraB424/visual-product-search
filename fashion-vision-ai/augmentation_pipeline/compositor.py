"""
Scene Compositor — assembles multi-person scenes from individual garments.
"""

from __future__ import annotations

import logging
from typing import List, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class SceneCompositor:
    """
    Compose multiple garments / people into a single scene.

    Features:
    - Multi-person placement with spacing
    - Scale randomisation per person
    - Maintains relative body proportions between upper & lower garments
    """

    @staticmethod
    def compose_multi_person(
        garment_groups: List[List[np.ndarray]],
        canvas_size: Tuple[int, int] = (640, 640),
        rng: Optional[np.random.RandomState] = None,
    ) -> Tuple[np.ndarray, List[np.ndarray]]:
        """
        Place multiple "person groups" onto a canvas.

        Parameters
        ----------
        garment_groups : list of lists of BGRA images.
            Each inner list represents one person and contains their
            garment images already composited (upper + lower stacked).
        canvas_size    : (H, W)
        rng            : random state

        Returns
        -------
        (canvas_bgr, masks_per_person)
        """
        if rng is None:
            rng = np.random.RandomState()

        H, W = canvas_size
        canvas = np.zeros((H, W, 4), dtype=np.uint8)
        masks = []

        n = len(garment_groups)
        if n == 0:
            return canvas[:, :, :3], masks

        # Horizontal slots for each person
        slot_w = W // n
        for i, grp in enumerate(garment_groups):
            if not grp:
                continue

            # Stack vertically (upper on top of lower)
            person_img = SceneCompositor._stack_garments(grp)

            # Scale to fit slot
            ph, pw = person_img.shape[:2]
            max_h = int(H * 0.9)
            max_w = int(slot_w * 0.85)
            scale = min(max_h / max(ph, 1), max_w / max(pw, 1), 1.0)
            scale *= rng.uniform(0.85, 1.0)  # slight random variation

            new_h, new_w = int(ph * scale), int(pw * scale)
            if new_h < 10 or new_w < 10:
                continue
            person_img = cv2.resize(person_img, (new_w, new_h))

            # Position within slot
            x_centre = slot_w * i + slot_w // 2
            x_start = max(0, x_centre - new_w // 2 + rng.randint(-10, 11))
            y_start = max(0, H - new_h - rng.randint(0, max(1, int(H * 0.05))))
            x_start = min(x_start, W - new_w)
            y_start = min(y_start, H - new_h)

            # Alpha composite
            alpha = person_img[:, :, 3].astype(np.float32) / 255.0
            roi = canvas[y_start:y_start+new_h, x_start:x_start+new_w]
            for c in range(3):
                roi[:, :, c] = (
                    alpha * person_img[:, :, c] +
                    (1 - alpha) * roi[:, :, c]
                ).astype(np.uint8)
            roi[:, :, 3] = np.clip(
                roi[:, :, 3].astype(np.float32) + alpha * 255, 0, 255
            ).astype(np.uint8)

            # Person mask
            pmask = np.zeros((H, W), dtype=np.uint8)
            pmask[y_start:y_start+new_h, x_start:x_start+new_w] = (alpha > 0.5).astype(np.uint8)
            masks.append(pmask)

        return canvas[:, :, :3], masks

    # ------------------------------------------------------------------ #

    @staticmethod
    def _stack_garments(garments: List[np.ndarray]) -> np.ndarray:
        """Stack garment images vertically (first on top)."""
        if len(garments) == 1:
            return garments[0]

        # Find max width
        max_w = max(g.shape[1] for g in garments)
        total_h = sum(g.shape[0] for g in garments)

        stacked = np.zeros((total_h, max_w, 4), dtype=np.uint8)
        y = 0
        for g in garments:
            gh, gw = g.shape[:2]
            x_off = (max_w - gw) // 2
            stacked[y:y+gh, x_off:x_off+gw] = g
            y += gh

        return stacked
