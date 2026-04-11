"""
Occlusion Simulator — generates realistic occlusion between garment layers.

This is the **core research contribution**: simulating how one garment
occludes another (jacket covering shirt, bag covering side of torso, etc.)
while maintaining semantic consistency.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class OcclusionResult:
    """Output of an occlusion simulation."""
    composite: np.ndarray          # combined BGRA image
    combined_mask: np.ndarray      # binary mask of all visible garments
    layer_masks: List[np.ndarray]  # per-layer visibility masks (after occlusion)
    layer_labels: List[str]        # label of each layer
    occlusion_ratios: List[float]  # actual occlusion ratio per layer


# ── Garment depth ordering (higher index = on top) ──────────────────────

_DEPTH_ORDER = {
    "tank_top": 0, "t-shirt": 1, "shirt": 2, "blouse": 2,
    "sweater": 3, "hoodie": 4, "jacket": 5, "coat": 6,
    "upper_garment": 2,
    "jeans": 1, "pants": 1, "shorts": 1, "skirt": 1,
    "lower_garment": 1,
    "dress": 2, "suit": 5,
    "garment": 2, "other": 1,
}


class OcclusionSimulator:
    """
    Simulate occlusion between multiple garment layers.

    Given a set of RGBA garment images and their labels it:
    1. Sorts them by depth (innermost first)
    2. Composites them with controlled overlap
    3. Returns per-layer visibility masks reflecting what's actually visible
    """

    @staticmethod
    def simulate(
        garment_images: List[np.ndarray],
        garment_labels: List[str],
        canvas_size: Tuple[int, int] = (640, 640),
        overlap_ratio: float = 0.3,
        randomize_position: bool = True,
        rng: Optional[np.random.RandomState] = None,
    ) -> OcclusionResult:
        """
        Compose garments onto a canvas with occlusion.

        Parameters
        ----------
        garment_images : list of BGRA np.ndarray
        garment_labels : parallel list of category strings
        canvas_size    : (H, W) of the output canvas
        overlap_ratio  : controls how much each layer overlaps the previous
                         (0 = no overlap, 1 = full overlap)
        randomize_position : if True, jitter placement slightly
        rng            : random state for reproducibility
        """
        if rng is None:
            rng = np.random.RandomState(42)

        H, W = canvas_size
        canvas = np.zeros((H, W, 4), dtype=np.uint8)  # BGRA

        # Sort by depth
        depth_indices = sorted(
            range(len(garment_images)),
            key=lambda i: _DEPTH_ORDER.get(garment_labels[i], 2),
        )

        layer_masks: List[np.ndarray] = []
        layer_labels: List[str] = []
        occlusion_ratios: List[float] = []

        # Base position: centre of canvas
        cx, cy = W // 2, H // 2

        for order, idx in enumerate(depth_indices):
            g_img = garment_images[idx].copy()
            label = garment_labels[idx]

            # Resize garment to fit within canvas (max 70% of canvas)
            gh, gw = g_img.shape[:2]
            max_dim = int(min(H, W) * 0.7)
            scale = max_dim / max(gh, gw)
            if scale < 1:
                g_img = cv2.resize(g_img, (int(gw * scale), int(gh * scale)))
                gh, gw = g_img.shape[:2]

            # Position with overlap
            if randomize_position:
                jx = rng.randint(-int(gw * overlap_ratio * 0.5),
                                  int(gw * overlap_ratio * 0.5) + 1)
                jy = rng.randint(-int(gh * overlap_ratio * 0.3),
                                  int(gh * overlap_ratio * 0.3) + 1)
            else:
                jx, jy = 0, 0

            x_start = cx - gw // 2 + jx
            y_start = cy - gh // 2 + jy

            # Clamp to canvas
            x_start = max(0, min(x_start, W - gw))
            y_start = max(0, min(y_start, H - gh))

            # Alpha mask of this garment
            alpha = g_img[:, :, 3].astype(np.float32) / 255.0

            # Feather edges for natural blending (3-pixel Gaussian)
            alpha = cv2.GaussianBlur(alpha, (5, 5), 1.0)

            # Composite onto canvas (back-to-front)
            roi = canvas[y_start:y_start+gh, x_start:x_start+gw]
            for c in range(3):
                roi[:, :, c] = (
                    alpha * g_img[:, :, c] +
                    (1 - alpha) * roi[:, :, c]
                ).astype(np.uint8)
            roi[:, :, 3] = np.clip(
                roi[:, :, 3].astype(np.float32) + alpha * 255, 0, 255
            ).astype(np.uint8)

            # Compute visibility mask (what's actually visible AFTER later layers)
            full_mask = np.zeros((H, W), dtype=np.float32)
            full_mask[y_start:y_start+gh, x_start:x_start+gw] = alpha

            layer_masks.append(full_mask)
            layer_labels.append(label)

        # Post-process: compute actual occlusion ratios
        # Each earlier layer is occluded by all later layers
        final_masks: List[np.ndarray] = []
        for i in range(len(layer_masks)):
            visible = layer_masks[i].copy()
            for j in range(i + 1, len(layer_masks)):
                # Later layers occlude earlier ones
                visible = visible * (1 - np.clip(layer_masks[j], 0, 1))
            final_masks.append((visible > 0.5).astype(np.uint8))

            # Occlusion ratio = 1 - (visible_area / original_area)
            orig_area = np.sum(layer_masks[i] > 0.5)
            vis_area = np.sum(visible > 0.5)
            ratio = 1 - (vis_area / max(orig_area, 1))
            occlusion_ratios.append(round(ratio, 3))

        combined_mask = np.clip(
            sum(m.astype(np.float32) for m in final_masks), 0, 1
        ).astype(np.uint8)

        return OcclusionResult(
            composite=canvas,
            combined_mask=combined_mask,
            layer_masks=final_masks,
            layer_labels=layer_labels,
            occlusion_ratios=occlusion_ratios,
        )
