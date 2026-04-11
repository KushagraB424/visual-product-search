"""
Classification service — EfficientNet-B0 for clothing category prediction.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
import timm

from app.config import (
    CLASSIFICATION_MODEL_PATH,
    CLOTHING_CATEGORIES,
    NUM_CLASSES,
)

logger = logging.getLogger(__name__)

# ImageNet normalisation (EfficientNet pretrained)
_TRANSFORM = transforms.Compose(
    [
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ]
)


class ClassificationService:
    """
    Loads EfficientNet-B0 with a custom head for clothing classification.
    Falls back to a heuristic if no fine-tuned weights are available.
    """

    def __init__(self, model_path: str = CLASSIFICATION_MODEL_PATH):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.categories = CLOTHING_CATEGORIES
        self._use_heuristic = False

        try:
            self.model = timm.create_model(
                "efficientnet_b0", pretrained=True, num_classes=NUM_CLASSES
            )
            weights_file = Path(model_path)
            if weights_file.exists():
                state = torch.load(str(weights_file), map_location=self.device)
                self.model.load_state_dict(state, strict=False)
                logger.info("Loaded fine-tuned classifier weights from %s", model_path)
            else:
                logger.warning(
                    "No fine-tuned weights at %s — using ImageNet backbone "
                    "with heuristic-assisted classification.",
                    model_path,
                )
                self._use_heuristic = True

            self.model.to(self.device).eval()
            logger.info("Classification model ready on %s", self.device)

        except Exception as exc:
            logger.error("Failed to load classification model: %s", exc)
            self._use_heuristic = True
            self.model = None

    # ------------------------------------------------------------------ #

    def classify(
        self, crop: np.ndarray, region_hint: str = ""
    ) -> Tuple[str, float, dict]:
        """
        Classify a cropped clothing image.

        Returns:
            (label, confidence, {"color": ..., "pattern": ...})
        """
        if self._use_heuristic or self.model is None:
            return self._heuristic_classify(crop, region_hint)

        # Prepare tensor — convert to grayscale first to match
        # FashionMNIST training data, then replicate to 3 channels
        bgr = crop[:, :, :3] if crop.shape[2] == 4 else crop
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)  # 3-ch grayscale
        tensor = _TRANSFORM(rgb).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits = self.model(tensor)
            probs = torch.softmax(logits, dim=1)[0]
            conf, idx = probs.max(0)
            label = self.categories[idx.item()]
            confidence = conf.item()

        attributes = self._detect_attributes(bgr)
        return label, confidence, attributes

    # ------------------------------------------------------------------ #

    @staticmethod
    def _detect_attributes(bgr: np.ndarray) -> dict:
        """Detect dominant color and basic pattern from a BGR crop."""
        # ── Dominant color ──
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        h, s, v = hsv[:, :, 0].mean(), hsv[:, :, 1].mean(), hsv[:, :, 2].mean()

        if s < 30:
            if v < 60:
                color = "black"
            elif v > 200:
                color = "white"
            else:
                color = "gray"
        elif h < 10 or h > 165:
            color = "red"
        elif 10 <= h < 25:
            color = "orange"
        elif 25 <= h < 35:
            color = "yellow"
        elif 35 <= h < 80:
            color = "green"
        elif 80 <= h < 130:
            color = "blue"
        elif 130 <= h < 165:
            color = "purple"
        else:
            color = "mixed"

        # ── Pattern detection (simple edge-density heuristic) ──
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        edge_density = edges.mean() / 255

        if edge_density > 0.15:
            pattern = "patterned"
        elif edge_density > 0.07:
            pattern = "striped"
        else:
            pattern = "solid"

        return {"color": color, "pattern": pattern}

    # ------------------------------------------------------------------ #

    @staticmethod
    def _heuristic_classify(
        crop: np.ndarray, region_hint: str
    ) -> Tuple[str, float, dict]:
        """
        Fallback heuristic when no fine-tuned weights are available.
        Uses the region hint (upper_garment / lower_garment) plus simple
        aspect-ratio and colour cues to guess the category.
        """
        h, w = crop.shape[:2]
        aspect = w / max(h, 1)

        if "upper" in region_hint:
            if aspect > 1.2:
                return "jacket", 0.55, {}
            return "shirt", 0.50, {}
        elif "lower" in region_hint:
            if aspect < 0.55:
                return "jeans", 0.50, {}
            return "pants", 0.50, {}
        else:
            if h > w * 1.8:
                return "dress", 0.45, {}
            return "shirt", 0.40, {}
