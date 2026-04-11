"""
Prediction pipeline — orchestrates segmentation → classification.

Shopping links are now handled separately via the /api/chat endpoint,
keeping prediction fast and decoupled.
"""

from __future__ import annotations

import logging
import time
from typing import List

import numpy as np

from app.schemas import PredictedItem, PredictionResponse
from app.services.segmentation import SegmentationService
from app.services.classification import ClassificationService
from app.services.agent import ShoppingAgent
from utils.image_utils import (
    crop_with_mask,
    detect_pattern,
    extract_dominant_color,
    save_crop,
)

logger = logging.getLogger(__name__)


class PredictionPipeline:
    """Full inference pipeline: segment → crop → classify."""

    def __init__(
        self,
        seg_service: SegmentationService,
        cls_service: ClassificationService,
        agent: ShoppingAgent,
    ):
        self.seg = seg_service
        self.cls = cls_service
        self.agent = agent

    # ------------------------------------------------------------------ #

    async def run(self, image: np.ndarray) -> PredictionResponse:
        t0 = time.perf_counter()
        h, w = image.shape[:2]

        # 1. Segment
        seg_objects = self.seg.segment(image)
        if not seg_objects:
            return PredictionResponse(
                success=True,
                items=[],
                num_items_detected=0,
                processing_time_ms=(time.perf_counter() - t0) * 1000,
                image_width=w,
                image_height=h,
                message="No clothing items detected in the image.",
            )

        # 2. Crop + classify each segment
        items: List[PredictedItem] = []

        for idx, obj in enumerate(seg_objects):
            # Crop with mask
            cropped = crop_with_mask(image, obj.mask, obj.bbox)

            # Classify
            label, conf, attrs = self.cls.classify(cropped, obj.class_name)

            # Colour & pattern
            color = extract_dominant_color(cropped)
            pattern = detect_pattern(cropped)

            # Save crop image
            crop_url = save_crop(cropped, prefix=label)

            item = PredictedItem(
                item_id=idx,
                label=label,
                confidence=round(conf, 3),
                color=color,
                pattern=pattern,
                bbox=obj.bbox,
                crop_path=crop_url,
                shopping_links=[],  # populated via /api/chat now
            )
            items.append(item)

        elapsed = (time.perf_counter() - t0) * 1000
        return PredictionResponse(
            success=True,
            items=items,
            num_items_detected=len(items),
            processing_time_ms=round(elapsed, 1),
            image_width=w,
            image_height=h,
            message=f"Detected {len(items)} clothing item(s).",
        )
