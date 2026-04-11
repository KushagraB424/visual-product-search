"""
Segmentation service — wraps YOLOv8-seg for multi-object fashion segmentation.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import List, Optional

import cv2
import numpy as np
from ultralytics import YOLO

from app.config import (
    CONFIDENCE_THRESHOLD,
    IMAGE_SIZE,
    PERSON_CLASS_ID,
    SEGMENTATION_MODEL_PATH,
)

logger = logging.getLogger(__name__)


@dataclass
class SegmentedObject:
    """One detected & segmented object."""
    class_id: int
    class_name: str
    confidence: float
    bbox: List[float]          # [x1, y1, x2, y2] in original image coords
    mask: np.ndarray           # binary mask (H×W) at original resolution
    cropped_bgra: Optional[np.ndarray] = None  # filled later


class SegmentationService:
    """
    Loads a YOLOv8-seg model once and exposes a *segment* method that
    returns per-object masks, bounding boxes, and class info.
    """

    def __init__(self, model_path: str = SEGMENTATION_MODEL_PATH):
        logger.info("Loading segmentation model: %s", model_path)
        self.model = YOLO(model_path)
        logger.info("Segmentation model loaded successfully.")

    # ------------------------------------------------------------------ #

    def segment(
        self,
        image: np.ndarray,
        conf: float = CONFIDENCE_THRESHOLD,
        target_size: int = IMAGE_SIZE,
    ) -> List[SegmentedObject]:
        """
        Run instance segmentation on *image* (BGR numpy array).

        For fashion use we keep **person** detections and any accessory
        classes.  Each person detection is then split into clothing sub-
        regions by dividing the mask vertically (upper-body / lower-body /
        full-body) so downstream classification can label individual
        garments.
        """
        results = self.model.predict(
            source=image,
            imgsz=target_size,
            conf=conf,
            retina_masks=True,
            verbose=False,
        )

        objects: List[SegmentedObject] = []
        if not results or results[0].masks is None:
            return objects

        result = results[0]
        masks_data = result.masks.data.cpu().numpy()    # (N, mask_h, mask_w)
        boxes = result.boxes

        for i in range(len(boxes)):
            cls_id = int(boxes.cls[i].item())
            conf_score = float(boxes.conf[i].item())
            bbox = boxes.xyxy[i].cpu().numpy().tolist()  # [x1,y1,x2,y2]

            raw_mask = masks_data[i]  # (mask_h, mask_w)

            # Resize mask to original image dimensions
            mask_resized = cv2.resize(
                raw_mask,
                (image.shape[1], image.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            )
            binary_mask = (mask_resized > 0.5).astype(np.uint8)

            # For *person* detections we split into upper/lower body
            if cls_id == PERSON_CLASS_ID:
                sub_objects = self._split_person(
                    image, binary_mask, bbox, conf_score
                )
                objects.extend(sub_objects)
            else:
                class_name = self.model.names.get(cls_id, "unknown")
                objects.append(
                    SegmentedObject(
                        class_id=cls_id,
                        class_name=class_name,
                        confidence=conf_score,
                        bbox=bbox,
                        mask=binary_mask,
                    )
                )

        return objects

    # ------------------------------------------------------------------ #
    # Private helpers
    # ------------------------------------------------------------------ #

    @staticmethod
    def _split_person(
        image: np.ndarray,
        mask: np.ndarray,
        bbox: List[float],
        confidence: float,
    ) -> List[SegmentedObject]:
        """
        Split a person mask into upper-body and lower-body regions to
        isolate tops from bottoms.  If the bounding box is too small we
        keep the whole mask as one item.
        """
        x1, y1, x2, y2 = map(int, bbox)
        h = y2 - y1
        if h < 60:  # too small to split meaningfully
            return [
                SegmentedObject(
                    class_id=PERSON_CLASS_ID,
                    class_name="garment",
                    confidence=confidence,
                    bbox=bbox,
                    mask=mask,
                )
            ]

        mid_y = y1 + int(h * 0.45)  # roughly waist line

        upper_mask = mask.copy()
        upper_mask[mid_y:, :] = 0
        lower_mask = mask.copy()
        lower_mask[:mid_y, :] = 0

        parts: List[SegmentedObject] = []

        if upper_mask.sum() > 500:
            uy1 = y1
            uy2 = mid_y
            parts.append(
                SegmentedObject(
                    class_id=PERSON_CLASS_ID,
                    class_name="upper_garment",
                    confidence=confidence,
                    bbox=[x1, uy1, x2, uy2],
                    mask=upper_mask,
                )
            )

        if lower_mask.sum() > 500:
            ly1 = mid_y
            ly2 = y2
            parts.append(
                SegmentedObject(
                    class_id=PERSON_CLASS_ID,
                    class_name="lower_garment",
                    confidence=confidence,
                    bbox=[x1, ly1, x2, ly2],
                    mask=lower_mask,
                )
            )

        return parts if parts else [
            SegmentedObject(
                class_id=PERSON_CLASS_ID,
                class_name="garment",
                confidence=confidence,
                bbox=bbox,
                mask=mask,
            )
        ]
