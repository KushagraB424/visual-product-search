"""
Map Roboflow inference JSON to PredictionResponse for the same UI as local ML.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List

import numpy as np

from app.schemas import PredictedItem, PredictionResponse
from utils.image_utils import detect_pattern, extract_dominant_color, save_crop


def _normalize_label(raw: str) -> str:
    s = raw.strip().lower()
    s = re.sub(r"[\s/]+", "_", s)
    s = s.replace("-", "_")
    return s or "item"


def _bbox_from_prediction(pred: Dict[str, Any], img_w: int, img_h: int) -> List[float]:
    """Roboflow often uses center (x,y) + width + height in pixels or normalized."""
    if "points" in pred and pred["points"]:
        pts = pred["points"]
        xs: List[float] = []
        ys: List[float] = []
        for p in pts:
            if isinstance(p, dict):
                xs.append(float(p.get("x", 0)))
                ys.append(float(p.get("y", 0)))
            elif isinstance(p, (list, tuple)) and len(p) >= 2:
                xs.append(float(p[0]))
                ys.append(float(p[1]))
        if xs and ys:
            x1, x2 = min(xs), max(xs)
            y1, y2 = min(ys), max(ys)
        else:
            return [0.0, 0.0, float(img_w), float(img_h)]
    elif all(k in pred for k in ("x", "y", "width", "height")):
        xc = float(pred["x"])
        yc = float(pred["y"])
        bw = float(pred["width"])
        bh = float(pred["height"])
        # Normalized 0–1 (typical when values are small)
        if max(abs(xc), abs(yc), bw, bh) <= 1.0 and img_w > 2:
            xc *= img_w
            yc *= img_h
            bw *= img_w
            bh *= img_h
        x1 = xc - bw / 2
        y1 = yc - bh / 2
        x2 = xc + bw / 2
        y2 = yc + bh / 2
    elif all(k in pred for k in ("x1", "y1", "x2", "y2")):
        x1, y1, x2, y2 = (
            float(pred["x1"]),
            float(pred["y1"]),
            float(pred["x2"]),
            float(pred["y2"]),
        )
    else:
        return [0.0, 0.0, float(img_w), float(img_h)]

    x1 = max(0.0, min(float(x1), float(img_w)))
    y1 = max(0.0, min(float(y1), float(img_h)))
    x2 = max(0.0, min(float(x2), float(img_w)))
    y2 = max(0.0, min(float(y2), float(img_h)))
    if x2 <= x1 or y2 <= y1:
        return [0.0, 0.0, float(img_w), float(img_h)]
    return [x1, y1, x2, y2]


def roboflow_json_to_prediction_response(
    image: np.ndarray,
    rf: Dict[str, Any],
    processing_time_ms: float,
) -> PredictionResponse:
    """
    Convert Roboflow ``model.predict().json()`` output into PredictionResponse.
    """
    h, w = image.shape[:2]
    predictions = rf.get("predictions")
    if predictions is None:
        predictions = []

    if not predictions:
        return PredictionResponse(
            success=True,
            items=[],
            num_items_detected=0,
            processing_time_ms=round(processing_time_ms, 1),
            image_width=w,
            image_height=h,
            message="No clothing items detected in the image.",
        )

    meta = rf.get("image") or {}
    iw = int(meta.get("width", w))
    ih = int(meta.get("height", h))

    items: List[PredictedItem] = []
    for idx, pred in enumerate(predictions):
        if not isinstance(pred, dict):
            continue
        conf = float(pred.get("confidence", pred.get("score", 0.0)))
        conf = max(0.0, min(1.0, conf))
        raw = pred.get("class") or pred.get("label") or "item"
        label = _normalize_label(str(raw))

        bbox = _bbox_from_prediction(pred, iw, ih)
        x1, y1, x2, y2 = map(int, bbox)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, max(x1 + 1, x2)), min(h, max(y1 + 1, y2))
        crop_bgr = image[y1:y2, x1:x2].copy()

        if crop_bgr.size == 0:
            color, pattern = "unknown", "unknown"
            crop_url = None
        else:
            color = extract_dominant_color(crop_bgr)
            pattern = detect_pattern(crop_bgr)
            # save_crop expects RGBA in pipeline; PNG accepts BGR
            crop_url = save_crop(crop_bgr, prefix=label)

        items.append(
            PredictedItem(
                item_id=idx,
                label=label,
                confidence=round(conf, 3),
                color=color,
                pattern=pattern,
                bbox=bbox,
                crop_path=crop_url,
                shopping_links=[],
            )
        )

    return PredictionResponse(
        success=True,
        items=items,
        num_items_detected=len(items),
        processing_time_ms=round(processing_time_ms, 1),
        image_width=w,
        image_height=h,
        message=f"Detected {len(items)} clothing item(s) (cloud segmentation).",
    )
