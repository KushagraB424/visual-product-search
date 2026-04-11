"""
Image processing utilities: loading, cropping, color extraction.
"""

from __future__ import annotations

import io
import uuid
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image
from sklearn.cluster import KMeans

from app.config import CROPS_DIR, IMAGE_SIZE


# ── Colour name mapping (simple HSV-based) ──────────────────────────────

_COLOR_RANGES = [
    ((0, 0, 0), (180, 255, 50), "black"),
    ((0, 0, 200), (180, 30, 255), "white"),
    ((0, 0, 50), (180, 30, 200), "gray"),
    ((0, 100, 50), (10, 255, 255), "red"),
    ((170, 100, 50), (180, 255, 255), "red"),
    ((11, 100, 50), (25, 255, 255), "orange"),
    ((26, 100, 50), (34, 255, 255), "yellow"),
    ((35, 100, 50), (85, 255, 255), "green"),
    ((86, 100, 50), (125, 255, 255), "blue"),
    ((126, 100, 50), (145, 255, 255), "purple"),
    ((146, 100, 50), (169, 255, 255), "pink"),
]


def _closest_color_name(bgr: np.ndarray) -> str:
    """Map a BGR pixel to a human-readable colour name via HSV ranges."""
    hsv = cv2.cvtColor(np.uint8([[bgr]]), cv2.COLOR_BGR2HSV)[0][0]
    h, s, v = int(hsv[0]), int(hsv[1]), int(hsv[2])
    for (h_lo, s_lo, v_lo), (h_hi, s_hi, v_hi), name in _COLOR_RANGES:
        if h_lo <= h <= h_hi and s_lo <= s <= s_hi and v_lo <= v <= v_hi:
            return name
    return "multicolor"


# ── Public helpers ───────────────────────────────────────────────────────


async def load_image_from_upload(file) -> np.ndarray:
    """Read an uploaded file into a BGR numpy array."""
    data = await file.read()
    arr = np.frombuffer(data, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Could not decode the uploaded image.")
    return img


def resize_image(img: np.ndarray, size: int = IMAGE_SIZE) -> np.ndarray:
    """Resize keeping aspect ratio, padded to square."""
    h, w = img.shape[:2]
    scale = size / max(h, w)
    new_w, new_h = int(w * scale), int(h * scale)
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((size, size, 3), 114, dtype=np.uint8)
    y_off = (size - new_h) // 2
    x_off = (size - new_w) // 2
    canvas[y_off : y_off + new_h, x_off : x_off + new_w] = resized
    return canvas


def crop_with_mask(
    img: np.ndarray,
    mask: np.ndarray,
    bbox: List[float],
) -> np.ndarray:
    """
    Crop a garment region using its segmentation mask.
    Returns an RGBA image with transparent background outside the mask.
    """
    x1, y1, x2, y2 = map(int, bbox)
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(img.shape[1], x2), min(img.shape[0], y2)

    cropped = img[y1:y2, x1:x2].copy()
    # Resize mask to image dimensions if different
    if mask.shape[:2] != img.shape[:2]:
        mask = cv2.resize(
            mask.astype(np.uint8),
            (img.shape[1], img.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        )
    mask_crop = mask[y1:y2, x1:x2]

    # Build RGBA
    rgba = cv2.cvtColor(cropped, cv2.COLOR_BGR2BGRA)
    rgba[:, :, 3] = (mask_crop * 255).astype(np.uint8)
    return rgba


def save_crop(img: np.ndarray, prefix: str = "crop") -> str:
    """Save a cropped RGBA image, return the relative URL path."""
    filename = f"{prefix}_{uuid.uuid4().hex[:8]}.png"
    path = CROPS_DIR / filename
    cv2.imwrite(str(path), img)
    return f"/static/crops/{filename}"


def extract_dominant_color(
    img: np.ndarray, mask: Optional[np.ndarray] = None, k: int = 3
) -> str:
    """
    Extract the dominant colour from an image region.
    If *mask* is provided (same HxW), only masked pixels are considered.
    """
    bgr = img[:, :, :3] if img.shape[2] == 4 else img
    if mask is not None:
        if mask.shape[:2] != bgr.shape[:2]:
            mask = cv2.resize(
                mask.astype(np.uint8),
                (bgr.shape[1], bgr.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            )
        pixels = bgr[mask > 0].reshape(-1, 3)
    else:
        pixels = bgr.reshape(-1, 3)

    if len(pixels) < k:
        return "unknown"

    kmeans = KMeans(n_clusters=k, n_init=5, random_state=42)
    kmeans.fit(pixels)

    # Largest cluster = dominant
    counts = np.bincount(kmeans.labels_)
    dominant = kmeans.cluster_centers_[counts.argmax()].astype(np.uint8)
    return _closest_color_name(dominant)


def detect_pattern(img: np.ndarray) -> str:
    """
    Simple pattern detection heuristic based on edge density.
    """
    gray = cv2.cvtColor(img[:, :, :3], cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    edge_ratio = np.count_nonzero(edges) / edges.size
    if edge_ratio > 0.25:
        return "patterned"
    elif edge_ratio > 0.15:
        return "striped"
    elif edge_ratio > 0.08:
        return "textured"
    return "solid"
