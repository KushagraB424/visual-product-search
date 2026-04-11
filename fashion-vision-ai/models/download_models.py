"""
Download / prepare pretrained model weights.

Run this script once before starting the server to ensure all weights
are cached locally:

    python -m models.download_models
"""

from __future__ import annotations

import logging
from pathlib import Path

import torch
import timm
from ultralytics import YOLO

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

WEIGHTS_DIR = Path(__file__).resolve().parent / "weights"
WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)

NUM_CLOTHING_CLASSES = 15


def download_segmentation_model():
    """Download YOLOv8n-seg pretrained on COCO."""
    logger.info("Downloading YOLOv8n-seg …")
    model = YOLO("yolov8n-seg.pt")
    logger.info("✅  YOLOv8n-seg ready (auto-cached by ultralytics)")
    return model


def prepare_classification_model():
    """
    Create an EfficientNet-B0 with a custom classification head for
    clothing categories and save the initial (untrained) weights.
    """
    out_path = WEIGHTS_DIR / "classifier.pth"
    if out_path.exists():
        logger.info("Classification weights already exist at %s", out_path)
        return

    logger.info("Creating EfficientNet-B0 classifier head (%d classes) …", NUM_CLOTHING_CLASSES)
    model = timm.create_model(
        "efficientnet_b0",
        pretrained=True,
        num_classes=NUM_CLOTHING_CLASSES,
    )
    torch.save(model.state_dict(), str(out_path))
    logger.info("✅  Saved classifier weights to %s", out_path)


def main():
    download_segmentation_model()
    prepare_classification_model()
    logger.info("All models ready.")


if __name__ == "__main__":
    main()
