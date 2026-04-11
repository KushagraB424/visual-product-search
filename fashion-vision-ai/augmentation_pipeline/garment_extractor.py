"""
Garment Extractor — uses segmentation to extract individual garments from
source images and build a reusable garment bank.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class GarmentSample:
    """One extracted garment with its metadata."""
    image_path: str          # path to RGBA garment image
    mask_path: str           # path to binary mask
    label: str               # e.g. upper_garment, lower_garment
    source_image: str        # original source image name
    bbox: List[float]        # [x1, y1, x2, y2] relative to crop
    width: int
    height: int


class GarmentExtractor:
    """
    Extract garments from images using a segmentation model and persist
    them as a garment bank for later recomposition.
    """

    def __init__(self, seg_model):
        """
        Parameters
        ----------
        seg_model : SegmentationService
            An already-loaded segmentation service instance.
        """
        self.seg = seg_model

    # ------------------------------------------------------------------ #

    def build_garment_bank(
        self,
        image_dir: Path,
        output_dir: Path,
        extensions: tuple = (".jpg", ".jpeg", ".png", ".webp"),
    ) -> List[GarmentSample]:
        """
        Scan *image_dir* for images, segment each into garments, save
        cropped RGBA + masks to *output_dir*, and return all samples.
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "images").mkdir(exist_ok=True)
        (output_dir / "masks").mkdir(exist_ok=True)

        samples: List[GarmentSample] = []
        image_paths = sorted(
            p for p in image_dir.iterdir() if p.suffix.lower() in extensions
        )

        for img_path in image_paths:
            logger.info("Extracting garments from %s", img_path.name)
            img = cv2.imread(str(img_path))
            if img is None:
                continue

            objects = self.seg.segment(img)
            for i, obj in enumerate(objects):
                garment = self._extract_single(
                    img, obj, img_path.stem, i, output_dir
                )
                if garment is not None:
                    samples.append(garment)

        # Save index
        index_path = output_dir / "index.json"
        with open(index_path, "w") as f:
            json.dump([asdict(s) for s in samples], f, indent=2)

        logger.info("Garment bank: %d garments from %d images", len(samples), len(image_paths))
        return samples

    # ------------------------------------------------------------------ #

    @staticmethod
    def _extract_single(
        image: np.ndarray,
        obj,
        source_stem: str,
        idx: int,
        output_dir: Path,
    ) -> Optional[GarmentSample]:
        """Crop one garment from the image and save it."""
        x1, y1, x2, y2 = map(int, obj.bbox)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(image.shape[1], x2), min(image.shape[0], y2)

        if (x2 - x1) < 20 or (y2 - y1) < 20:
            return None

        crop = image[y1:y2, x1:x2].copy()

        # Resize mask to image dimensions and crop
        mask = obj.mask
        if mask.shape[:2] != image.shape[:2]:
            mask = cv2.resize(
                mask.astype(np.uint8),
                (image.shape[1], image.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            )
        mask_crop = mask[y1:y2, x1:x2]

        # RGBA crop
        rgba = cv2.cvtColor(crop, cv2.COLOR_BGR2BGRA)
        rgba[:, :, 3] = (mask_crop * 255).astype(np.uint8)

        name = f"{source_stem}_{idx}"
        img_path = output_dir / "images" / f"{name}.png"
        mask_path = output_dir / "masks" / f"{name}.png"

        cv2.imwrite(str(img_path), rgba)
        cv2.imwrite(str(mask_path), (mask_crop * 255).astype(np.uint8))

        return GarmentSample(
            image_path=str(img_path),
            mask_path=str(mask_path),
            label=obj.class_name,
            source_image=source_stem,
            bbox=[0, 0, x2 - x1, y2 - y1],
            width=x2 - x1,
            height=y2 - y1,
        )

    # ------------------------------------------------------------------ #

    @staticmethod
    def load_garment_bank(bank_dir: Path) -> List[GarmentSample]:
        """Load a previously-built garment bank from its index file."""
        index_path = bank_dir / "index.json"
        if not index_path.exists():
            return []
        with open(index_path) as f:
            data = json.load(f)
        return [GarmentSample(**d) for d in data]
