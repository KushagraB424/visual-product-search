"""
Configuration dataclass for the augmentation pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Tuple


@dataclass
class AugmentationConfig:
    """All tuneable knobs for the occlusion-aware augmentation pipeline."""

    # How many augmented images to generate per source image
    num_augmented_per_image: int = 5

    # Occlusion
    enable_occlusion: bool = True
    occlusion_ratio_range: Tuple[float, float] = (0.10, 0.70)

    # Multi-person
    enable_multi_person: bool = True
    max_people_per_scene: int = 4

    # Background replacement
    enable_background_replace: bool = True
    background_dir: Path = Path("augmentation_pipeline/backgrounds")

    # Geometric / Photometric transforms
    enable_transforms: bool = True
    rotation_range: Tuple[float, float] = (-15.0, 15.0)
    scale_range: Tuple[float, float] = (0.8, 1.2)
    brightness_range: Tuple[float, float] = (0.7, 1.3)
    contrast_range: Tuple[float, float] = (0.7, 1.3)
    hue_shift_range: Tuple[int, int] = (-10, 10)

    # Output
    output_size: Tuple[int, int] = (640, 640)
    output_format: str = "png"

    # Garment bank
    garment_bank_dir: Path = Path("augmentation_pipeline/garment_bank")

    # Reproducibility
    random_seed: int = 42
