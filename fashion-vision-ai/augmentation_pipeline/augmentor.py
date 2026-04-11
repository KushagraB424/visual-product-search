"""
OcclusionAwareAugmentor — Main orchestration class that ties together
garment extraction, occlusion simulation, background replacement, multi-person
composition, and standard transforms to produce an augmented dataset.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np

from augmentation_pipeline.background import BackgroundManager
from augmentation_pipeline.compositor import SceneCompositor
from augmentation_pipeline.config import AugmentationConfig
from augmentation_pipeline.garment_extractor import GarmentExtractor, GarmentSample
from augmentation_pipeline.occlusion import OcclusionSimulator
from augmentation_pipeline.transforms import ImageTransforms

logger = logging.getLogger(__name__)


@dataclass
class AugmentationReport:
    """Summary statistics of an augmentation run."""
    source_images: int
    garments_extracted: int
    augmented_images_generated: int
    output_dir: str


class OcclusionAwareAugmentor:
    """
    End-to-end augmentation pipeline:

    1. Extract garments from source images (→ garment bank)
    2. For each augmented sample:
       a. Sample random garments
       b. Simulate occlusion between layers
       c. Compose multi-person scenes
       d. Replace background
       e. Apply photometric / geometric transforms
    3. Save augmented images + YOLO-format labels
    """

    def __init__(self, seg_model, config: Optional[AugmentationConfig] = None):
        self.config = config or AugmentationConfig()
        self.extractor = GarmentExtractor(seg_model)
        self.bg_manager = BackgroundManager(self.config.background_dir)
        self.rng = np.random.RandomState(self.config.random_seed)

    # ------------------------------------------------------------------ #

    def augment_dataset(
        self,
        input_dir: Path,
        output_dir: Path,
    ) -> AugmentationReport:
        """Run the full augmentation pipeline."""
        input_dir = Path(input_dir)
        output_dir = Path(output_dir)

        img_out = output_dir / "images"
        lbl_out = output_dir / "labels"
        img_out.mkdir(parents=True, exist_ok=True)
        lbl_out.mkdir(parents=True, exist_ok=True)

        # Step 1: build garment bank
        bank_dir = Path(self.config.garment_bank_dir)
        garments = self.extractor.build_garment_bank(input_dir, bank_dir)
        if not garments:
            logger.warning("No garments extracted — check input images.")
            return AugmentationReport(0, 0, 0, str(output_dir))

        # Count source images
        exts = (".jpg", ".jpeg", ".png", ".webp")
        src_count = sum(1 for p in input_dir.iterdir() if p.suffix.lower() in exts)

        generated = 0
        target = src_count * self.config.num_augmented_per_image

        for aug_idx in range(target):
            try:
                img, labels = self._generate_one(garments, aug_idx)
                if img is None:
                    continue

                fname = f"aug_{aug_idx:05d}"
                cv2.imwrite(str(img_out / f"{fname}.png"), img)

                # Save YOLO-format labels
                with open(lbl_out / f"{fname}.txt", "w") as f:
                    for lbl in labels:
                        f.write(lbl + "\n")

                generated += 1
            except Exception as exc:
                logger.error("Augmentation %d failed: %s", aug_idx, exc)

        report = AugmentationReport(
            source_images=src_count,
            garments_extracted=len(garments),
            augmented_images_generated=generated,
            output_dir=str(output_dir),
        )
        logger.info(
            "Augmentation complete: %d images from %d sources (%d garments)",
            generated, src_count, len(garments),
        )

        # Save report
        with open(output_dir / "report.json", "w") as f:
            json.dump(report.__dict__, f, indent=2)

        return report

    # ------------------------------------------------------------------ #

    def _generate_one(
        self, garments: List[GarmentSample], idx: int
    ) -> tuple:
        """Generate one augmented image with labels."""
        cfg = self.config
        H, W = cfg.output_size

        # How many people in this scene
        n_people = (
            self.rng.randint(1, cfg.max_people_per_scene + 1)
            if cfg.enable_multi_person else 1
        )

        person_groups = []
        all_labels = []

        for _ in range(n_people):
            # Sample 1-3 garments for this "person"
            n_garments = self.rng.randint(1, min(4, len(garments) + 1))
            sampled_indices = self.rng.choice(len(garments), size=n_garments, replace=False)
            sampled = [garments[i] for i in sampled_indices]

            # Load RGBA images
            g_images = []
            g_labels = []
            for gs in sampled:
                g_img = cv2.imread(gs.image_path, cv2.IMREAD_UNCHANGED)
                if g_img is None or g_img.shape[2] != 4:
                    continue
                g_images.append(g_img)
                g_labels.append(gs.label)

            if not g_images:
                continue

            # Occlusion simulation
            if cfg.enable_occlusion and len(g_images) > 1:
                overlap = self.rng.uniform(*cfg.occlusion_ratio_range)
                occ_result = OcclusionSimulator.simulate(
                    g_images, g_labels,
                    canvas_size=(H, W),
                    overlap_ratio=overlap,
                    rng=self.rng,
                )
                person_groups.append([occ_result.composite])
                all_labels.extend(occ_result.layer_labels)
            else:
                person_groups.append(g_images)
                all_labels.extend(g_labels)

        if not person_groups:
            return None, []

        # Multi-person composition
        canvas_bgr, person_masks = SceneCompositor.compose_multi_person(
            person_groups, canvas_size=(H, W), rng=self.rng
        )

        # Background replacement
        if cfg.enable_background_replace:
            # Build combined foreground BGRA from canvas
            combined_alpha = np.zeros((H, W), dtype=np.uint8)
            for pm in person_masks:
                combined_alpha = np.clip(
                    combined_alpha.astype(int) + pm.astype(int) * 255, 0, 255
                ).astype(np.uint8)
            fg_bgra = cv2.merge([canvas_bgr[:,:,0], canvas_bgr[:,:,1],
                                  canvas_bgr[:,:,2], combined_alpha])
            canvas_bgr = self.bg_manager.replace_background(
                fg_bgra, target_size=(H, W), rng=self.rng
            )

        # Standard transforms
        if cfg.enable_transforms:
            canvas_bgr = ImageTransforms.apply_all(
                canvas_bgr,
                rng=self.rng,
                rotation_range=cfg.rotation_range,
                scale_range=cfg.scale_range,
                brightness_range=cfg.brightness_range,
                contrast_range=cfg.contrast_range,
                hue_range=cfg.hue_shift_range,
            )

        # YOLO-format labels (class_id cx cy w h — normalised)
        label_lines = []
        class_map = {
            "upper_garment": 0, "lower_garment": 1, "garment": 2,
            "shirt": 0, "t-shirt": 0, "jacket": 0, "coat": 0,
            "sweater": 0, "hoodie": 0, "blouse": 0, "tank_top": 0,
            "jeans": 1, "pants": 1, "shorts": 1, "skirt": 1,
            "dress": 2, "suit": 2, "other": 2,
        }
        for i, pm in enumerate(person_masks):
            ys, xs = np.where(pm > 0)
            if len(xs) == 0:
                continue
            x1, y1, x2, y2 = xs.min(), ys.min(), xs.max(), ys.max()
            cx = ((x1 + x2) / 2) / W
            cy = ((y1 + y2) / 2) / H
            bw = (x2 - x1) / W
            bh = (y2 - y1) / H
            lbl = all_labels[i] if i < len(all_labels) else "garment"
            cls_id = class_map.get(lbl, 2)
            label_lines.append(f"{cls_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")

        return canvas_bgr, label_lines
