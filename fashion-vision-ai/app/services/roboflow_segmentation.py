"""
Roboflow segmentation service — uses Roboflow API for DeepFashion2 segmentation.
"""

from roboflow import Roboflow
import numpy as np

class RoboflowSegmentationService:
    def __init__(self, api_key: str, project_name: str = "deepfashion2-m-10k-atr13", version: int = 2):
        self.rf = Roboflow(api_key=api_key)
        self.project = self.rf.workspace().project(project_name)
        self.model = self.project.version(version).model

    def segment(self, image: np.ndarray):
        # Convert numpy image to file or base64 as required by Roboflow
        # Roboflow expects file path or PIL Image, so save temp file or convert
        import tempfile
        import cv2
        import os
        from PIL import Image

        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
            cv2.imwrite(tmp.name, image)
            tmp_path = tmp.name

        try:
            result = self.model.predict(tmp_path, confidence=40, overlap=30)
            # result.json() gives predictions
            return result.json()
        finally:
            os.remove(tmp_path)
