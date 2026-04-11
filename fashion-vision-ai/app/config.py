"""
Application configuration loaded from environment variables.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()


def _env_truthy(name: str) -> bool:
    return os.getenv(name, "").strip().lower() in ("1", "true", "yes", "on")


# When True, YOLO / EfficientNet are not loaded (for Render and other small hosts).
# Render sets RENDER=true automatically; you can also set SKIP_LOCAL_ML=1 explicitly.
SKIP_LOCAL_ML = _env_truthy("SKIP_LOCAL_ML") or os.getenv(
    "RENDER", ""
).strip().lower() == "true"

# ── Paths ────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent
STATIC_DIR = BASE_DIR / "static"
UPLOAD_DIR = STATIC_DIR / "uploads"
CROPS_DIR = STATIC_DIR / "crops"
TEMPLATE_DIR = BASE_DIR / "app" / "templates"
MODELS_DIR = BASE_DIR / "models" / "weights"

# Create dirs on import
for d in [STATIC_DIR, UPLOAD_DIR, CROPS_DIR, TEMPLATE_DIR, MODELS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ── Model Settings ───────────────────────────────────────────────────────
SEGMENTATION_MODEL_PATH = os.getenv("SEGMENTATION_MODEL_PATH", "yolov8n-seg.pt")
CLASSIFICATION_MODEL_PATH = os.getenv(
    "CLASSIFICATION_MODEL_PATH", str(MODELS_DIR / "classifier.pth")
)
IMAGE_SIZE = int(os.getenv("IMAGE_SIZE", "640"))
CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", "0.35"))

# ── Clothing Categories ─────────────────────────────────────────────────
CLOTHING_CATEGORIES = [
    "shirt", "t-shirt", "jacket", "coat", "sweater",
    "hoodie", "jeans", "pants", "shorts", "dress",
    "skirt", "blouse", "suit", "tank_top", "other",
]
NUM_CLASSES = len(CLOTHING_CATEGORIES)

# ── COCO classes that relate to people / clothing ────────────────────────
# YOLOv8-seg COCO class indices we care about
PERSON_CLASS_ID = 0  # 'person' in COCO
RELEVANT_COCO_CLASSES = {
    0: "person",
    24: "backpack",
    25: "umbrella",
    26: "handbag",
    27: "tie",
    28: "suitcase",
}

# ── OpenRouter / Agent ───────────────────────────────────────────────────
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "qwen/qwen3.6-plus:free")
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1/chat/completions"

# ── Server ───────────────────────────────────────────────────────────────
HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "10000"))
DEBUG = os.getenv("DEBUG", "true").lower() == "true"
