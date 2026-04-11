"""
FastAPI application entry-point.

* Models are loaded once at startup via a lifespan handler.
* A premium frontend webpage is served at GET /.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from app.config import SKIP_LOCAL_ML, STATIC_DIR, TEMPLATE_DIR
from app.routes.predict import router as predict_router

import os

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)


# ── Lifespan: load models once ──────────────────────────────────────────


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load services at startup. Local YOLO/classifier are optional for small hosts."""
    from app.services.agent import ShoppingAgent

    logger.info("🚀  Starting application …")
    app.state.skip_local_ml = SKIP_LOCAL_ML
    app.state.agent = ShoppingAgent()

    if SKIP_LOCAL_ML:
        logger.info(
            "Lightweight mode: skipping local YOLO/EfficientNet "
            "(SKIP_LOCAL_ML or RENDER=true)."
        )
        app.state.seg_service = None
        app.state.cls_service = None
        app.state.pipeline = None
    else:
        from app.services.classification import ClassificationService
        from app.services.pipeline import PredictionPipeline
        from app.services.segmentation import SegmentationService

        seg_service = SegmentationService()
        cls_service = ClassificationService()
        pipeline = PredictionPipeline(seg_service, cls_service, app.state.agent)
        app.state.seg_service = seg_service
        app.state.cls_service = cls_service
        app.state.pipeline = pipeline
        logger.info("Local segmentation and classification models loaded.")

    roboflow_api_key = os.getenv("ROBOFLOW_API_KEY", "").strip()
    if roboflow_api_key:
        from app.services.roboflow_segmentation import RoboflowSegmentationService

        try:
            app.state.roboflow_seg_service = RoboflowSegmentationService(
                api_key=roboflow_api_key
            )
            logger.info("RoboflowSegmentationService ready.")
        except Exception as exc:
            logger.warning("RoboflowSegmentationService failed to init: %s", exc)
            app.state.roboflow_seg_service = None
    else:
        app.state.roboflow_seg_service = None
        logger.warning("ROBOFLOW_API_KEY not set — /api/roboflow-segment unavailable.")

    logger.info("✅ Server is ready.")
    try:
        yield
    finally:
        logger.info("🛑  Shutting down …")


# ── App factory ──────────────────────────────────────────────────────────

app = FastAPI(
    title="Fashion Segmentation & Shopping API",
    description=(
        "Multi-object fashion segmentation with occlusion-aware augmentation "
        "and AI-powered shopping recommendations."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Static files (crops, uploads)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# Templates
templates = Jinja2Templates(directory=str(TEMPLATE_DIR))

# API routes
app.include_router(predict_router, prefix="/api")


# ── Frontend ─────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse, tags=["Frontend"])
async def index(request: Request):
    """Serve the upload webpage."""
    return templates.TemplateResponse("index.html", {"request": request})
