
from __future__ import annotations

import logging

from fastapi import APIRouter, File, Request, UploadFile, HTTPException
from fastapi.responses import JSONResponse

from app.schemas import (
    ChatRequest,
    ChatResponse,
    HealthResponse,
    PredictionResponse,
    ShoppingLink,
)
from utils.image_utils import load_image_from_upload

logger = logging.getLogger(__name__)

router = APIRouter()

_LOCAL_ML_DISABLED_MSG = (
    "Local segmentation and classification are disabled on this deployment "
    "(lightweight server). Use /api/roboflow-segment with ROBOFLOW_API_KEY, "
    "or run the app locally with full ML dependencies and SKIP_LOCAL_ML unset."
)


@router.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check(request: Request):
    """Check whether models are loaded and the server is ready."""
    skip = getattr(request.app.state, "skip_local_ml", False)
    return HealthResponse(
        status="ok",
        segmentation_model_loaded=request.app.state.seg_service is not None,
        classification_model_loaded=request.app.state.cls_service is not None,
        local_ml_enabled=not skip,
    )


@router.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
async def predict(request: Request, file: UploadFile = File(...)):
    """
    Full pipeline: image → segmentation → classification.
    Shopping links are handled separately via /api/chat.
    """
    pipeline = getattr(request.app.state, "pipeline", None)
    if pipeline is None:
        return PredictionResponse(success=False, message=_LOCAL_ML_DISABLED_MSG)

    try:
        image = await load_image_from_upload(file)
    except ValueError as exc:
        return PredictionResponse(success=False, message=str(exc))

    result = await pipeline.run(image)
    return result


@router.post("/segment", tags=["Segmentation"])
async def segment_only(request: Request, file: UploadFile = File(...)):
    """Run segmentation only and return bounding boxes + class info."""
    seg = getattr(request.app.state, "seg_service", None)
    if seg is None:
        return JSONResponse(
            {"success": False, "error": _LOCAL_ML_DISABLED_MSG},
            status_code=503,
        )

    try:
        image = await load_image_from_upload(file)
    except ValueError as exc:
        return JSONResponse({"success": False, "error": str(exc)}, status_code=400)

    objects = seg.segment(image)
    return {
        "success": True,
        "num_objects": len(objects),
        "objects": [
            {
                "class_name": o.class_name,
                "confidence": round(o.confidence, 3),
                "bbox": o.bbox,
            }
            for o in objects
        ],
    }


@router.post("/roboflow-segment", tags=["Segmentation"])
async def roboflow_segment(request: Request, file: UploadFile = File(...)):
    """Run segmentation using Roboflow DeepFashion2 model via API."""
    try:
        image = await load_image_from_upload(file)
    except ValueError as exc:
        return JSONResponse({"success": False, "error": str(exc)}, status_code=400)

    roboflow_seg = getattr(request.app.state, "roboflow_seg_service", None)
    if roboflow_seg is None:
        raise HTTPException(
            status_code=503,
            detail="RoboflowSegmentationService not available. Set ROBOFLOW_API_KEY.",
        )

    result = roboflow_seg.segment(image)
    return result


@router.post("/chat", response_model=ChatResponse, tags=["Chat"])
async def chat(request: Request, body: ChatRequest):
    """
    AI Shopping Chat endpoint.

    Receives detected items, generates real shopping URLs dynamically,
    then asks the LLM to present them in a friendly conversational format.
    The LLM does NOT invent URLs — only uses the ones we provide.
    """
    agent = request.app.state.agent

    items_data = [
        {
            "label": item.label,
            "color": item.color,
            "pattern": item.pattern,
            "confidence": item.confidence,
        }
        for item in body.items
    ]

    message, links = await agent.chat(items_data, body.user_message)

    shopping_links = [ShoppingLink(**lnk) for lnk in links]

    return ChatResponse(
        success=True,
        message=message,
        shopping_links=shopping_links,
    )
