"""
Pydantic response schemas for the API.
"""

from __future__ import annotations

from typing import List, Optional
from pydantic import BaseModel, Field


class ShoppingLink(BaseModel):
    """A single shopping link suggestion."""
    title: str = Field(..., description="Product title")
    url: str = Field(..., description="Product URL")
    platform: str = Field(..., description="E-commerce platform name")
    price_range: Optional[str] = Field(None, description="Estimated price range")


class PredictedItem(BaseModel):
    """A single detected & classified clothing item."""
    item_id: int = Field(..., description="Unique item index in this prediction")
    label: str = Field(..., description="Clothing category label")
    confidence: float = Field(..., ge=0, le=1, description="Classification confidence")
    color: Optional[str] = Field(None, description="Dominant color")
    pattern: Optional[str] = Field(None, description="Detected pattern type")
    bbox: List[float] = Field(default_factory=list, description="[x1,y1,x2,y2]")
    crop_path: Optional[str] = Field(None, description="URL path to cropped image")
    shopping_links: List[ShoppingLink] = Field(
        default_factory=list, description="Shopping suggestions"
    )


class PredictionResponse(BaseModel):
    """Full prediction response."""
    success: bool = True
    items: List[PredictedItem] = Field(default_factory=list)
    num_items_detected: int = 0
    processing_time_ms: float = 0.0
    image_width: int = 0
    image_height: int = 0
    message: str = ""


class HealthResponse(BaseModel):
    """Health-check response."""
    status: str = "ok"
    segmentation_model_loaded: bool = False
    classification_model_loaded: bool = False
    local_ml_enabled: bool = True


# ── Chat Widget Schemas ──────────────────────────────────────────────────


class ChatItem(BaseModel):
    """Simplified item data sent from the frontend to the chat endpoint."""
    label: str
    confidence: float
    color: Optional[str] = None
    pattern: Optional[str] = None
    crop_path: Optional[str] = None


class ChatRequest(BaseModel):
    """Request body for the /api/chat endpoint."""
    items: List[ChatItem] = Field(..., description="Detected items from prediction")
    user_message: Optional[str] = Field(
        None, description="Optional follow-up message from the user"
    )


class ChatResponse(BaseModel):
    """Response from the AI shopping chat."""
    success: bool = True
    message: str = Field("", description="The AI assistant's conversational response")
    shopping_links: List[ShoppingLink] = Field(
        default_factory=list,
        description="Dynamically generated shopping links (for frontend rendering)",
    )
