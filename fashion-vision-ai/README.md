# Multi-Object Fashion Segmentation & Agentic Shopping

> Occlusion-Aware Data Augmentation Pipeline with AI-Powered Shopping Recommendations

## 🎯 Overview

This project implements an end-to-end AI system that:

1. **Segments** multiple clothing items from fashion images using YOLOv8-seg
2. **Classifies** each garment into 15 categories using EfficientNet-B0
3. **Finds shopping links** using an AI agent (OpenRouter / Qwen 3.6 Plus)
4. **Augments training data** with a novel occlusion-aware pipeline

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure Environment

```bash
cp .env.example .env
# Edit .env and add your OpenRouter API key
```

### 3. Download Pretrained Models

```bash
python -m models.download_models
```

### 4. Start the Server

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### 5. Open the Web Interface

Navigate to **http://localhost:8000** in your browser.

## 📡 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Web upload interface |
| `/api/predict` | POST | Full pipeline (segment → classify → shop) |
| `/api/segment` | POST | Segmentation only |
| `/api/health` | GET | Health check |
| `/docs` | GET | Interactive API docs (Swagger) |

### Example API Call

```bash
curl -X POST http://localhost:8000/api/predict \
  -F "file=@your_fashion_image.jpg"
```

### Example Response

```json
{
  "success": true,
  "items": [
    {
      "item_id": 0,
      "label": "jacket",
      "confidence": 0.92,
      "color": "black",
      "pattern": "solid",
      "shopping_links": [
        {
          "title": "Black Leather Jacket",
          "url": "https://www.amazon.in/...",
          "platform": "Amazon",
          "price_range": "₹2000-₹5000"
        }
      ]
    }
  ],
  "num_items_detected": 3,
  "processing_time_ms": 845.2
}
```

## 📁 Project Structure

```
├── app/
│   ├── main.py              # FastAPI app + lifespan model loading
│   ├── config.py             # Environment configuration
│   ├── schemas.py            # Pydantic response models
│   ├── routes/
│   │   └── predict.py        # API endpoints
│   ├── services/
│   │   ├── segmentation.py   # YOLOv8-seg wrapper
│   │   ├── classification.py # EfficientNet-B0 classifier
│   │   ├── agent.py          # OpenRouter shopping agent
│   │   └── pipeline.py       # Full prediction pipeline
│   └── templates/
│       └── index.html        # Web upload interface
├── augmentation_pipeline/
│   ├── augmentor.py          # Main orchestrator
│   ├── garment_extractor.py  # Garment bank builder
│   ├── occlusion.py          # Occlusion simulator (core contribution)
│   ├── compositor.py         # Multi-person scene compositor
│   ├── background.py         # Background replacement
│   ├── transforms.py         # Geometric/photometric transforms
│   └── config.py             # Pipeline configuration
├── models/
│   ├── download_models.py    # Weight downloader
│   └── weights/              # Model weights directory
├── colab_notebook/
│   └── fashion_segmentation_finetuning.ipynb
├── utils/
│   └── image_utils.py        # Image processing utilities
├── static/                   # Served static files
├── requirements.txt
├── .env.example
└── README.md
```

## 🧪 Augmentation Pipeline (Research Contribution)

The occlusion-aware augmentation pipeline generates synthetic training data with:

- **Garment extraction** → builds a reusable garment bank from source images
- **Occlusion simulation** → layers garments with controlled overlap ratios (10-70%)
- **Multi-person composition** → places up to 4 people per scene
- **Background replacement** → real-world, gradient, clutter, or solid backgrounds
- **Standard transforms** → rotation, scale, flip, brightness, hue, perspective

## 📓 Colab Notebook

The notebook (`colab_notebook/fashion_segmentation_finetuning.ipynb`) includes:

- Complete fine-tuning for both segmentation and classification
- **10+ comparison plots**: loss curves, accuracy curves, confusion matrices, per-class accuracy, training dashboard
- Head-to-head comparison: **with augmentation vs without augmentation**
- Model export for deployment

## 🔑 Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `OPENROUTER_API_KEY` | (required) | Your OpenRouter API key |
| `OPENROUTER_MODEL` | `qwen/qwen3.6-plus:free` | LLM model for shopping agent |
| `SEGMENTATION_MODEL_PATH` | `yolov8n-seg.pt` | Path to segmentation weights |
| `CONFIDENCE_THRESHOLD` | `0.35` | Minimum detection confidence |
| `IMAGE_SIZE` | `640` | Input image size for segmentation |
