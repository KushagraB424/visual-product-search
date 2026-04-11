# Fashion Vision AI

**Multi-object fashion segmentation, garment classification, and an AI shopping assistant** (OpenRouter). Includes an occlusion-aware training augmentation pipeline and Colab fine-tuning notebooks.

---

## What it does

1. **Segment** clothing regions (local **YOLOv8-seg** or cloud **Roboflow** instance segmentation).
2. **Classify** each crop into clothing categories with **EfficientNet-B0** (local pipeline only).
3. **Shopping assistant**: builds real search URLs per item and optionally wraps them with an LLM via **OpenRouter** (`/api/chat`).

The web UI at `/` uploads an image, runs **`/api/predict`**, shows detected items, and opens the chat panel with recommendations.

---

## Run modes

| Mode | When | Segmentation | Classification |
|------|------|----------------|----------------|
| **Full local ML** | Default locally; `SKIP_LOCAL_ML` unset and `RENDER` unset | YOLOv8-seg | EfficientNet-B0 |
| **Lightweight / cloud** | `SKIP_LOCAL_ML=true` **or** `RENDER=true` (e.g. Render) | Roboflow API if `ROBOFLOW_API_KEY` is set | Heuristic color/pattern on bbox crops only |

When local models are off but **Roboflow** is configured, **`/api/predict` still works**: it calls Roboflow, maps results to the same JSON shape as the local pipeline, and saves bbox crops for the UI.

---

## Quick start (full local stack)

```bash
cd fashion-vision-ai   # inner app directory containing app/, utils/, etc.
python -m venv .venv
.venv\Scripts\activate   # Windows
# source .venv/bin/activate  # macOS/Linux

pip install -r requirements.txt
cp .env.example .env
# Set OPENROUTER_API_KEY (optional, for LLM chat). Download weights if needed:
python -m models.download_models

uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Open **http://localhost:8000**.

---

## Deploy on Render

The repo includes a **Blueprint** at the repository root: `render.yaml` (service **`rootDir`**: inner `fashion-vision-ai` folder).

- **Build:** `pip install -r requirements-render.txt` (no PyTorch / Ultralytics; smaller image).
- **Start:** `uvicorn app.main:app --host 0.0.0.0 --port $PORT`
- **Required for segmentation in production:** set **`ROBOFLOW_API_KEY`** in the Render dashboard.
- **Optional:** `OPENROUTER_API_KEY` for conversational shopping messages in `/api/chat`.

`SKIP_LOCAL_ML=true` is set in the blueprint so the server does not load YOLO or the classifier. Render also sets **`RENDER=true`**, which alone enables lightweight mode if you omit `SKIP_LOCAL_ML`.

---

## Environment variables

| Variable | Description |
|----------|-------------|
| `OPENROUTER_API_KEY` | OpenRouter API key for the shopping chat LLM (optional; static link fallback if unset). |
| `OPENROUTER_MODEL` | Model id (default: `qwen/qwen3.6-plus:free`). |
| `ROBOFLOW_API_KEY` | Roboflow API key; enables cloud segmentation and powers `/api/predict` when local ML is disabled. |
| `SKIP_LOCAL_ML` | If `true` / `1` / `yes`, do not load YOLO or EfficientNet. |
| `RENDER` | Set to `true` on Render; treated like lightweight mode for local ML (same as above). |
| `SEGMENTATION_MODEL_PATH` | Local YOLO weights path (default: `yolov8n-seg.pt`). |
| `CLASSIFICATION_MODEL_PATH` | Local classifier weights under `models/weights/`. |
| `CONFIDENCE_THRESHOLD` | Local YOLO confidence (default `0.35`). |
| `IMAGE_SIZE` | Local YOLO input size (default `640`). |
| `HOST` / `PORT` | Server bind (default `0.0.0.0` / `8000`). |

---

## API

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Web UI |
| `/api/predict` | POST | `multipart/form-data` with `file` — full pipeline locally, or Roboflow + adapter when lightweight |
| `/api/segment` | POST | Local YOLO segmentation only (503 if local ML off) |
| `/api/roboflow-segment` | POST | Raw Roboflow JSON (503 if no API key) |
| `/api/chat` | POST | Shopping assistant from detected items |
| `/api/health` | GET | Status; includes `local_ml_enabled`, model loaded flags |
| `/docs` | GET | Swagger UI |

Example:

```bash
curl -s -X POST http://localhost:8000/api/predict -F "file=@photo.jpg"
```

Example success shape (abbreviated):

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
      "bbox": [120.0, 80.0, 400.0, 520.0],
      "crop_path": "/static/crops/jacket_abc123.png",
      "shopping_links": []
    }
  ],
  "num_items_detected": 1,
  "processing_time_ms": 845.2,
  "image_width": 1024,
  "image_height": 768,
  "message": "Detected 1 clothing item(s)."
}
```

Shopping copy is produced via **`/api/chat`** using programmatic URLs (LLM optional).

---

## Project layout (app folder)

```
├── app/
│   ├── main.py                 # FastAPI, lifespan, optional ML + Roboflow
│   ├── config.py
│   ├── schemas.py
│   ├── routes/predict.py       # /api/predict, segment, chat, health, roboflow
│   ├── services/
│   │   ├── segmentation.py     # YOLOv8-seg
│   │   ├── classification.py   # EfficientNet-B0
│   │   ├── pipeline.py         # Local segment → classify
│   │   ├── roboflow_segmentation.py
│   │   ├── roboflow_prediction_adapter.py  # Roboflow JSON → PredictionResponse
│   │   └── agent.py            # OpenRouter + shopping URLs
│   └── templates/index.html
├── utils/image_utils.py
├── models/download_models.py
├── augmentation_pipeline/      # Occlusion-aware data augmentation (research)
├── colab_notebook/
├── requirements.txt            # Full stack (torch, ultralytics, timm, …)
├── requirements-render.txt     # Slim deps for Render
└── README.md
```

At the **repository root** (parent of this folder), `render.yaml` defines the Render web service and points **`rootDir`** at this inner `fashion-vision-ai` directory.

---

## Augmentation pipeline

The `augmentation_pipeline/` package implements garment extraction, occlusion simulation, multi-person composition, and background replacement for synthetic training data. See module docstrings and configs inside that directory.

---

## Colab

`colab_notebook/` contains notebooks for segmentation and classifier fine-tuning, comparisons, and export for deployment.

---

## License

Copyright (c) 2026 Kushagra Gupta

Permission is hereby granted, free of charge, to any person obtaining a copy
