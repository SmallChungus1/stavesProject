# stavesProject

## Project overview

`stavesProject` is a two-stage wood stave detection and counting system built with YOLO object detection and ONNX inference. It detects the stave region in an image, crops to that region, then counts staves inside the crop.

This repo includes training scripts (`train.py`), benchmark/export scripts (`scripts/benchmark_and_register.py`), and a FastAPI + web UI app (`router.py`, `static/index.html`) for live inference.


## Problem solved

- Input: photos of stacked wooden staves (planks)
- Output: per-image stave count and annotated result image
- Uses a localizer model to find the relevant region (1 box) then a counter model to detect each stave inside that crop

### Why this is useful

- Practical for inventory management, automated mill counting, and quality control in wood processing
- Avoids manual visual counting in noisy, variable-background images


## Key modeling design decisions

1. Two-stage pipeline
   - Stage 1: `localizer` model (single bbox). Goal: reduce false positives and focus the counter on the relevant region.
   - Stage 2: `finetune` counter model (staves detection)

2. Single-class YOLO labels (`staves_region` for localizer, `wood` / `stave` for counter)

3. ONNX for production inference
   - `yolo_onnx.py` handles preprocessing, inference, NMS, postprocess
   - backend endpoint `POST /predictOnnx/`

4. Versioning by filename: `staves_detector_{mode}_onnx_v{N}.onnx`
   - `router.py` picks latest via glob + regex
   - UI displays localizer / counter model versions in header


## End-to-end system flow

1. User uploads image(s) in web app (`static/index.html`)
2. `POST /predictOnnx/` in `router.py`
3. Stage 1 localizer inference
4. Crop image to localizer bbox
5. Stage 2 counter inference on cropped image
6. Send annotated image back and header `staves-count`
7. Show result cards + log table in UI


## How to run

### Prerequisites

- Python 3.12
- `uv` or `pip install -r requirements.txt` (if relevant)
- `.env` with `MLFLOW_TRACKING_URI`, `MLFLOW_TRACKING_USERNAME`, `MLFLOW_TRACKING_PASSWORD` for benchmark/email features

### Train

```bash
uv run python train.py --model yolo11n --mode pretrain
uv run python train.py --model yolo11n --mode localizer
uv run python train.py --model yolo11s --mode finetune
```

### Benchmark + register ONNX

```bash
uv run python scripts/benchmark_and_register.py --model yolo11s --mode finetune
```

### Download registered ONNX (optionally)

```bash
uv run python scripts/download_registered_model.py --mode finetune
uv run python scripts/download_registered_model.py --mode finetune --version 3
```

### Run server

```bash
uv run uvicorn router:app --reload --port 8000
```

Open `http://localhost:8000/`.


## Model versions in UI

- `router.py` finds latest ONNX files in `models/` using a filename pattern
- `GET /model_version/` returns JSON with localizer and counter model names and versions
- Frontend JS in `static/index.html` loads this into the header text


## Dataset structure

- `pretrain_dataset/` for initial pretraining
- `finetune_dataset/staves_localizer/` and `finetune_dataset/stave_count/` for final models
- Each has `images/train`, `images/val`, same shape labels in YOLO format `class x_center y_center w h`

## Attribution

- I wrote all code in this repository (`train.py`, `router.py`, `yolo_onnx.py`, `yoloPostprocessUtils.py`, `scripts/*`, `static/*`).
- Libraries used: Ultralytics YOLO, OpenCV, onnxruntime, fastapi, pandas, PIL


## Video walkthrough notes (Tubi builders format)

In your 3-5 minute video, cover:
- Problem: automated stave counting from photos
- Data + labels: single-class bounding boxes
- Model design: two-stage localizer + counter
- End-to-end flow: upload → localizer → crop → counter → annotation → display
- Results: number of staves and logs
- New feature: model version reporting in UI
- What you wrote: all repo code; mention adapted parts are from ultralytics library and standard ASR scaffolding


## Optional: academic reference

- No paper published for this project (N/A). If you publish, include link here.


