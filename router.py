import secrets
from fastapi import FastAPI, UploadFile, Form, HTTPException, Query, Depends
from fastapi.responses import StreamingResponse, HTMLResponse, JSONResponse
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from fastapi.staticfiles import StaticFiles
import glob
import io
import os
import re
import time
import uuid
import numpy as np
from PIL import Image, ImageOps
from yoloPostprocessUtils import (
    crop_pillow_img_from_bbox,
    extract_highest_conf_bbox,
    convert_cornerWidthHeight_to_cornerCords,
)
from yolo_onnx import YOLO_OnnxRuntime
from db import init_log_db, insert_inference_log, query_inference_logs, render_logs_table, utc_now_sql


def find_latest_onnx(directory: str, mode: str) -> str:
    pattern = os.path.join(directory, f"staves_detector_{mode}_onnx_v*.onnx")
    matches = glob.glob(pattern)
    if not matches:
        raise FileNotFoundError(f"No ONNX model found for mode '{mode}' in '{directory}'")

    def version_num(path: str) -> int:
        match = re.search(r"_v(\d+)\.onnx$", path)
        return int(match.group(1)) if match else -1

    return max(matches, key=version_num)


app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
basic_auth = HTTPBasic()

onnx_localizer_path = find_latest_onnx("models", "localizer")
onnx_counter_path = find_latest_onnx("models", "finetune")
print(f"Localizer model: {onnx_localizer_path}")
print(f"Counter model:   {onnx_counter_path}")


@app.on_event("startup")
async def startup_event() -> None:
    init_log_db()


def require_logs_auth(credentials: HTTPBasicCredentials = Depends(basic_auth)) -> None:
    expected_user = os.getenv("LOGS_BASIC_AUTH_USER")
    expected_password = os.getenv("LOGS_BASIC_AUTH_PASSWORD")

    if not expected_user or not expected_password:
        return

    username_ok = secrets.compare_digest(credentials.username, expected_user)
    password_ok = secrets.compare_digest(credentials.password, expected_password)

    if username_ok and password_ok:
        return

    raise HTTPException(
        status_code=401,
        detail="Unauthorized",
        headers={"WWW-Authenticate": "Basic"},
    )


def get_model_info() -> dict:
    localizer_name = os.path.basename(onnx_localizer_path)
    counter_name = os.path.basename(onnx_counter_path)

    def version(name: str) -> str:
        match = re.search(r"_v(\d+)\.onnx$", name)
        return match.group(1) if match else "unknown"

    return {
        "localizer": {"file": localizer_name, "version": version(localizer_name)},
        "counter": {"file": counter_name, "version": version(counter_name)},
    }


def build_log_record(
    *,
    inference_id: str,
    image_width: int | None,
    image_height: int | None,
    latency_ms: float,
    predicted_count: int | None,
    conf_threshold: float | None,
    iou_threshold: float | None,
    conf_scores,
    error_message: str = "",
) -> dict:
    model_info = get_model_info()
    scores = np.array(conf_scores or [], dtype=float)
    low_conf_threshold = 0.50

    if scores.size:
        mean_conf = round(float(np.mean(scores)), 4)
        median_conf = round(float(np.median(scores)), 4)
        min_conf = round(float(np.min(scores)), 4)
        max_conf = round(float(np.max(scores)), 4)
        low_conf_count = int(np.sum(scores < low_conf_threshold))
    else:
        mean_conf = None
        median_conf = None
        min_conf = None
        max_conf = None
        low_conf_count = 0

    return {
        "inference_id": inference_id,
        "created_at": utc_now_sql(),
        "image_width": image_width,
        "image_height": image_height,
        "latency_ms": round(latency_ms, 2),
        "predicted_count": predicted_count,
        "conf_threshold": conf_threshold,
        "iou_threshold": iou_threshold,
        "mean_confidence": mean_conf,
        "median_confidence": median_conf,
        "min_confidence": min_conf,
        "max_confidence": max_conf,
        "low_confidence_detections": low_conf_count,
        "localizer_version": model_info["localizer"]["version"],
        "counter_version": model_info["counter"]["version"],
        "error_message": error_message,
    }


@app.get("/")
async def read_root() -> HTMLResponse:
    with open("static/index.html", "r", encoding="utf-8") as file_obj:
        return HTMLResponse(content=file_obj.read())


@app.get("/logs/")
async def logs_page(_: None = Depends(require_logs_auth)) -> HTMLResponse:
    with open("static/logs.html", "r", encoding="utf-8") as file_obj:
        return HTMLResponse(content=file_obj.read())


@app.get("/model_version/")
async def model_version() -> JSONResponse:
    return JSONResponse(get_model_info())


@app.get("/logs_table/", response_class=HTMLResponse)
async def return_logs_table(
    time_range: str = Query("24h"),
    error_filter: str = Query("all"),
    _: None = Depends(require_logs_auth),
) -> HTMLResponse:
    rows = query_inference_logs(time_range=time_range, error_filter=error_filter)
    return HTMLResponse(render_logs_table(rows))


@app.post("/predictOnnx/")
async def count_staves(
    file: UploadFile,
    conf_thresh: float = Form(0.45),
    iou_thresh: float = Form(0.60),
):
    inference_id = str(uuid.uuid4())
    image_width = None
    image_height = None
    start_time = time.perf_counter()

    try:
        image_bytes = await file.read()
        image_stream = io.BytesIO(image_bytes)
        image = Image.open(image_stream)
        image.load()
        image = ImageOps.exif_transpose(image)
        image_width, image_height = image.size

        print("running through first stage localizer model")
        localizer_results = YOLO_OnnxRuntime(
            onnx_localizer_path,
            image,
            confidence_thres=conf_thresh,
            iou_thres=iou_thresh,
        )
        _, localizer_boxes, localizer_conf_scores, _ = localizer_results.main()
        if not localizer_boxes:
            raise ValueError("No pallet region detected by the localizer model.")

        highest_conf_localizer_bbox, _ = extract_highest_conf_bbox(localizer_boxes, localizer_conf_scores)
        bbox_corner_cords = convert_cornerWidthHeight_to_cornerCords(highest_conf_localizer_bbox.tolist())
        cropped_image = crop_pillow_img_from_bbox(bbox_corner_cords, image)

        print("running through second stage staves counter model")
        counter_results = YOLO_OnnxRuntime(
            onnx_counter_path,
            cropped_image,
            confidence_thres=conf_thresh,
            iou_thres=iou_thresh,
        )
        counter_annotated_arr, counter_boxes, counter_conf_scores, _ = counter_results.main()
        staves_count = len(counter_boxes)
        counter_annotated_img = Image.fromarray(counter_annotated_arr)

        latency_ms = (time.perf_counter() - start_time) * 1000
        insert_inference_log(
            build_log_record(
                inference_id=inference_id,
                image_width=image_width,
                image_height=image_height,
                latency_ms=latency_ms,
                predicted_count=staves_count,
                conf_threshold=conf_thresh,
                iou_threshold=iou_thresh,
                conf_scores=counter_conf_scores,
            )
        )

        output_buffer = io.BytesIO()
        counter_annotated_img.save(output_buffer, format="JPEG")
        output_buffer.seek(0)
        return StreamingResponse(
            output_buffer,
            media_type="image/jpeg",
            headers={"staves-count": str(staves_count)},
        )

    except Exception as exc:
        latency_ms = (time.perf_counter() - start_time) * 1000
        insert_inference_log(
            build_log_record(
                inference_id=inference_id,
                image_width=image_width,
                image_height=image_height,
                latency_ms=latency_ms,
                predicted_count=None,
                conf_threshold=conf_thresh,
                iou_threshold=iou_thresh,
                conf_scores=[],
                error_message=str(exc),
            )
        )
        raise HTTPException(status_code=500, detail=str(exc))
