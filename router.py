from datetime import datetime
from fastapi import FastAPI, UploadFile, Form, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
import glob
import io
import os
import re
import time
import uuid
from PIL import Image, ImageOps
from yoloPostprocessUtils import crop_pillow_img_from_bbox, extract_highest_conf_bbox, convert_cornerWidthHeight_to_cornerCords
from fastapi.responses import HTMLResponse, JSONResponse
import pandas as pd
import numpy as np
from yolo_onnx import YOLO_OnnxRuntime


def find_latest_onnx(directory: str, mode: str) -> str:
    pattern = os.path.join(directory, f"staves_detector_{mode}_onnx_v*.onnx")
    matches = glob.glob(pattern)
    if not matches:
        raise FileNotFoundError(f"No ONNX model found for mode '{mode}' in '{directory}'")
    # extract version number from filename and return the highest
    def version_num(path):
        m = re.search(r"_v(\d+)\.onnx$", path)
        return int(m.group(1)) if m else -1
    return max(matches, key=version_num)


app = FastAPI()
#mount static folder, for acess to all things inside static folder
app.mount("/static", StaticFiles(directory="static"), name="static")

onnx_localizer_path = find_latest_onnx("models", "localizer")
onnx_counter_path   = find_latest_onnx("models", "finetune")
print(f"Localizer model: {onnx_localizer_path}")
print(f"Counter model:   {onnx_counter_path}")

#create pandas table to track inference results
table_df = pd.DataFrame(columns=["img_name", "count", "conf_threshold"])
inference_log_df = pd.DataFrame(columns=[
    "inference_id",
    "timestamp",
    "img_name",
    "image_width",
    "image_height",
    "latency_ms",
    "predicted_count",
    "conf_threshold",
    "iou_threshold",
    "mean_confidence",
    "median_confidence",
    "min_confidence",
    "max_confidence",
    "low_confidence_detections",
    "localizer_version",
    "counter_version",
    "error_message",
])

#routes

#html page for root
@app.get("/")
async def read_root():
    with open("static/index.html", "r", encoding="utf-8") as f:
        html_content = f.read()
    return HTMLResponse(content=html_content)

#running prediction with ONNXRuntime
@app.post("/predictOnnx/")
async def count_staves(file: UploadFile,
                        conf_thresh: float = Form(0.45),
                        iou_thresh: float = Form(0.60)):

    global table_df
    inference_id = str(uuid.uuid4())
    img_name = file.filename or "uploaded_image"
    image_width = None
    image_height = None
    start_time = time.perf_counter()

    try:
        image_bytes = await file.read()
        #this is needed for pillow
        image_stream = io.BytesIO(image_bytes)
        image = Image.open(image_stream)
        image.load()
        image = ImageOps.exif_transpose(image)
        image_width, image_height = image.size

        #run image thru first stage staves localizer
        print("running through first stage localizer model")
        localizer_results = YOLO_OnnxRuntime(onnx_localizer_path, image, confidence_thres=conf_thresh, iou_thres=iou_thresh)
        _, localzier_boxes, localizer_conf_scores, _ = localizer_results.main()
        if not localzier_boxes:
            raise ValueError("No pallet region detected by the localizer model.")
        highest_conf_localizer_bbox, _ = extract_highest_conf_bbox(localzier_boxes, localizer_conf_scores)
        bbox_corner_cords = convert_cornerWidthHeight_to_cornerCords(highest_conf_localizer_bbox.tolist())
        cropped_image = crop_pillow_img_from_bbox(bbox_corner_cords, image)
        
        #second stage staves counter
        print("running through second stage staves counter model")
        counter_results = YOLO_OnnxRuntime(onnx_counter_path, cropped_image, confidence_thres=conf_thresh, iou_thres=iou_thresh)
        counter_annotated_arr, counter_boxes, counter_conf_scores, _ = counter_results.main()
        staves_count = len(counter_boxes)
        
        #counter_results.main() returns numpy, need to conver to pil for further processing
        counter_annotated_img = Image.fromarray(counter_annotated_arr)
        
        # #store results to pandas table
        if img_name in table_df["img_name"].values:
            table_df.loc[table_df["img_name"]==img_name, "count"]          = staves_count
            table_df.loc[table_df["img_name"]==img_name, "conf_threshold"] = conf_thresh
        else:
            table_df = pd.concat([table_df, pd.DataFrame([{"img_name": img_name, "count": staves_count, "conf_threshold": conf_thresh}])], ignore_index=True)

        latency_ms = (time.perf_counter() - start_time) * 1000
        append_inference_log(
            inference_id=inference_id,
            img_name=img_name,
            image_width=image_width,
            image_height=image_height,
            latency_ms=latency_ms,
            predicted_count=staves_count,
            conf_threshold=conf_thresh,
            iou_threshold=iou_thresh,
            conf_scores=counter_conf_scores,
        )
        
        #final img to buffer, send back to index page
        output_buffer = io.BytesIO()
        counter_annotated_img.save(output_buffer, format="JPEG")
        output_buffer.seek(0)
        
        return StreamingResponse(output_buffer, media_type="image/jpeg", headers={"staves-count": str(staves_count)})

    except Exception as exc:
        latency_ms = (time.perf_counter() - start_time) * 1000
        append_inference_log(
            inference_id=inference_id,
            img_name=img_name,
            image_width=image_width,
            image_height=image_height,
            latency_ms=latency_ms,
            predicted_count=None,
            conf_threshold=conf_thresh,
            iou_threshold=iou_thresh,   
            conf_scores=[],
            error_message=str(exc),
        )
        raise HTTPException(status_code=500, detail=str(exc))

def get_model_info():
    localizer_name = os.path.basename(onnx_localizer_path)
    counter_name = os.path.basename(onnx_counter_path)

    def version(name):
        m = re.search(r"_v(\d+)\.onnx$", name)
        return m.group(1) if m else "unknown"

    return {
        "localizer": {"file": localizer_name, "version": version(localizer_name)},
        "counter": {"file": counter_name, "version": version(counter_name)}
    }


def append_inference_log(
    *,
    inference_id: str,
    img_name: str,
    image_width: int | None,
    image_height: int | None,
    latency_ms: float,
    predicted_count: int | None,
    conf_threshold: float | None = None,
    iou_threshold: float | None = None,
    conf_scores,
    error_message: str = "",
):
    global inference_log_df

    model_info = get_model_info()
    scores = np.array(conf_scores or [], dtype=float)
    #assume anything below 0.5 is low conf for now, tracking conf thresh used for inference gives us better picture
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

    inference_log_df = pd.concat([
        inference_log_df,
        pd.DataFrame([{
            "inference_id": inference_id,
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "img_name": img_name,
            "image_width": image_width,
            "image_height": image_height,
            "latency_ms": round(latency_ms, 2),
            "predicted_count": predicted_count,
            "iou_threshold": iou_threshold,
            "conf_threshold": conf_threshold,
            "mean_confidence": mean_conf,
            "median_confidence": median_conf,
            "min_confidence": min_conf,
            "max_confidence": max_conf,
            "low_confidence_detections": low_conf_count,
            "localizer_version": model_info["localizer"]["version"],
            "counter_version": model_info["counter"]["version"],
            "error_message": error_message,
        }])
    ], ignore_index=True)


@app.get("/model_version/")
async def model_version():
    return JSONResponse(get_model_info())


#returns pandas table storing results as a html string
@app.get("/table/", response_class=HTMLResponse)
async def return_table():
    global table_df
    return HTMLResponse(table_df.to_html())


@app.get("/logs/")
async def logs_page():
    with open("static/logs.html", "r", encoding="utf-8") as f:
        html_content = f.read()
    return HTMLResponse(content=html_content)


@app.get("/logs_table/", response_class=HTMLResponse)
async def return_logs_table():
    global inference_log_df
    if inference_log_df.empty:
        return HTMLResponse("<p class='table-empty'>No inference logs yet.</p>")

    logs_df = inference_log_df.iloc[::-1]
    return HTMLResponse(logs_df.to_html(index=False))

#clears the pandas table
@app.post("/clear_table/")
async def clear_table():
    global table_df
    # reinitialize with the same columns
    table_df = pd.DataFrame(columns=["img_name", "count", "conf_threshold"])
    return JSONResponse({"status": "cleared"})
