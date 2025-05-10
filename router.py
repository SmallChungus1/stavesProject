from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
import io
from PIL import Image
from ultralytics import YOLO
import numpy as np
import cv2 
from yoloPostprocessUtils import crop_pillow_img_from_bbox, draw_detections_on_img, extract_highest_conf_bbox, convert_cornerWidthHeight_to_cornerCords
from fastapi.responses import HTMLResponse, JSONResponse
import torch
import pandas as pd
#entry point for onnx inferences
from yolo_onnx import YOLO_OnnxRuntime


device_name = "cuda" if torch.cuda.is_available() else "mps" if  torch.backends.mps.is_available() else "cpu"
print(f"using device {device_name}")
device = torch.device(device_name)

app = FastAPI()
#mount static folder, for acess to all things inside static folder
app.mount("/static", StaticFiles(directory="static"), name="static")

#load model
staves_localzier_model = YOLO("models/localizerModel.pt")
staves_counter_model = YOLO("models/counterModel.pt")

#create pandas table to track inference results
table_df = pd.DataFrame(columns=["img_name", "count"])

#routes

#html page for root
@app.get("/")
async def read_root():
    with open("static/index.html", "r", encoding="utf-8") as f:
        html_content = f.read()
    return HTMLResponse(content=html_content)

@app.post("/predict/")
async def count_staves(file: UploadFile,
                        conf_thresh: float = Form(0.25),
                        iou_thresh: float = Form(0.60)):

    global table_df
    image_bytes = await file.read()
    #this is needed for pillow
    image_stream = io.BytesIO(image_bytes)
    image = Image.open(image_stream)
    image.load()

    #run image thru first stage staves localizer
    print("running through first stage localizer model")
    localizer_detection = staves_localzier_model.predict(image, half=True, max_det=1, device=device_name)
    bbox_corner_cords = localizer_detection[0].boxes.xyxy.cpu().detach().numpy()
    cropped_image = crop_pillow_img_from_bbox(bbox_corner_cords, image)

    #second stage staves counter
    print("running through second stage staves counter model")
    counter_detections = staves_counter_model(cropped_image, half=True, max_det=9999, device=device_name, conf=conf_thresh, iou=iou_thresh)
    counter_detections = counter_detections[0]

    annotated_img, staves_count = draw_detections_on_img(counter_detections, cropped_image)

    #store results to pandas table
    img_name = file.filename
    if img_name in table_df["img_name"].values:
        table_df.loc[table_df["img_name"]==img_name, "count"] = staves_count
    else:
        table_df = table_df._append({"img_name": img_name, "count": staves_count}, ignore_index=True)
    
    #final img to buffer, send back to index page
    output_buffer = io.BytesIO()
    annotated_img.save(output_buffer, format="JPEG")
    output_buffer.seek(0)
    
    return StreamingResponse(output_buffer, media_type="image/jpeg", headers={"staves-count": str(staves_count)})

#running prediction with ONNXRuntime
@app.post("/predictOnnx/")
async def count_staves(file: UploadFile,
                        conf_thresh: float = Form(0.25),
                        iou_thresh: float = Form(0.60)):

    global table_df
    image_bytes = await file.read()
    #this is needed for pillow
    image_stream = io.BytesIO(image_bytes)
    image = Image.open(image_stream)
    image.load()

    onnx_localizer_path = ""
    onnx_counter_path = ""

    #run image thru first stage staves localizer
    print("running through first stage localizer model")
    _, localzier_boxes, localizer_conf_scores, _ = YOLO_OnnxRuntime(onnx_localizer_path, image, confidence_thres=conf_thresh, iou_thres=iou_thresh)
    highest_conf_localizer_bbox, _ = extract_highest_conf_bbox(localzier_boxes, localizer_conf_scores)
    bbox_corner_cords = highest_conf_localizer_bbox[0].cpu().detach().numpy()
    bbox_corner_cords = bbox_corner_cords[0]
    cropped_image = crop_pillow_img_from_bbox(bbox_corner_cords, image)

    #second stage staves counter
    print("running through second stage staves counter model")
    counter_detections = staves_counter_model(cropped_image, half=True, max_det=9999, device=device_name, conf=conf_thresh, iou=iou_thresh)
    counter_detections = counter_detections[0]

    annotated_img, staves_count = draw_detections_on_img(counter_detections, cropped_image)

    #store results to pandas table
    img_name = file.filename
    if img_name in table_df["img_name"].values:
        table_df.loc[table_df["img_name"]==img_name, "count"] = staves_count
    else:
        table_df = table_df._append({"img_name": img_name, "count": staves_count}, ignore_index=True)
    
    #final img to buffer, send back to index page
    output_buffer = io.BytesIO()
    annotated_img.save(output_buffer, format="JPEG")
    output_buffer.seek(0)
    
    return StreamingResponse(output_buffer, media_type="image/jpeg", headers={"staves-count": str(staves_count)})

#returns pandas table storing results as a html string
@app.get("/table/", response_class=HTMLResponse)
async def return_table():
    global table_df
    return HTMLResponse(table_df.to_html())

#clears the pandas table
@app.post("/clear_table/")
async def clear_table():
    global table_df
    # reinitialize with the same columns
    table_df = pd.DataFrame(columns=["img_name", "count"])
    return JSONResponse({"status": "cleared"})