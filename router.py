from fastapi import FastAPI, UploadFile, Form
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
import glob
import io
import os
import re
from PIL import Image, ImageOps
from yoloPostprocessUtils import crop_pillow_img_from_bbox, extract_highest_conf_bbox, convert_cornerWidthHeight_to_cornerCords
from fastapi.responses import HTMLResponse, JSONResponse
import pandas as pd
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
    image_bytes = await file.read()
    #this is needed for pillow
    image_stream = io.BytesIO(image_bytes)
    image = Image.open(image_stream)
    image.load()
    image = ImageOps.exif_transpose(image)

    #run image thru first stage staves localizer
    print("running through first stage localizer model")
    localizer_results = YOLO_OnnxRuntime(onnx_localizer_path, image, confidence_thres=conf_thresh, iou_thres=iou_thresh)
    _, localzier_boxes, localizer_conf_scores, _ = localizer_results.main()
    highest_conf_localizer_bbox, _ = extract_highest_conf_bbox(localzier_boxes, localizer_conf_scores)
    bbox_corner_cords = convert_cornerWidthHeight_to_cornerCords(highest_conf_localizer_bbox.tolist())
    cropped_image = crop_pillow_img_from_bbox(bbox_corner_cords, image)
    
    #second stage staves counter
    print("running through second stage staves counter model")
    counter_results = YOLO_OnnxRuntime(onnx_counter_path, cropped_image, confidence_thres=conf_thresh, iou_thres=iou_thresh)
    counter_annotated_arr, counter_boxes, _, _ = counter_results.main()
    staves_count = len(counter_boxes)
    
    #counter_results.main() returns numpy, need to conver to pil for further processing
    counter_annotated_img = Image.fromarray(counter_annotated_arr)
    
    # #store results to pandas table
    img_name = file.filename
    if img_name in table_df["img_name"].values:
        table_df.loc[table_df["img_name"]==img_name, "count"]          = staves_count
        table_df.loc[table_df["img_name"]==img_name, "conf_threshold"] = conf_thresh
    else:
        table_df = pd.concat([table_df, pd.DataFrame([{"img_name": img_name, "count": staves_count, "conf_threshold": conf_thresh}])], ignore_index=True)
    
    #final img to buffer, send back to index page
    output_buffer = io.BytesIO()
    counter_annotated_img.save(output_buffer, format="JPEG")
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