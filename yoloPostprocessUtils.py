from fastapi import FastAPI, UploadFile, File
from fastapi.responses import StreamingResponse
import io
from PIL import Image, ImageDraw
from ultralytics import YOLO
import numpy as np
import cv2 



def crop_pillow_img_from_bbox(bbox_cords, pil_img_obj):
    cropped_img_obj = pil_img_obj.crop(bbox_cords)

    return cropped_img_obj

def draw_detections_on_img(detection, pil_img_obj, dot_radius=10):
    total_detections = len(detection.boxes)
    draw = ImageDraw.Draw(pil_img_obj)
    

    for bbox in detection.boxes.xyxy.cpu().detach().numpy():
        x_min, y_min, x_max, y_max = bbox[:4]
        
        center_x = int(((x_min + x_max) / 2))
        center_y = int(((y_min + y_max) / 2))
        
        draw.ellipse(
            (center_x - dot_radius, center_y - dot_radius, center_x + dot_radius, center_y + dot_radius),
            fill='red'
        )
    
    return pil_img_obj, total_detections

def extract_highest_conf_bbox(bboxes, conf_scores):
    localizer_conf_scores = np.array(conf_scores)
    localizer_boxes = np.array(bboxes)

    max_idx = np.argmax(localizer_conf_scores)

    top_box = localizer_boxes[max_idx]
    top_score = localizer_conf_scores[max_idx]

    return top_box, top_score

def convert_cornerWidthHeight_to_cornerCords(bbox_cords):
    x1, y1, w, h = bbox_cords
    x2 = x1 + w
    y2 = y1 + h
    
    return (x1, y1, x2, y2)