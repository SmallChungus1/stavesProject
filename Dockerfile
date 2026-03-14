FROM python:3.12-slim

WORKDIR /app

#need this so docker container doesn't run into libGL issues
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6 -y

RUN pip install --no-cache-dir \
    onnxruntime \
    fastapi \
    "uvicorn[standard]" \
    numpy \
    opencv-python-headless \
    python-dotenv \
    pillow \
    python-multipart \
    pandas

COPY router.py yoloPostprocessUtils.py yolo_onnx.py ./
COPY static/ ./static/
COPY weights/ ./models/

EXPOSE 8000

#command to run webapp via uvicorn
CMD ["uvicorn", "router:app", "--host", "0.0.0.0", "--port", "8000"]
