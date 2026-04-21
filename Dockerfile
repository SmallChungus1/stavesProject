FROM python:3.12-slim

WORKDIR /app
#fall back path, is .env not provided
ENV LOG_DB_PATH=/app/data/inference_logs.db

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
    python-multipart

COPY router.py db.py yoloPostprocessUtils.py yolo_onnx.py ./
COPY static/ ./static/
COPY weights/ ./models/

EXPOSE 8000

CMD ["sh", "-c", "python -m db && uvicorn router:app --host 0.0.0.0 --port 8000"]
