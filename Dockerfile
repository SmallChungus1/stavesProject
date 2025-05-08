FROM python:3.12-slim

WORKDIR /app

#need this so docker container doesn't run into libGL issues
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y

COPY requirements.txt .

#change line below for other cuda-pytorch versions
#RUN pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
RUN pip install --no-cache-dir torch torchvision torchaudio
#removed pytorch-cuda versions from requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY router.py yoloPostprocessUtils.py ./
COPY static/ ./static/
COPY models/ ./models/

EXPOSE 8000

#command to run webapp via uvicorn
CMD ["uvicorn", "router:app", "--host", "0.0.0.0", "--port", "8000"]