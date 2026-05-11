FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libgl1 \
    libglib2.0-0 \
    wget \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
RUN test -f yolo-coco/yolov3.weights || wget -q -O yolo-coco/yolov3.weights https://pjreddie.com/media/files/yolov3.weights

EXPOSE 7860

CMD ["python", "app.py"]
