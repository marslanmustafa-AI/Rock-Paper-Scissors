"""FastAPI inference service for Rock-Paper-Scissors image classification."""
import os
import tempfile

# import keras
# from keras.utils import load_img, img_to_array
# import numpy as np
# import tensorflow as tf
from ultralytics import YOLO
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from contextlib import asynccontextmanager


MODEL_PATH = os.getenv("MODEL_PATH", "./notebooks/yolo/runs/classify/train/weights/best.pt")
LABELS = ["paper", "rock", "scissors"]


class PredictionResponse(BaseModel):
    predicted_class: str
    confidence: float
    all_predictions: dict[str, float]
    model_type: str = "keras"
    top5_classes: list[str] | None = None
    top5_confidences: list[float] | None = None


model = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model
    print(f"Loading model from {MODEL_PATH}...")
    model = YOLO(MODEL_PATH)
    print("Model loaded successfully!")
    yield
    print("Shutting down...")


app = FastAPI(
    title="RPS Classifier API",
    description="Rock-Paper-Scissors image classification API",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
        # "http://localhost:3000",
        # "https://rps.marslanmustafa.com",
        # "http://rps.marslanmustafa.com",
        # "http://localhost:3001",
        # "http://167.172.81.241:8080",
        # "http://167.172.81.241:3000",
        # "http://167.172.81.241:3000",
    # ],
    # allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def preprocess_image(file_bytes: bytes):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name
    return tmp_path


def predict(image_path):
    results = model(image_path)
    result = results[0]
    probs = result.probs.data
    class_idx = probs.argmax().item()
    confidence = float(100 * probs[class_idx].item())
    predicted_class = LABELS[class_idx]
    all_preds = {LABELS[i]: float(100 * probs[i].item()) for i in range(len(LABELS))}

    # YOLO-specific extended data
    top5_indices = result.probs.top5
    top5_classes = [LABELS[i] for i in top5_indices]
    top5_confidences = [float(100 * result.probs.top5conf[i].item()) for i in range(len(top5_indices))]

    return predicted_class, confidence, all_preds, top5_classes, top5_confidences


@app.get("/")
async def root():
    return {"status": "ok", "model": "yolo-classify", "model_type": "yolo", "labels": LABELS}


@app.post("/predict", response_model=PredictionResponse)
async def classify_image(file: UploadFile = File(...)):
    contents = await file.read()
    image_path = preprocess_image(contents)
    predicted_class, confidence, all_preds, top5_classes, top5_confidences = predict(image_path)

    os.unlink(image_path)

    return PredictionResponse(
        predicted_class=predicted_class,
        confidence=confidence,
        all_predictions=all_preds,
        model_type="yolo",
        top5_classes=top5_classes,
        top5_confidences=top5_confidences,
    )