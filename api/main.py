"""FastAPI inference service for Rock-Paper-Scissors image classification."""
import os
from io import BytesIO

import keras
from keras.utils import load_img, img_to_array
import numpy as np
import tensorflow as tf
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from contextlib import asynccontextmanager


MODEL_PATH = os.getenv("MODEL_PATH", "./notebooks/models/rps_cnn_model.keras")
LABELS = ["paper", "rock", "scissors"]


class PredictionResponse(BaseModel):
    predicted_class: str
    confidence: float
    all_predictions: dict[str, float]


model = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model
    print(f"Loading model from {MODEL_PATH}...")
    model = keras.saving.load_model(MODEL_PATH)
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
    allow_origins=["http://localhost:3000", "https://rps.marslanmustafa.com", "http://localhost:3001"],
    # allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def preprocess_image(file_bytes: bytes):
    img = load_img(BytesIO(file_bytes), target_size=(150, 150))
    arr = img_to_array(img)
    arr = tf.expand_dims(arr, 0)
    return arr


def predict(image_array):
    predictions = model.predict(image_array, verbose=0)
    probs = predictions[0]
    class_idx = np.argmax(probs)
    confidence = float(100 * probs[class_idx])
    predicted_class = LABELS[class_idx]
    all_preds = {LABELS[i]: float(100 * probs[i]) for i in range(len(LABELS))}
    return predicted_class, confidence, all_preds


@app.get("/")
async def root():
    return {"status": "ok", "model": "rps_cnn_model", "labels": LABELS}


@app.post("/predict", response_model=PredictionResponse)
async def classify_image(file: UploadFile = File(...)):
    contents = await file.read()
    image_array = preprocess_image(contents)
    predicted_class, confidence, all_preds = predict(image_array)
    return PredictionResponse(
        predicted_class=predicted_class,
        confidence=confidence,
        all_predictions=all_preds,
    )