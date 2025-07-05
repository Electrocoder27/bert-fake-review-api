import os
import zipfile
import gdown
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F

app = FastAPI()

# Model path and download config
MODEL_DIR = "bert_op_spam_model"
ZIP_PATH = "bert_op_spam_model.zip"
GOOGLE_DRIVE_URL = "https://drive.google.com/uc?id=1f4QyHVnLGmXh-mqjvtRQzHFv_xS7z69K"

# Download and extract model if not present
def download_and_extract_model():
    if not os.path.exists(MODEL_DIR):
        print("Downloading model from Google Drive...")
        gdown.download(GOOGLE_DRIVE_URL, ZIP_PATH, quiet=False)
        print("Extracting model...")
        with zipfile.ZipFile(ZIP_PATH, 'r') as zip_ref:
            zip_ref.extractall(MODEL_DIR)
        os.remove(ZIP_PATH)
        print("Model ready!")

download_and_extract_model()

# Load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR, use_safetensors=True)
model.eval()

# Input model for POST
class RequestModel(BaseModel):
    text: str

# Label mapping
id2label = {0: "fake", 1: "genuine"}

# ✅ Root health check route
@app.get("/")
def root():
    return {"message": "Fake Review Detector is LIVE!"}

# Prediction route
@app.post("/predict")
def predict(request: RequestModel):
    inputs = tokenizer(request.text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
        probs = F.softmax(outputs.logits, dim=1)
        confidence, predicted = torch.max(probs, dim=1)

    return {
        "label": id2label[predicted.item()],
        "confidence": round(confidence.item(), 4)
    }

