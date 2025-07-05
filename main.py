import os
import zipfile
import gdown
from fastapi import FastAPI
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

app = FastAPI()

MODEL_DIR = "bert_op_spam_model"
ZIP_PATH = "bert_op_spam_model.zip"
GOOGLE_DRIVE_URL = "https://drive.google.com/uc?id=1f4QyHVnLGmXh-mqjvtRQzHFv_xS7z69K"

def download_and_extract_model():
    if not os.path.exists(MODEL_DIR):
        print("Model not found. Downloading...")

        # Download from Google Drive using gdown
        gdown.download(GOOGLE_DRIVE_URL, ZIP_PATH, quiet=False)

        print("Extracting model...")
        with zipfile.ZipFile(ZIP_PATH, 'r') as zip_ref:
            zip_ref.extractall(MODEL_DIR)

        os.remove(ZIP_PATH)
        print("Model ready!")

# Download and extract the model if not already done
download_and_extract_model()

# Load model and tokenizer from the extracted directory
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR, use_safetensors=True)

@app.get("/")
def root():
    return {"message": "Fake Review Detector is LIVE!"}

@app.post("/predict")
def predict(text: str):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    outputs = model(**inputs)
    probs = torch.softmax(outputs.logits, dim=1)
    predicted_class = torch.argmax(probs, dim=1).item()
    confidence = probs[0][predicted_class].item()

    return {
        "prediction": predicted_class,
        "confidence": round(confidence, 4)
    }
