import os
import zipfile
import requests
from fastapi import FastAPI
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

app = FastAPI()

MODEL_DIR = "bert_op_spam_model"
ZIP_PATH = "bert_op_spam_model.zip"
GOOGLE_DRIVE_URL = "https://drive.google.com/uc?export=download&id=1ypaDFgM4d7fwTWAz1YIWch7BHyIV5gCy"

def download_and_extract_model():
    if not os.path.exists(MODEL_DIR):
        print("Model not found. Downloading...")
        response = requests.get(GOOGLE_DRIVE_URL)
        with open(ZIP_PATH, "wb") as f:
            f.write(response.content)

        print("Extracting model...")
        with zipfile.ZipFile(ZIP_PATH, 'r') as zip_ref:
            zip_ref.extractall()
        os.remove(ZIP_PATH)
        print("Model ready!")

download_and_extract_model()

tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)

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
