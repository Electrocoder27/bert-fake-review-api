from pydantic import BaseModel
from fastapi import FastAPI
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F

app = FastAPI()

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("./bert_op_spam_model")
model = AutoModelForSequenceClassification.from_pretrained("./bert_op_spam_model")
model.eval()

# Define input format using Pydantic
class RequestModel(BaseModel):
    text: str

# Define label mapping
id2label = {0: "fake", 1: "genuine"}

# Correct endpoint that expects JSON body
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
