import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch.nn.functional as F

MODEL_PATH = r"c:/Users/sajal/Desktop/project/final_model"
MAX_LEN    = 128
ID2LABEL   = {0: "negative", 1: "neutral", 2: "positive"}

print("Loading model from:", MODEL_PATH)
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model     = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
model.eval()
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
model.to(DEVICE)
print("Model loaded on", DEVICE)


def predict_single(text: str) -> dict:
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding="max_length", max_length=MAX_LEN).to(DEVICE)
    with torch.no_grad():
        logits = model(**inputs).logits
    probs      = F.softmax(logits, dim=-1)[0]
    pred_id    = torch.argmax(probs).item()
    label      = ID2LABEL[pred_id]
    confidence = probs[pred_id].item()
    scores     = {ID2LABEL[i]: round(probs[i].item(), 4) for i in range(len(ID2LABEL))}
    return {"label": label, "confidence": round(confidence, 4), "scores": scores}


def predict_batch(texts: list) -> list:
    inputs = tokenizer(texts, return_tensors="pt", truncation=True, padding="max_length", max_length=MAX_LEN).to(DEVICE)
    with torch.no_grad():
        logits = model(**inputs).logits
    probs   = F.softmax(logits, dim=-1)
    results = []
    for i in range(len(texts)):
        pred_id    = torch.argmax(probs[i]).item()
        label      = ID2LABEL[pred_id]
        confidence = probs[i][pred_id].item()
        scores     = {ID2LABEL[j]: round(probs[i][j].item(), 4) for j in range(len(ID2LABEL))}
        results.append({"text": texts[i], "label": label, "confidence": round(confidence, 4), "scores": scores})
    return results