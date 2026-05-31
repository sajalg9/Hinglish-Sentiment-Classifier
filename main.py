from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from predict import predict_single, predict_batch
import time

app = FastAPI(
    title="Hinglish Sentiment API",
    description="Sentiment analysis for code-mixed Hinglish text using fine-tuned MuRIL",
    version="1.0.0"
)

app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

class SingleRequest(BaseModel):
    text: str

class BatchRequest(BaseModel):
    texts: list

@app.get("/")
def root():
    return {"message": "Hinglish Sentiment API is running", "model": "MuRIL fine-tuned on 18K+ Hinglish tweets", "labels": ["negative", "neutral", "positive"], "endpoints": ["/predict", "/batch", "/health"]}

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict")
def predict(req: SingleRequest):
    if not req.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty")
    if len(req.text) > 512:
        raise HTTPException(status_code=400, detail="Text too long (max 512 chars)")
    start   = time.time()
    result  = predict_single(req.text)
    latency = round((time.time() - start) * 1000, 2)
    return {"text": req.text, "label": result["label"], "confidence": result["confidence"], "scores": result["scores"], "latency_ms": latency}

@app.post("/batch")
def batch_predict(req: BatchRequest):
    if not req.texts:
        raise HTTPException(status_code=400, detail="texts list cannot be empty")
    if len(req.texts) > 32:
        raise HTTPException(status_code=400, detail="Max 32 texts per batch")
    start   = time.time()
    results = predict_batch(req.texts)
    latency = round((time.time() - start) * 1000, 2)
    return {"results": results, "count": len(results), "latency_ms": latency}