# Hinglish Sentiment Analyzer

Sentiment analysis for **code-mixed Hindi-English (Hinglish)** text using a fine-tuned [MuRIL](https://huggingface.co/google/muril-base-cased) transformer model.

🟢 **[Live Demo](https://huggingface.co/spaces/sajalgoel/hinglish-sentiment-analyzer)** | 🤗 **[Model on HuggingFace](https://huggingface.co/sajalgoel/muril-hinglish-sentiment)**

---

## What makes this different

Standard NLP models (BERT, RoBERTa) are trained on clean English and fail on Hinglish — the informal code-mixed language used across Indian social media, app reviews, and messaging. This project specifically targets:

- Hindi-English code-mixing ("yaar ye product ekdum bekar hai")
- Roman script transliteration of Hindi words
- Informal social media tone and slang

---

## Results

| Metric | Score |
|--------|-------|
| Accuracy | 71.8% |
| Weighted F1 | 0.72 |
| Negative F1 | 0.74 |
| Neutral F1 | 0.64 |
| Positive F1 | 0.76 |

Evaluated on held-out Hinglish validation set.

---

## Dataset

- **DS6:** ~14,500 labeled Hinglish tweets (negative / neutral / positive)
- **DS8:** 30,000 Hinglish tweets with 10 emotion labels (anger, joy, love, fear, etc.)
- Combined and cleaned: **18,000+ training samples**
- Preprocessing: URL removal, mention stripping, whitespace normalization, short-text filtering

---

## Model

- **Base model:** `google/muril-base-cased` — pretrained on 17 Indian languages including Hinglish
- **Task:** 3-class sequence classification (negative / neutral / positive)
- **Fine-tuning:** 4 epochs, batch size 32, learning rate 2e-5, warmup ratio 0.1
- **Hardware:** Kaggle P100 GPU (~25 minutes training)
- **Early stopping:** patience of 2 epochs on weighted F1

Why MuRIL over BERT/DistilBERT? MuRIL was pretrained on Hindi, English, and transliterated Indian text — making it significantly better suited for code-mixed Hinglish than models trained on English only.

---

## API

FastAPI backend with two endpoints:

```bash
# Single prediction
POST /predict
{"text": "yaar bahut bakwaas tha ekdum"}

# Response
{
  "text": "yaar bahut bakwaas tha ekdum",
  "label": "negative",
  "confidence": 0.832,
  "scores": {"negative": 0.832, "neutral": 0.135, "positive": 0.033},
  "latency_ms": 54.4
}
```

```bash
# Batch prediction (up to 32 texts)
POST /batch
{"texts": ["mast product hai", "bilkul bekar hai", "theek theek hai"]}
```

---

## Project Structure

```
hinglish-sentiment-analyzer/
├── data/
│   ├── sentiment_train.csv     # 18K+ training samples
│   ├── sentiment_val.csv       # validation split
│   └── emotion_data.csv        # emotion labels (DS8)
├── app/
│   ├── main.py                 # FastAPI app
│   └── predict.py              # model inference logic
├── spaces/
│   ├── app.py                  # Gradio demo (HuggingFace Spaces)
│   └── requirements.txt
├── preprocess.py               # data cleaning and merging
├── train_kaggle.ipynb          # MuRIL fine-tuning notebook
└── requirements.txt
```

---

## Run Locally

```bash
git clone https://github.com/sajalgoel/hinglish-sentiment-analyzer
cd hinglish-sentiment-analyzer
pip install -r requirements.txt

# Download model from HuggingFace
# Place in final_model/ folder

cd app
uvicorn main:app --reload --port 8000
# Open http://localhost:8000/docs
```

---

## Tech Stack

- **Model:** HuggingFace Transformers (MuRIL)
- **Training:** PyTorch + HuggingFace Trainer API
- **API:** FastAPI + Uvicorn
- **Demo:** Gradio
- **Deployment:** HuggingFace Spaces (free tier)
- **Training compute:** Kaggle P100 GPU (free)
