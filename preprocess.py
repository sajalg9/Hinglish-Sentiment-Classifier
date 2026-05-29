"""
preprocess.py
Hinglish Sentiment + Emotion — Data Preprocessing Script
Run this locally in VS Code before uploading to Kaggle.
Output: data/sentiment_train.csv, data/sentiment_val.csv, data/sentiment_test.csv
"""

import pandas as pd
import os
import re
from sklearn.model_selection import train_test_split

# ── Paths ──────────────────────────────────────────────────────────────────────
DS6_TRAIN  = "archive (6)/FinalTrainingOnly.csv"
DS6_VAL    = "archive (6)/ValidationOnly.csv"
DS6_TEST   = "archive (6)/FinalTest.csv"
DS8_EMOTION = "archive (8)/emotion_hinghlish_dataset.xlsx"
OUTPUT_DIR  = "data"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ── 1. Load DS6 (Sentiment: 0=negative, 1=neutral, 2=positive) ────────────────
print("Loading DS6 (sentiment)...")

df6_train = pd.read_csv(DS6_TRAIN, header=None, names=["id", "text", "sentiment_label"])
df6_val   = pd.read_csv(DS6_VAL,   header=None, names=["id", "text", "sentiment_label"])
df6_test  = pd.read_csv(DS6_TEST,  header=None, names=["id", "text"])  # no labels in test

# Map numeric labels to strings
sentiment_map = {0: "negative", 1: "neutral", 2: "positive"}
df6_train["sentiment"] = df6_train["sentiment_label"].astype(int).map(sentiment_map)
df6_val["sentiment"]   = df6_val["sentiment_label"].astype(int).map(sentiment_map)

print(f"  DS6 train: {len(df6_train)} rows")
print(f"  DS6 val:   {len(df6_val)} rows")
print(f"  Label distribution:\n{df6_train['sentiment'].value_counts()}\n")


# ── 2. Load DS8 Emotion (joy/anger/love/fear/sadness etc.) ────────────────────
print("Loading DS8 (emotion)...")
df8 = pd.read_excel(DS8_EMOTION)
df8.columns = df8.columns.str.strip().str.lower()

# Rename columns to match our schema
df8 = df8.rename(columns={"text": "text", "emotion": "emotion_label"})

# Map emotion → rough sentiment (useful for combining datasets)
emotion_to_sentiment = {
    "joy":         "positive",
    "love":        "positive",
    "admiration":  "positive",
    "surprise":    "neutral",
    "neutral":     "neutral",
    "fear":        "negative",
    "sadness":     "negative",
    "anger":       "negative",
    "disgust":     "negative",
    "disapproval": "negative",
}
df8["sentiment"] = df8["emotion_label"].map(emotion_to_sentiment)
df8 = df8.dropna(subset=["sentiment"])

print(f"  DS8 rows: {len(df8)}")
print(f"  Emotion distribution:\n{df8['emotion_label'].value_counts()}\n")


# ── 3. Text Cleaning ───────────────────────────────────────────────────────────
def clean_text(text):
    if not isinstance(text, str):
        return ""
    # Remove URLs
    text = re.sub(r"http\S+|www\S+", "", text)
    # Remove @mentions
    text = re.sub(r"@\w+", "", text)
    # Remove hashtag symbol but keep word
    text = re.sub(r"#(\w+)", r"\1", text)
    # Remove extra whitespace
    text = re.sub(r"\s+", " ", text).strip()
    # Remove very short texts
    return text if len(text) > 5 else ""

print("Cleaning text...")
for df in [df6_train, df6_val, df8]:
    df["text"] = df["text"].apply(clean_text)

# Drop empty rows after cleaning
df6_train = df6_train[df6_train["text"] != ""].reset_index(drop=True)
df6_val   = df6_val[df6_val["text"] != ""].reset_index(drop=True)
df8       = df8[df8["text"] != ""].reset_index(drop=True)

print(f"  After cleaning — DS6 train: {len(df6_train)}, DS8: {len(df8)}\n")


# ── 4. Build Combined Sentiment Dataset ───────────────────────────────────────
print("Combining datasets...")

# Keep only what we need
cols = ["text", "sentiment"]
df6_combined = df6_train[cols].copy()

# Sample DS8 to avoid class imbalance (3000 per emotion = 30K total, too heavy)
# We'll take 4000 rows from DS8 to augment DS6
df8_sample = df8[cols].sample(n=min(4000, len(df8)), random_state=42)

combined = pd.concat([df6_combined, df8_sample], ignore_index=True)
combined = combined.drop_duplicates(subset=["text"]).reset_index(drop=True)
combined = combined.sample(frac=1, random_state=42).reset_index(drop=True)  # shuffle

print(f"  Combined dataset: {len(combined)} rows")
print(f"  Final label distribution:\n{combined['sentiment'].value_counts()}\n")


# ── 5. Split & Save ───────────────────────────────────────────────────────────
print("Saving splits...")

# Use DS6's existing val split + our combined train
train_df = combined
val_df   = df6_val[cols].copy()

# Add integer label column (needed by HuggingFace Trainer)
label2id = {"negative": 0, "neutral": 1, "positive": 2}
train_df["label"] = train_df["sentiment"].map(label2id)
val_df["label"]   = val_df["sentiment"].map(label2id)

train_df.to_csv(f"{OUTPUT_DIR}/sentiment_train.csv", index=False)
val_df.to_csv(f"{OUTPUT_DIR}/sentiment_val.csv",   index=False)

# Also save emotion dataset separately for the dual-output model later
df8[["text", "emotion_label"]].to_csv(f"{OUTPUT_DIR}/emotion_data.csv", index=False)

print(f"  Saved: {OUTPUT_DIR}/sentiment_train.csv ({len(train_df)} rows)")
print(f"  Saved: {OUTPUT_DIR}/sentiment_val.csv   ({len(val_df)} rows)")
print(f"  Saved: {OUTPUT_DIR}/emotion_data.csv    ({len(df8)} rows)")
print("\n✅ Preprocessing complete! Upload the data/ folder to Kaggle next.")
