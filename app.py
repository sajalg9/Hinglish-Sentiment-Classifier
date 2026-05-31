import gradio as gr
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODEL_ID = "sajalgoel/muril-hinglish-sentiment"
MAX_LEN  = 128
ID2LABEL = {0: "Negative", 1: "Neutral", 2: "Positive"}
EMOJI    = {"Negative": "negative", "Neutral": "neutral", "Positive": "positive"}

print("Loading model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model     = AutoModelForSequenceClassification.from_pretrained(MODEL_ID)
model.eval()
print("Model ready.")


def predict(text: str):
    if not text.strip():
        return "Please enter some text.", {}

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding="max_length",
        max_length=MAX_LEN
    )
    with torch.no_grad():
        logits = model(**inputs).logits

    probs      = F.softmax(logits, dim=-1)[0]
    pred_id    = torch.argmax(probs).item()
    label      = ID2LABEL[pred_id]
    confidence = probs[pred_id].item()

    scores = {
        ID2LABEL[i]: round(probs[i].item(), 4)
        for i in range(len(ID2LABEL))
    }

    result_text = f"{label} ({confidence:.0%} confidence)"
    return result_text, scores


# ---- UI ----
examples = [
    ["yaar ye product ekdum bekar hai, total waste of money"],
    ["bhai mast tha, bilkul maza aaya ordering se"],
    ["delivery thodi late thi but product theek hai"],
    ["kya bakwaas service hai, never ordering again from here"],
    ["bahut acha experience tha, highly recommend karta hu"],
    ["average hai, na zyada acha na zyada bura"],
]

with gr.Blocks(title="Hinglish Sentiment Analyzer", theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    # Hinglish Sentiment Analyzer
    **Sentiment analysis for code-mixed Hindi-English (Hinglish) text**
    Fine-tuned [MuRIL](https://huggingface.co/google/muril-base-cased) on 18K+ Hinglish tweets and reviews.
    Handles transliteration, code-mixing, and informal Indian social media language.
    """)

    with gr.Row():
        with gr.Column(scale=2):
            text_input = gr.Textbox(
                label="Enter Hinglish Text",
                placeholder="e.g. yaar bahut mast tha, totally worth it!",
                lines=3
            )
            submit_btn = gr.Button("Analyze Sentiment", variant="primary")

        with gr.Column(scale=1):
            label_out = gr.Textbox(label="Prediction")
            scores_out = gr.Label(label="Confidence Scores")

    gr.Examples(examples=examples, inputs=text_input)

    gr.Markdown("""
    ---
    **Model:** `sajalgoel/muril-hinglish-sentiment` | **Labels:** Positive / Neutral / Negative
    **Accuracy:** 71.8% | **Weighted F1:** 0.72 on Hinglish validation set
    """)

    submit_btn.click(fn=predict, inputs=text_input, outputs=[label_out, scores_out])
    text_input.submit(fn=predict, inputs=text_input, outputs=[label_out, scores_out])

demo.launch()