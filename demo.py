"""Load the trained BERT from saved_model/ and classify comments interactively (no training)."""

import os

import torch
from transformers import BertForSequenceClassification, BertTokenizer

_ROOT = os.path.dirname(os.path.abspath(__file__))
_MODEL_DIR = os.environ.get("TOXIC_MODEL_DIR", os.path.join(_ROOT, "saved_model"))

LABEL_COLS = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]


def predict_comment(text, model, tokenizer, label_cols, threshold=0.5):
    model.eval()
    device = next(model.parameters()).device
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=128,
    ).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.sigmoid(outputs.logits).squeeze().cpu().numpy()

    preds = (probs > threshold).astype(int)
    label_map = dict(zip(label_cols, preds))

    print(f"\nInput: {text}\nProbabilities:")
    for label, p in zip(label_cols, probs):
        print(f"  {label:16s} {p:.3f}")
    print("Predicted categories (threshold 0.5):")
    for label, value in label_map.items():
        if value == 1:
            print(f"  - {label}")
    if int(preds.sum()) == 0:
        print("  - (none)")

    return label_map


def main():
    if not os.path.isdir(_MODEL_DIR):
        raise FileNotFoundError(
            f"No folder at {_MODEL_DIR}. Train first (python script.py) or set TOXIC_MODEL_DIR."
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading model from {_MODEL_DIR} ({device})...")
    tokenizer = BertTokenizer.from_pretrained(_MODEL_DIR)
    model = BertForSequenceClassification.from_pretrained(_MODEL_DIR)
    model.to(device)
    model.eval()

    print("\n=== Toxic comment classifier (demo) ===")
    print("Type a comment and press Enter. Type 'exit' or 'quit' to stop.\n")

    while True:
        text = input("Your comment: ").strip()
        if text.lower() in ("exit", "quit"):
            print("Goodbye!")
            break
        if not text:
            continue
        predict_comment(text, model, tokenizer, LABEL_COLS)


if __name__ == "__main__":
    main()
