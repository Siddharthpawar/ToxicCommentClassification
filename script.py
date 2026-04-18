# BERT Classification + Clustering + WordCloud + External Test Set Evaluation

import os

_ROOT = os.path.dirname(os.path.abspath(__file__))
# Fast runs by default (subset + 1 epoch). Full data + 3 epochs: TOXIC_FULL=1
_FULL = os.environ.get("TOXIC_FULL", "").lower() in ("1", "true", "yes")
if _FULL:
    _QUICK = False
else:
    _QUICK = os.environ.get("TOXIC_QUICK", "1").lower() not in ("0", "false", "no", "off")
_TRAIN_CSV = os.environ.get("TOXIC_TRAIN_CSV", os.path.join(_ROOT, "train.csv"))
if not os.path.isfile(_TRAIN_CSV):
    _fallback_train = os.path.join(_ROOT, "train1.csv")
    if os.path.isfile(_fallback_train):
        _TRAIN_CSV = _fallback_train
    else:
        raise FileNotFoundError(
            f"Training CSV not found. Expected train.csv or train1.csv under {_ROOT}, "
            "or set TOXIC_TRAIN_CSV to your file path."
        )
_TEST_CSV = os.environ.get("TOXIC_TEST_CSV", os.path.join(_ROOT, "test.csv"))
_BERT_MODEL = os.environ.get("TOXIC_BERT_MODEL", "bert-base-uncased")

import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset, DatasetDict
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score
from sklearn.decomposition import TruncatedSVD
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import seaborn as sns

_SAVE_PLOTS = os.environ.get("TOXIC_SAVE_PLOTS", "").lower() in ("1", "true", "yes")
_PLOT_DIR = os.path.join(_ROOT, "plots")


def _finalize_figure(filename: str) -> None:
    if _SAVE_PLOTS:
        os.makedirs(_PLOT_DIR, exist_ok=True)
        path = os.path.join(_PLOT_DIR, filename)
        plt.savefig(path, dpi=150, bbox_inches="tight")
        print(f"Saved plot: {path}")
    plt.show()


# --- 1. Load Training Data ---
train_df = pd.read_csv(_TRAIN_CSV, header=0, low_memory=False)
label_cols = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
train_df["labels"] = train_df[label_cols].apply(lambda row: row.astype(float).tolist(), axis=1)
if _QUICK:
    _n = int(os.environ.get("TOXIC_QUICK_ROWS", "2000"))
    train_df = train_df.head(_n).copy()

# --- 2. Train-Test Split ---
train_data, val_data = train_test_split(train_df[["comment_text", "labels"]], test_size=0.2, random_state=42)
dataset = DatasetDict({
    "train": Dataset.from_pandas(train_data),
    "test": Dataset.from_pandas(val_data)
})

# --- 3. Tokenizer ---
tokenizer = BertTokenizer.from_pretrained(_BERT_MODEL)
def tokenize(batch):
    return tokenizer(batch["comment_text"], padding="max_length", truncation=True, max_length=128)
dataset = dataset.map(tokenize, batched=True)
dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

# --- 4. Load Model ---
model = BertForSequenceClassification.from_pretrained(
    _BERT_MODEL,
    num_labels=6,
    problem_type="multi_label_classification",
)

# --- 5. Metrics ---
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    probs = torch.sigmoid(torch.tensor(logits)).cpu().numpy()
    preds = (probs > 0.5).astype(int)
    return {
        "f1": f1_score(labels, preds, average="macro"),
        "accuracy": accuracy_score(labels, preds)
    }

# --- 6. Trainer Setup ---
_num_epochs = 1 if _QUICK else 3
training_args = TrainingArguments(
    output_dir=os.path.join(_ROOT, "results"),
    report_to="none",
    eval_strategy="epoch",
    save_strategy="epoch",
    logging_dir=os.path.join(_ROOT, "logs"),
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=_num_epochs,
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    greater_is_better=True,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    compute_metrics=compute_metrics
)

# --- 7. Train Model ---
os.environ["WANDB_DISABLED"] = "true"
trainer.train()
_saved = os.path.join(_ROOT, "saved_model")
model.save_pretrained(_saved)
tokenizer.save_pretrained(_saved)

dataset["train"].reset_format()
dataset["train"].set_format(
    type="torch",
    columns=["input_ids", "attention_mask", "labels", "comment_text"]
)


# --- 8. Extract CLS Embeddings ---
def extract_cls_embeddings(model, dataset):
    model.eval()
    device = next(model.parameters()).device
    loader = DataLoader(dataset, batch_size=16)
    embeddings, labels, texts = [], [], []

    for batch in tqdm(loader):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        with torch.no_grad():
            outputs = model.bert(input_ids=input_ids, attention_mask=attention_mask)
            cls_embed = outputs.last_hidden_state[:, 0, :].cpu().numpy()
            embeddings.append(cls_embed)
            labels.extend(batch['labels'].cpu().numpy())
        texts.extend(batch['comment_text'])

    return np.vstack(embeddings), np.array(labels), texts

cls_embeddings, label_array, comment_texts = extract_cls_embeddings(model, dataset["train"])

# --- 9. SVD + Clustering ---
scaler = StandardScaler()
X_scaled = scaler.fit_transform(cls_embeddings)
svd = TruncatedSVD(n_components=2, random_state=42)
X_reduced = svd.fit_transform(X_scaled)

kmeans = KMeans(n_clusters=5, random_state=42)
cluster_labels = kmeans.fit_predict(X_reduced)

# --- 10. Plot Clusters ---
plt.figure(figsize=(10, 6))
sns.scatterplot(x=X_reduced[:, 0], y=X_reduced[:, 1], hue=cluster_labels, palette='viridis')
plt.title("KMeans Clustering of BERT CLS Embeddings (2D SVD)")
plt.xlabel("Component 1")
plt.ylabel("Component 2")
plt.legend(title="Cluster")
plt.grid(True)
_finalize_figure("clusters_kmeans_svd.png")

# --- 11. WordClouds per Cluster ---
comment_array = np.array(comment_texts)
for i in range(5):
    cluster_comments = comment_array[cluster_labels == i]
    text = " ".join(cluster_comments)
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
    plt.figure(figsize=(10, 4))
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    plt.title(f"WordCloud for Cluster {i}")
    _finalize_figure(f"wordcloud_cluster_{i}.png")

# --- 12. Load External Test Set ---
if not os.path.isfile(_TEST_CSV):
    raise FileNotFoundError(
        f"Test CSV not found at {_TEST_CSV}. Download test.csv from the Kaggle competition "
        "or set TOXIC_TEST_CSV."
    )
external_test_df = pd.read_csv(_TEST_CSV, low_memory=False)  # expects 'comment_text' column
if _QUICK:
    _tn = int(os.environ.get("TOXIC_QUICK_TEST_ROWS", "500"))
    external_test_df = external_test_df.head(_tn).copy()
test_dataset = Dataset.from_pandas(external_test_df)
test_dataset = test_dataset.map(tokenize, batched=True)
test_dataset.set_format(type="torch", columns=["input_ids", "attention_mask"])

# --- 13. Predict on Test Set ---
def predict_batch(dataset, model, threshold=0.5):
    model.eval()
    device = next(model.parameters()).device
    loader = DataLoader(dataset, batch_size=16)
    results = []

    for batch in loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            probs = torch.sigmoid(outputs.logits).cpu().numpy()
            preds = (probs > threshold).astype(int)
            results.extend(preds)
    return results

test_predictions = predict_batch(test_dataset, model)
pred_df = pd.DataFrame(test_predictions, columns=label_cols)
final_output = pd.concat([external_test_df, pred_df], axis=1)
final_output.to_csv(os.path.join(_ROOT, "final_predictions.csv"), index=False)

from sklearn.metrics import roc_curve, auc, hamming_loss, multilabel_confusion_matrix
from sklearn.preprocessing import label_binarize

# --- 14. Evaluate on Validation Set with ROC, Hamming Loss, Heatmap ---

# Get true labels and predictions
val_loader = DataLoader(dataset["test"], batch_size=16)
all_logits, all_labels = [], []

model.eval()
device = next(model.parameters()).device

for batch in val_loader:
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    labels = batch["labels"].cpu().numpy()
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits.cpu().numpy()
    all_logits.append(logits)
    all_labels.append(labels)

all_logits = np.vstack(all_logits)
all_labels = np.vstack(all_labels)
probs = torch.sigmoid(torch.tensor(all_logits)).numpy()
preds = (probs > 0.5).astype(int)

# --- 14.1 Hamming Loss ---
hloss = hamming_loss(all_labels, preds)
print(f"\nHamming Loss: {hloss:.4f}")

# --- 14.2 ROC Curve Plot ---
fpr = dict()
tpr = dict()
roc_auc = dict()

for i in range(len(label_cols)):
    fpr[i], tpr[i], _ = roc_curve(all_labels[:, i], probs[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

plt.figure(figsize=(10, 7))
for i in range(len(label_cols)):
    plt.plot(fpr[i], tpr[i], label=f"{label_cols[i]} (AUC = {roc_auc[i]:.2f})")
plt.plot([0, 1], [0, 1], "k--", lw=1)
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curves for Toxic Comment Categories")
plt.legend(loc="lower right")
plt.grid(True)
_finalize_figure("roc_curves.png")

# --- 14.3 Heatmap from Confusion Matrix ---
conf_matrices = multilabel_confusion_matrix(all_labels, preds)
fig, axs = plt.subplots(2, 3, figsize=(15, 10))
axs = axs.ravel()
for i in range(len(label_cols)):
    sns.heatmap(conf_matrices[i], annot=True, fmt="d", ax=axs[i], cmap="Blues", cbar=False)
    axs[i].set_title(f"Confusion Matrix: {label_cols[i]}")
    axs[i].set_xlabel("Predicted")
    axs[i].set_ylabel("True")

plt.tight_layout()
_finalize_figure("confusion_matrices.png")

# --- 15. Predicting Toxicity from User Input ---
def predict_comment(text, model, tokenizer, label_cols, threshold=0.5):
    model.eval()

    # Automatically detect the model's device (cuda or cpu)
    device = next(model.parameters()).device

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=128
    ).to(device)  # Move input tensors to same device as model

    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.sigmoid(outputs.logits).squeeze().cpu().numpy()

    preds = (probs > threshold).astype(int)
    label_map = dict(zip(label_cols, preds))

    print(f"\nInput: {text}\nPredicted categories:")
    for label, value in label_map.items():
        if value == 1:
            print(f" - {label}")
    if sum(preds) == 0:
        print(" - (none)")

    return label_map


if __name__ == "__main__":

    print("=== Toxic Comment Classifier ===")
    print("Type a comment and press Enter. Type 'exit' or 'quit' to stop.\n")

    while True:
        text = input("Your comment: ").strip()
        if text.lower() in ('exit', 'quit'):
            print("Goodbye, Have A Great Day!")
            break


        predict_comment(text, model, tokenizer, label_cols)
