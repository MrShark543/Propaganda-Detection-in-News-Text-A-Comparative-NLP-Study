import os
import pandas as pd
import torch
import random
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
)
from tqdm.auto import tqdm
import torch.nn.functional as F

os.environ["WANDB_DISABLED"] = "true"    # disable wandb to avoid api call

# Load data
df = pd.read_csv("/content/propaganda_train.tsv", sep="\t")

rows = []
NEG_PER_POS = 5
MAX_SPAN_WORDS = 10

for _, r in df.iterrows():
    sent = r["tagged_in_context"].replace("<BOS>", "").replace("<EOS>", "").strip()
    gold = r["tagged_in_context"].split("<BOS>")[1].split("<EOS>")[0].strip()
    tech = r["label"]
    words = sent.split()
    n = len(words)

    # positive example
    rows.append({"sentence": sent, "span": gold, "label": tech})

    # collect all other spans up to MAX_SPAN_WORDS
    all_spans = []
    for i in range(n):
        for L in range(1, MAX_SPAN_WORDS+1):
            if i+L > n: break
            span = " ".join(words[i:i+L])
            if span != gold:
                all_spans.append(span)
    # sample a few negatives
    for neg in random.sample(all_spans, min(NEG_PER_POS, len(all_spans))):
        rows.append({"sentence": sent, "span": neg, "label": "not_propaganda"})

cand_df = pd.DataFrame(rows)
print("Total candidates:", len(cand_df))

# Train/val split
train_df, val_df = train_test_split(
    cand_df, test_size=0.2, stratify=cand_df["label"], random_state=42
)

# label mapping
labels = sorted(train_df["label"].unique())
label2id = {l:i for i,l in enumerate(labels)}

# Tokenizer & model
MODEL = "roberta-base"
tokenizer = AutoTokenizer.from_pretrained(MODEL)
model     = AutoModelForSequenceClassification.from_pretrained(
    MODEL, num_labels=len(labels), id2label={i:l for l,i in label2id.items()},
    label2id=label2id
)

#  Datasets
class SpanDataset(torch.utils.data.Dataset):
    """
    A custom PyTorch Dataset for sentence-span classification.

    Each sample consists of a sentence and a candidate span (either a 
    propaganda technique or a negative span), which are tokenized jointly. 
    The dataset is used to train a sequence classification model that 
    predicts whether a span in the sentence is a propaganda technique or not.

    Attributes:
        df (pd.DataFrame): The input dataframe containing 'sentence', 'span', and 'label' columns.
    
    Methods:
        __len__: Returns the number of samples in the dataset.
        __getitem__: Returns the tokenized input and label for a given index.
    """
    def __init__(self, df): self.df = df.reset_index(drop=True)
    def __len__(self): return len(self.df)
    def __getitem__(self, i):
        r = self.df.loc[i]
        enc = tokenizer(
            r["sentence"], r["span"],
            truncation=True, padding="max_length", max_length=128
        )
        enc["labels"] = label2id[r["label"]]
        return enc

train_ds = SpanDataset(train_df)
val_ds   = SpanDataset(val_df)

# Trainer args (legacy eval flags)
args = TrainingArguments(
    output_dir="roberta-candspan",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=8,
    do_eval=True,
    eval_steps=500,
    logging_steps=100,
    save_steps=500,
    save_total_limit=2,
    learning_rate=2e-5,
    weight_decay=0.01,
    
)

# Metrics
def compute_metrics(p):
    """
    Computes evaluation metrics for the model predictions during training and validation.

    This function calculates the classification report (precision, recall, F1-score) 
    for each label using sklearn's `classification_report`, and also computes the overall accuracy.

    Args:
        p: A namedtuple containing `predictions` (logits) and `label_ids` (ground-truth labels) 
           from the HuggingFace Trainer.

    Returns:
        dict: A dictionary of classification metrics including precision, recall, F1-score per class, 
              and overall accuracy.
    """
    preds = p.predictions.argmax(-1)
    return {
        **classification_report(p.label_ids, preds, target_names=labels, output_dict=True),
        "accuracy": (preds == p.label_ids).mean()
    }

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

# Train & eval
trainer.train()
print(trainer.evaluate())

# Detailed report
preds_out = trainer.predict(val_ds)
preds = preds_out.predictions.argmax(-1)
print(classification_report(
    [label2id[l] for l in val_df["label"]],
    preds, target_names=labels
))

trainer.save_model("roberta-candspan")        # saves model & config
tokenizer.save_pretrained("roberta-candspan")



## For testing the model on a new dataset
# Load model & tokenizer
device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_DIR = "roberta-candspan"
tokenizer = AutoTokenizer.from_pretrained("roberta-base")
model     = AutoModelForSequenceClassification.from_pretrained(
                MODEL_DIR, local_files_only=True
            ).to(device)
model.eval()

# Load test set
df_test = pd.read_csv("/content/propaganda_val.tsv", sep="\t")

# Labels
label2id = model.config.label2id
not_id   = label2id["not_propaganda"]

# Inference with chunking
MAX_SPAN_WORDS = 10
CHUNK_SIZE     = 64   
results        = []
y_true_match   = []
y_pred_match   = []

for _, row in tqdm(df_test.iterrows(), total=len(df_test), desc="Scoring sentences"):
    sent      = row["tagged_in_context"].replace("<BOS>", "").replace("<EOS>", "").strip()
    gold_span = row["tagged_in_context"].split("<BOS>")[1].split("<EOS>")[0].strip()
    label     = row['label']
    words = sent.split()
    cands = [
        " ".join(words[i : i+L])
        for i in range(len(words))
        for L in range(1, MAX_SPAN_WORDS+1)
        if i+L <= len(words)
    ]

    best_score = 0.0
    best_span  = ""

    # process in small batches
    for i in range(0, len(cands), CHUNK_SIZE):
        chunk = cands[i : i+CHUNK_SIZE]
        enc = tokenizer(
            [sent]*len(chunk), chunk,
            truncation=True, padding=True, max_length=128,
            return_tensors="pt"
        ).to(device)

        with torch.no_grad():
            logits = model(**enc).logits        # (batch, num_labels)
            probs  = F.softmax(logits, dim=-1)

        # mask out the 'not_propaganda' column
        probs[:, not_id] = 0.0

        # find the best within this chunk
        chunk_scores, chunk_idxs = probs.max(dim=1)
        max_idx  = torch.argmax(chunk_scores).item()
        max_score = chunk_scores[max_idx].item()

        if max_score > best_score:
            best_score = max_score
            best_span  = chunk[max_idx]

        # free GPU memory
        del enc, logits, probs, chunk_scores, chunk_idxs
        torch.cuda.empty_cache()

    match = (best_span == gold_span)
    results.append({
        "sentence":       sent,
        "gold_span":      gold_span,
        "predicted_span": best_span,
        "match":          match
    })
    y_true_match.append(True)
    y_pred_match.append(match)
    print(y_true_match[-1], y_pred_match[-1], label)

# Report & save
print("=== Span‐Match Detection Report ===")
print(classification_report(
    y_true_match, y_pred_match,
    target_names=["no_match","match"], zero_division=0
))

df_out = pd.DataFrame(results)
df_out.to_excel("span_eval_results.xlsx", index=False)
print("\nSaved /content/span_eval_results.xlsx")