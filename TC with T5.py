import pandas as pd
from transformers import T5Tokenizer
from datasets import Dataset
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.optim import AdamW
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from transformers import (
    T5ForConditionalGeneration,
    get_linear_schedule_with_warmup,
)
import numpy as np


df = pd.read_csv("/content/propaganda_train.tsv", sep="\t")
print(df.shape)
df.head()



# Build label ↔ ID mapping once
labels = sorted(df["label"].unique())
label2id = {lab:i for i,lab in enumerate(labels)}

def preprocess_row(row):
    """
    Prepares each row of the dataset for T5 training by formatting the input and target text.

    - Replaces <BOS> and <EOS> markers in the sentence with [SPAN] and [/SPAN].
    - Builds a prompt that includes the sentence and all possible label options.
    - Returns a dictionary with the input prompt, the correct label as target text, and its ID.

    Args:
        row (pd.Series): A row from the DataFrame containing 'tagged_in_context' and 'label'.

    Returns:
        dict: A dictionary with 'input_text', 'target_text', and 'label_id'.
    """
    # Replace markers
    text = (
        row["tagged_in_context"]
          .replace("<BOS>", "[SPAN]")
          .replace("<EOS>", "[/SPAN]")
    )
    #  prompt: list all options
    opts   = ", ".join(labels)
    prompt = f"classify_span: {text} Options: {opts}. Answer:"
    return {
        "input_text": prompt,
        "target_text": row["label"],
        "label_id":   label2id[row["label"]]
    }

hf_ds = Dataset.from_pandas(df)
hf_ds = hf_ds.map(
    preprocess_row,
    remove_columns=["label", "tagged_in_context"]
)
hf_ds = hf_ds.train_test_split(test_size=0.1, seed=42)
train_ds, val_ds = hf_ds["train"], hf_ds["test"]




# Load T5-base tokenizer
tokenizer = T5Tokenizer.from_pretrained("t5-base")

#  Add each label as _one_ special token
tokenizer.add_special_tokens({
    "additional_special_tokens": labels
})

def tokenize_fn(batch):
    """
    Tokenizes input and target text for T5 training.

    This function:
    - Tokenizes the input prompt (`input_text`) and the target label (`target_text`).
    - Pads and truncates both to fixed lengths.
    - Adds tokenized inputs, attention masks, and label IDs to the batch.

    Args:
        batch (dict): A dictionary with 'input_text', 'target_text', and 'label_id'.

    Returns:
        dict: A dictionary containing tokenized input IDs, attention masks, labels, and label ID.
    """
    inputs = tokenizer(
        batch["input_text"],
        truncation=True,
        padding="max_length",
        max_length=150,
    )
    targets = tokenizer(
        batch["target_text"],
        truncation=True,
        padding="max_length",
        max_length=25,
    )
    inputs["labels"]   = targets["input_ids"]
    inputs["label_id"] = batch["label_id"]    # propagate the ID for sampling
    return inputs

train_tok = train_ds.map(tokenize_fn, batched=True)
val_tok   = val_ds.map(tokenize_fn, batched=True)




# Load model & resize embeddings for new tokens
model_name = "t5-base"
model      = T5ForConditionalGeneration.from_pretrained(model_name)
model.resize_token_embeddings(len(tokenizer))   # grow embedding matrix
device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Build class weights for oversampling 
# count per class in train_tok
train_label_ids = np.array(train_tok["label_id"])
class_counts   = np.bincount(train_label_ids, minlength=len(labels))
class_weights  = 1.0 / class_counts
# weight for each sample = weight of its class
sample_weights = class_weights[train_label_ids]
sampler = WeightedRandomSampler(
    weights     = sample_weights,
    num_samples = len(sample_weights),
    replacement = True
)

# DataLoader with sampler
def collate_fn(batch):
    """
    Prepares a batch of samples for T5 model training by padding inputs.

    This function:
    - Pads input IDs, attention masks, and labels to the same length.
    - Converts them into tensors and moves them to the appropriate device (CPU/GPU).

    Args:
        batch (List[dict]): A list of samples with input IDs, attention masks, and labels.

    Returns:
        dict: A dictionary containing padded tensors for 'input_ids', 'attention_mask', and 'labels'.
    """
    input_ids      = [torch.tensor(x["input_ids"]) for x in batch]
    attention_mask = [torch.tensor(x["attention_mask"]) for x in batch]
    labels         = [torch.tensor(x["labels"])      for x in batch]
    # pad
    input_ids      = torch.nn.utils.rnn.pad_sequence(
                          input_ids, batch_first=True, padding_value=tokenizer.pad_token_id)
    attention_mask = torch.nn.utils.rnn.pad_sequence(
                          attention_mask, batch_first=True, padding_value=0)
    labels         = torch.nn.utils.rnn.pad_sequence(
                          labels, batch_first=True, padding_value=-100)
    return {
        "input_ids":      input_ids.to(device),
        "attention_mask": attention_mask.to(device),
        "labels":         labels.to(device),
    }

train_loader = DataLoader(
    train_tok,
    batch_size=8,
    sampler=sampler,
    collate_fn=collate_fn
)
eval_loader  = DataLoader(
    val_tok,
    batch_size=8,
    shuffle=False,
    collate_fn=collate_fn
)

#  Optimizer & scheduler
optimizer   = AdamW(model.parameters(), lr=3e-5)
total_steps = len(train_loader) * 3   # epochs=3
scheduler   = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps   = 0,
    num_training_steps = total_steps
)

#  Training loop
for epoch in range(1, 11):
    model.train()
    total_loss = 0
    for batch in train_loader:
        outputs = model(**batch)
        loss    = outputs.loss
        loss.backward()
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
        total_loss += loss.item()
    print(f"Epoch {epoch} — train loss: {total_loss/len(train_loader):.4f}")

    model.eval()
    eval_loss = 0
    with torch.no_grad():
        for batch in eval_loader:
            eval_loss += model(**batch).loss.item()
    print(f"Epoch {epoch} — eval  loss: {eval_loss/len(eval_loader):.4f}")

# Save
model.save_pretrained("t5-prop-finetuned")
tokenizer.save_pretrained("t5-prop-finetuned")





## For testing on the test data
# Load the test data
test_df = pd.read_csv("/content/propaganda_val.tsv", sep="\t")
print(f"Test set shape: {test_df.shape}")

# Load the fine-tuned model and tokenizer
model_path = "t5-prop-finetuned"
tokenizer = T5Tokenizer.from_pretrained(model_path)
model = T5ForConditionalGeneration.from_pretrained(model_path)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Get the labels
labels = sorted(test_df["label"].unique())
label2id = {lab: i for i, lab in enumerate(labels)}
id2label = {i: lab for i, lab in enumerate(labels)}


# Convert to HuggingFace Dataset and preprocess
test_ds = Dataset.from_pandas(test_df)
test_ds = test_ds.map(
    preprocess_row,
    remove_columns=["label", "tagged_in_context"]
)

# Function to run inference
def run_inference(model, tokenizer, test_ds):
    """
    Runs inference on a test dataset using a fine-tuned T5 model for propaganda span classification.

    This function:
    - Processes each example by extracting the span and tokenizing the input prompt.
    - Generates predictions using the model.
    - Decodes the predictions and matches them to the closest valid label if needed.
    - Collects predicted labels, true labels, and spans for further evaluation.

    Args:
        model (T5ForConditionalGeneration): The fine-tuned T5 model.
        tokenizer (T5Tokenizer): The tokenizer used during training.
        test_ds (Dataset): HuggingFace Dataset containing preprocessed test examples.

    Returns:
        Tuple[List[str], List[str], List[str]]: Predicted labels, true labels, and extracted spans.
    """
    model.eval()
    all_preds = []
    all_true_labels = []
    all_spans = []

    for item in test_ds:
        input_text = item["input_text"]
        true_label = item["target_text"]
        label_id = item["label_id"]

        # Extract the span for analysis
        start_marker = "[SPAN]"
        end_marker = "[/SPAN]"
        start_pos = input_text.find(start_marker)
        end_pos = input_text.find(end_marker)
        span = input_text[start_pos + len(start_marker):end_pos] if start_pos != -1 and end_pos != -1 else ""

        # Tokenize
        inputs = tokenizer(
            input_text,
            return_tensors="pt",
            truncation=True,
            max_length=150,
            padding="max_length",
        ).to(device)

        # Generate prediction
        with torch.no_grad():
            output = model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_length=25,
                num_beams=4,
                early_stopping=True,
            )

        # Decode prediction
        pred_text = tokenizer.decode(output[0], skip_special_tokens=True)

        # Check if the prediction is one of the labels
        if pred_text not in labels:
            # Find the closest match if prediction is not exact
            closest_label = max(labels, key=lambda x: sum(1 for a, b in zip(pred_text, x) if a == b) / max(len(pred_text), len(x)))
            pred_text = closest_label

        all_preds.append(pred_text)
        all_true_labels.append(true_label)
        all_spans.append(span)

    return all_preds, all_true_labels, all_spans

# Run inference
predictions, true_labels, spans = run_inference(model, tokenizer, test_ds)

# Generate classification report
report = classification_report(true_labels, predictions, target_names=labels)
conf_matrix = confusion_matrix(true_labels, predictions, labels=labels)

# Print results
print("\nClassification Report:")
print(report)

print("\nConfusion Matrix:")
print(conf_matrix)

# Additional analysis - show some examples with errors
errors = [(pred, true, span) for pred, true, span in zip(predictions, true_labels, spans) if pred != true]
print(f"\nNumber of errors: {len(errors)} out of {len(predictions)} ({len(errors)/len(predictions):.2%})")

if errors:
    print("\nSample of errors (first 10):")
    for i, (pred, true, span) in enumerate(errors[:10]):
        print(f"{i+1}. Span: '{span}'")
        print(f"   Predicted: {pred}")
        print(f"   Actual: {true}")
        print()

# Calculate F1 score per class

f1_per_class = f1_score(true_labels, predictions, labels=labels, average=None)
for label, f1 in zip(labels, f1_per_class):
    print(f"F1 score for {label}: {f1:.4f}")

# Calculate overall F1 score
f1_micro = f1_score(true_labels, predictions, average='micro')
f1_macro = f1_score(true_labels, predictions, average='macro')
f1_weighted = f1_score(true_labels, predictions, average='weighted')
print(f"\nOverall F1 scores:")
print(f"Micro-average F1: {f1_micro:.4f}")
print(f"Macro-average F1: {f1_macro:.4f}")
print(f"Weighted-average F1: {f1_weighted:.4f}")