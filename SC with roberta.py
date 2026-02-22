import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, random_split
from transformers import RobertaTokenizerFast, RobertaForTokenClassification
from torch.optim import AdamW
from tqdm import tqdm





# Set maximum sequence length
MAX_LENGTH = 256

# Define BIO tags
label2id = {"O": 0, "B-PROP": 1, "I-PROP": 2}
id2label = {v: k for k, v in label2id.items()}

# Load tokenizer
tokenizer = RobertaTokenizerFast.from_pretrained("roberta-base")

# Custom Dataset class
class PropagandaDataset(Dataset):
    """
    A custom PyTorch Dataset for token-level propaganda span detection using BIO tagging.

    This dataset processes input sentences with optional labeled propaganda spans, identified using
    <BOS> (beginning of span) and <EOS> (end of span) markers in the text. It generates tokenized
    inputs using the RoBERTa tokenizer and assigns BIO labels (B-PROP, I-PROP, O) to each token 
    based on the span location. It also handles non-propaganda examples by assigning all tokens 
    the "O" label.

    Attributes:
        texts (List[str]): List of input sentences, some containing <BOS> and <EOS> markers.
        labels (List[str] or None): Corresponding propaganda technique labels (optional).
        tokenizer (transformers.PreTrainedTokenizer): Tokenizer used to convert text to tokens.
        max_length (int): Maximum sequence length for tokenization.
        encoded_data (List[Dict]): List of tokenized inputs with attention masks, labels, and metadata.

    Methods:
        __len__(): Returns the number of data samples.
        __getitem__(idx): Returns the tokenized input, attention mask, label, and metadata for a sample.
    """
    def __init__(self, texts, labels=None, tokenizer=None, max_length=MAX_LENGTH):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.encoded_data = self.preprocess()

    def preprocess(self):
        """
        Preprocesses the input texts to generate tokenized inputs and BIO-encoded labels.

        For each sentence in the dataset, this method:
        - Identifies the propaganda span marked by <BOS> and <EOS> tags.
        - Removes these tags to produce clean text.
        - Uses the tokenizer to generate token IDs and offset mappings.
        - Creates corresponding BIO labels (B-PROP, I-PROP, O) aligned with token spans.
        - Skips examples where the propaganda span is entirely lost due to truncation.
        - Handles non-propaganda sentences by labeling all tokens as "O".

        Returns:
            List[Dict]: A list of dictionaries containing tokenized inputs, attention masks, 
                        BIO labels, original text, offset mappings, and propaganda technique labels.
        """
        encoded_data = []
        skipped_count = 0

        for i, text in enumerate(self.texts):
            # Extract propaganda span using <BOS> and <EOS> markers
            if "<BOS>" in text and "<EOS>" in text:
                label = self.labels[i] if self.labels is not None else None

                start_marker = text.find("<BOS>")
                end_marker = text.find("<EOS>")

                # Clean text by removing markers
                clean_text = text.replace("<BOS>", "").replace("<EOS>", "")

                # Calculate propaganda span positions
                propaganda_start = start_marker
                if end_marker > start_marker:
                    propaganda_end = end_marker - 5  # Adjust for removal of <BOS>
                else:
                    propaganda_end = end_marker - 5
                    propaganda_start, propaganda_end = min(propaganda_start, propaganda_end), max(propaganda_start, propaganda_end)

                # Tokenize the text
                encoded = self.tokenizer(
                    clean_text,
                    truncation=True,
                    max_length=self.max_length,
                    return_offsets_mapping=True,
                    padding="max_length",
                    return_tensors="pt"
                )

                # Create BIO tags
                labels = torch.ones(encoded['input_ids'].shape, dtype=torch.long) * -100  # Initialize with -100 (ignored)
                span_included = False

                for j, (start, end) in enumerate(encoded['offset_mapping'][0]):
                    # Skip special tokens
                    if start == 0 and end == 0:
                        continue

                    # Check for overlap with propaganda span
                    if end > propaganda_start and start < propaganda_end:
                        span_included = True
                        # B tag for first token in span
                        if start <= propaganda_start < end or (j > 0 and
                                                           encoded['offset_mapping'][0][j-1][1] <= propaganda_start and
                                                           start > propaganda_start):
                            labels[0, j] = label2id["B-PROP"]
                        else:
                            labels[0, j] = label2id["I-PROP"]
                    else:
                        labels[0, j] = label2id["O"]

                # Skip if no part of propaganda span is included
                if not span_included and "<BOS>" in text:
                    skipped_count += 1
                    continue

                # Store all relevant data
                item = {
                    'input_ids': encoded['input_ids'][0],
                    'attention_mask': encoded['attention_mask'][0],
                    'labels': labels[0],
                    'text': clean_text,
                    'offset_mapping': encoded['offset_mapping'][0],
                    'propaganda_technique': label
                }
                encoded_data.append(item)
            else:
                # Handle "not propaganda" examples
                clean_text = text
                encoded = self.tokenizer(
                    clean_text,
                    truncation=True,
                    max_length=self.max_length,
                    return_offsets_mapping=True,
                    padding="max_length",
                    return_tensors="pt"
                )

                # All "O" tags for not propaganda
                labels = torch.ones(encoded['input_ids'].shape, dtype=torch.long) * -100
                for j, (start, end) in enumerate(encoded['offset_mapping'][0]):
                    if start != 0 or end != 0:  # Not a special token
                        labels[0, j] = label2id["O"]

                item = {
                    'input_ids': encoded['input_ids'][0],
                    'attention_mask': encoded['attention_mask'][0],
                    'labels': labels[0],
                    'text': clean_text,
                    'offset_mapping': encoded['offset_mapping'][0],
                    'propaganda_technique': "not propaganda" if self.labels is not None else None
                }
                encoded_data.append(item)

        print(f"Skipped {skipped_count} examples where propaganda span was completely lost due to truncation")
        return encoded_data

    def __len__(self):
        return len(self.encoded_data)

    def __getitem__(self, idx):
        return self.encoded_data[idx]

# Data collator for batching
def collate_fn(batch):
    """
    Custom collate function to prepare batches for token-level sequence classification.

    This function takes a list of dataset samples (each containing input IDs, attention masks, 
    labels, and metadata) and stacks them into batch tensors. It also separates out metadata 
    such as the original text, token offset mappings, and propaganda technique labels for 
    evaluation and span reconstruction.

    Args:
        batch (List[Dict]): A list of samples returned by the dataset's __getitem__ method.

    Returns:
        Tuple[Dict[str, torch.Tensor], Dict[str, List]]:
            - A dictionary with batched tensors for 'input_ids', 'attention_mask', and 'labels'.
            - A dictionary containing metadata: 'texts', 'offset_mappings', and 'propaganda_techniques'.
    """
    input_ids = torch.stack([item['input_ids'] for item in batch])
    attention_mask = torch.stack([item['attention_mask'] for item in batch])
    labels = torch.stack([item['labels'] for item in batch])

    # Store metadata separately 
    texts = [item['text'] for item in batch]
    offset_mappings = [item['offset_mapping'] for item in batch]
    propaganda_techniques = [item['propaganda_technique'] for item in batch]

    # Only return tensor inputs to the model
    batch_inputs = {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'labels': labels
    }

    # Return metadata separately
    metadata = {
        'texts': texts,
        'offset_mappings': offset_mappings,
        'propaganda_techniques': propaganda_techniques
    }

    return batch_inputs, metadata

# Function to convert predictions back to spans
def predictions_to_spans(text, predictions, offset_mapping):
    """
    Custom collate function to prepare batches for token-level sequence classification.

    This function takes a list of dataset samples (each containing input IDs, attention masks, 
    labels, and metadata) and stacks them into batch tensors. It also separates out metadata 
    such as the original text, token offset mappings, and propaganda technique labels for 
    evaluation and span reconstruction.

    Args:
        batch (List[Dict]): A list of samples returned by the dataset's __getitem__ method.

    Returns:
        Tuple[Dict[str, torch.Tensor], Dict[str, List]]:
            - A dictionary with batched tensors for 'input_ids', 'attention_mask', and 'labels'.
            - A dictionary containing metadata: 'texts', 'offset_mappings', and 'propaganda_techniques'.
    """
    spans = []
    current_span = None

    for i, (pred, offset) in enumerate(zip(predictions, offset_mapping)):
        # Skip special tokens and ignored indices
        if offset[0] == 0 and offset[1] == 0 or pred == -100:
            continue

        if pred == label2id["B-PROP"]:
            if current_span is not None:
                spans.append(current_span)
            current_span = {"start": offset[0].item(), "end": offset[1].item()}
        elif pred == label2id["I-PROP"] and current_span is not None:
            current_span["end"] = offset[1].item()
        elif current_span is not None:
            spans.append(current_span)
            current_span = None

    if current_span is not None:
        spans.append(current_span)

    # Convert character positions to text spans
    text_spans = []
    for span in spans:
        text_spans.append({
            "start": span["start"],
            "end": span["end"],
            "text": text[span["start"]:span["end"]]
        })

    return text_spans

# Post-process spans for better accuracy
def post_process_spans(text, spans):
    """
    Apply post-processing to improve predicted spans
    """
    if not spans:
        return []

    # Sort spans by start position
    spans = sorted(spans, key=lambda x: x["start"])

    # Merge overlapping or adjacent spans
    merged_spans = []
    if spans:
        current_span = spans[0]

        for span in spans[1:]:
            # Check if spans overlap or are adjacent (with 1-2 char gap)
            if span["start"] <= current_span["end"] + 2:
                # Merge them
                current_span["end"] = max(current_span["end"], span["end"])
            else:
                merged_spans.append(current_span)
                current_span = span

        merged_spans.append(current_span)

    # Ensure spans start and end with alphanumeric characters
    refined_spans = []
    for span in merged_spans:
        start = span["start"]
        end = span["end"]

        # Adjust start to first alphanumeric character
        while start < end and start < len(text) and not text[start].isalnum():
            start += 1

        # Adjust end to last alphanumeric character
        while end > start and end > 0 and not text[end-1].isalnum():
            end -= 1

        # 3. Expand to include quotation marks if applicable
        if start > 0 and text[start-1] == '"':
            start -= 1
        if end < len(text) and text[end] == '"':
            end += 1

        # Only keep span if it has meaningful length
        if end - start >= 2:  # Filter spans of length 1
            refined_spans.append({
                "start": start,
                "end": end,
                "text": text[start:end]
            })

    return refined_spans

# Evaluation function
def evaluate(model, dataloader, device):
    """
    Evaluates a token classification model on a given dataset and computes span-level metrics.

    This function runs inference over the provided dataloader, converts token-level predictions and 
    ground truth labels into character-level spans, and computes precision, recall, and F1-score 
    based on overlapping spans.

    Args:
        model (torch.nn.Module): The trained token classification model.
        dataloader (torch.utils.data.DataLoader): Dataloader providing batches of inputs and metadata.
        device (torch.device): Device (CPU or GPU) to run the evaluation on.

    Returns:
        dict: A dictionary containing span-level evaluation metrics:
            - "precision": Proportion of predicted spans that overlap with true spans.
            - "recall": Proportion of true spans that are correctly predicted.
            - "f1": Harmonic mean of precision and recall.
    """

    model.eval()

    all_predictions = []
    all_labels = []
    all_texts = []
    all_offsets = []

    with torch.no_grad():
        for batch, metadata in tqdm(dataloader, desc="Evaluating"):
            # Move batch to device
            batch = {k: v.to(device) for k, v in batch.items()}

            # Forward pass
            outputs = model(**batch)
            logits = outputs.logits

            # Get predictions
            predictions = torch.argmax(logits, dim=2)

            # Store results
            all_predictions.extend(predictions.detach().cpu())
            all_labels.extend(batch['labels'].detach().cpu())
            all_texts.extend(metadata['texts'])
            all_offsets.extend(metadata['offset_mappings'])

    # Calculate span-level metrics
    correct_spans = 0
    total_pred_spans = 0
    total_true_spans = 0

    for preds, labels, text, offsets in zip(all_predictions, all_labels, all_texts, all_offsets):
        # Convert token predictions to spans
        pred_spans = predictions_to_spans(text, preds, offsets)
        pred_spans = post_process_spans(text, pred_spans)

        # Convert token labels to spans
        true_spans = predictions_to_spans(text, labels, offsets)

        # Compare spans
        for pred_span in pred_spans:
            for true_span in true_spans:
                # Consider a span correct if there's significant overlap
                overlap_start = max(pred_span["start"], true_span["start"])
                overlap_end = min(pred_span["end"], true_span["end"])

                if overlap_end > overlap_start:
                    overlap_length = overlap_end - overlap_start
                    pred_length = pred_span["end"] - pred_span["start"]
                    true_length = true_span["end"] - true_span["start"]

                    overlap_ratio = overlap_length / min(pred_length, true_length)

                    if overlap_ratio > 0.5:  # More than 50% overlap
                        correct_spans += 1
                        break

        total_pred_spans += len(pred_spans)
        total_true_spans += len(true_spans)

    # Calculate standard metrics
    precision = correct_spans / total_pred_spans if total_pred_spans > 0 else 0
    recall = correct_spans / total_true_spans if total_true_spans > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1
    }

# Main training function
def train_propaganda_detector():
    """
    Trains a RoBERTa-based token classification model to detect propaganda spans using BIO tagging.

    This function performs the following steps:
    - Loads and processes the training data from a TSV file.
    - Initializes the custom `PropagandaDataset` and splits it into training and validation sets.
    - Prepares DataLoaders for efficient batching and shuffling.
    - Initializes a `RobertaForTokenClassification` model with the appropriate number of labels.
    - Trains the model for a set number of epochs, tracking the best model based on F1 score.
    - Evaluates the model on the validation set after each epoch using span-level metrics.
    - Saves the best-performing model to disk.

    Returns:
        model (RobertaForTokenClassification): The trained model with the best validation F1 score.
        best_f1 (float): The highest F1 score achieved on the validation set.
    """
    # Load data
    df = pd.read_csv('propaganda_train.tsv', sep='\t')

    print(f"Total data: {len(df)} examples")

    # Create full dataset
    full_dataset = PropagandaDataset(
        texts=df['tagged_in_context'].tolist(),
        labels=df['label'].tolist(),
        tokenizer=tokenizer
    )

    print(f"Processed data: {len(full_dataset)} examples")

    # Split into training and validation (80/20)
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size

    train_dataset, val_dataset = random_split(
        full_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)  # For reproducibility
    )

    print(f"Training set: {len(train_dataset)} examples")
    print(f"Validation set: {len(val_dataset)} examples")

    # Create dataloaders
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=8,
        shuffle=True,
        collate_fn=collate_fn
    )

    val_dataloader = DataLoader(
        val_dataset,
        batch_size=8,
        shuffle=False,
        collate_fn=collate_fn
    )

    # Initialize model
    model = RobertaForTokenClassification.from_pretrained(
        "roberta-base",
        num_labels=len(label2id),
        id2label=id2label,
        label2id=label2id
    )

    # Set up training
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    optimizer = AdamW(model.parameters(), lr=2e-5)
    num_epochs = 3

    # Training loop
    best_f1 = 0
    best_model_state = None

    for epoch in range(num_epochs):
        # Training phase
        model.train()
        total_loss = 0

        progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{num_epochs} [Training]")
        for batch, _ in progress_bar:
            # Move batch to device
            batch = {k: v.to(device) for k, v in batch.items()}

            # Forward pass
            outputs = model(**batch)
            loss = outputs.loss

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            progress_bar.set_postfix({"loss": loss.item()})

        avg_loss = total_loss / len(train_dataloader)
        print(f"Epoch {epoch+1}/{num_epochs} - Average training loss: {avg_loss:.4f}")

        # Evaluation phase
        metrics = evaluate(model, val_dataloader, device)
        print(f"Evaluation metrics - Precision: {metrics['precision']:.4f}, Recall: {metrics['recall']:.4f}, F1: {metrics['f1']:.4f}")

        # Save best model
        if metrics['f1'] > best_f1:
            best_f1 = metrics['f1']
            best_model_state = model.state_dict().copy()
            print(f"New best F1 score: {best_f1:.4f} - Saving model")

    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print(f"Loaded best model with F1 score: {best_f1:.4f}")

    # Save the best model
    torch.save(model.state_dict(), "propaganda_detection_model.pt")
    print("Model saved!")

    return model, best_f1

#for training the model
model, best_f1 = train_propaganda_detector()



#For prediction on the test data
def evaluate_test_data(model_path, test_file_path, output_excel_path):
    """
    Evaluates a trained RoBERTa-based token classification model on a test dataset for propaganda span detection.

    This function:
    - Loads the saved model and tokenizer.
    - Reads test data from a TSV file, parsing tagged propaganda spans.
    - Performs tokenization and predicts token-level BIO labels.
    - Converts predictions to character-level spans and applies post-processing.
    - Compares predicted spans with gold spans to compute precision, recall, and F1-score.
    - Saves a detailed results table (including gold spans, predictions, and correctness) to an Excel file.

    Args:
        model_path (str): Path to the saved model weights (".pt" file).
        test_file_path (str): Path to the TSV test file containing 'tagged_in_context' and optional 'label' columns.
        output_excel_path (str): Path to save the evaluation results as an Excel file.

    Returns:
        dict: A dictionary containing:
            - "precision" (float): Precision score based on span-level overlap.
            - "recall" (float): Recall score based on span-level overlap.
            - "f1" (float): F1-score based on span-level overlap.
            - "results_df" (pd.DataFrame): DataFrame containing per-sample prediction results.
    """
    # Load model and tokenizer
    tokenizer = RobertaTokenizerFast.from_pretrained("roberta-base")

    # Define BIO tags
    label2id = {"O": 0, "B-PROP": 1, "I-PROP": 2}
    id2label = {v: k for k, v in label2id.items()}

    # Load the model
    model = RobertaForTokenClassification.from_pretrained(
        "roberta-base",
        num_labels=len(label2id),
        id2label=id2label,
        label2id=label2id
    )
    model.load_state_dict(torch.load(model_path))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    # Load test data
    test_df = pd.read_csv(test_file_path, sep='\t')

    # Prepare results
    results = []

    # Set up metrics tracking
    correct_spans = 0
    total_pred_spans = 0
    total_true_spans = 0

    # Process each test example
    for _, row in tqdm(test_df.iterrows(), total=len(test_df), desc="Evaluating test data"):
        text = row["tagged_in_context"]

        # Extract gold span
        gold_span_text = ""
        if "<BOS>" in text and "<EOS>" in text:
            start_idx = text.find("<BOS>") + 5
            end_idx = text.find("<EOS>")
            if start_idx < end_idx:
                gold_span_text = text[start_idx:end_idx]

        # Clean text for model input
        clean_text = text.replace("<BOS>", "").replace("<EOS>", "")

        # Get gold span positions
        gold_spans = []
        if "<BOS>" in text and "<EOS>" in text:
            start_marker = text.find("<BOS>")
            end_marker = text.find("<EOS>")

            if end_marker > start_marker:
                gold_start = start_marker
                gold_end = end_marker - 5  # Adjust for removal of <BOS>
            else:
                gold_start = min(start_marker, end_marker)
                gold_end = max(start_marker, end_marker) - 5

            gold_spans.append({
                "start": gold_start,
                "end": gold_end,
                "text": clean_text[gold_start:gold_end]
            })

        # Tokenize the text
        inputs = tokenizer(
            clean_text,
            return_tensors="pt",
            return_offsets_mapping=True,
            truncation=True,
            max_length=256
        )

        offset_mapping = inputs.pop("offset_mapping")
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # Get predictions
        with torch.no_grad():
            outputs = model(**inputs)

        predictions = torch.argmax(outputs.logits, dim=2)[0].cpu().numpy()

        # Convert predictions to spans
        pred_spans = []
        current_span = None

        for i, (pred, offset) in enumerate(zip(predictions, offset_mapping[0])):
            # Skip special tokens
            if offset[0] == 0 and offset[1] == 0:
                continue

            offset_start, offset_end = offset.cpu().numpy().tolist()

            if pred == label2id["B-PROP"]:
                if current_span is not None:
                    pred_spans.append(current_span)
                current_span = {"start": offset_start, "end": offset_end}
            elif pred == label2id["I-PROP"] and current_span is not None:
                current_span["end"] = offset_end
            elif current_span is not None:
                pred_spans.append(current_span)
                current_span = None

        if current_span is not None:
            pred_spans.append(current_span)

        # Post-process predicted spans
        processed_pred_spans = post_process_spans(clean_text, pred_spans)

        # Extract span text
        pred_span_texts = [span["text"] for span in processed_pred_spans]
        pred_span_text = " | ".join(pred_span_texts) if pred_span_texts else ""

        # Compare spans for metrics
        for pred_span in processed_pred_spans:
            for gold_span in gold_spans:
                # Consider a span correct if there's significant overlap
                overlap_start = max(pred_span["start"], gold_span["start"])
                overlap_end = min(pred_span["end"], gold_span["end"])

                if overlap_end > overlap_start:
                    overlap_length = overlap_end - overlap_start
                    pred_length = pred_span["end"] - pred_span["start"]
                    gold_length = gold_span["end"] - gold_span["start"]

                    overlap_ratio = overlap_length / min(pred_length, gold_length)

                    if overlap_ratio > 0.5:  # More than 50% overlap
                        correct_spans += 1
                        break

        total_pred_spans += len(processed_pred_spans)
        total_true_spans += len(gold_spans)

        # Save results
        results.append({
            "Statement": clean_text,
            "Gold Label": gold_span_text,
            "Prediction": pred_span_text,
            "Label": row["label"] if "label" in row else "",
            "Is Correct Match": any(overlap_significant(pred_span, gold_span)
                                    for pred_span in processed_pred_spans
                                    for gold_span in gold_spans) if gold_spans and processed_pred_spans else False
        })

    # Calculate metrics
    precision = correct_spans / total_pred_spans if total_pred_spans > 0 else 0
    recall = correct_spans / total_true_spans if total_true_spans > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    # Print metrics
    print(f"Test Evaluation Results:")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")

    # Save results to Excel for viewing later
    results_df = pd.DataFrame(results)
    results_df.to_excel(output_excel_path, index=False)
    print(f"Results saved to {output_excel_path}")

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "results_df": results_df
    }

def overlap_significant(span1, span2, threshold=0.5):
    """Check if two spans have significant overlap"""
    overlap_start = max(span1["start"], span2["start"])
    overlap_end = min(span1["end"], span2["end"])

    if overlap_end > overlap_start:
        overlap_length = overlap_end - overlap_start
        span1_length = span1["end"] - span1["start"]
        span2_length = span2["end"] - span2["start"]

        overlap_ratio = overlap_length / min(span1_length, span2_length)
        return overlap_ratio >= threshold

    return False

def post_process_spans(text, spans):
    """
    Apply post-processing to improve predicted spans:
    1. Merge overlapping or adjacent spans
    2. Ensure spans start and end with alphanumeric characters
    3. Expand to include quotation marks if applicable
    4. Filter out very short spans (likely noise)
    """
    if not spans:
        return []

    # Sort spans by start position
    spans = sorted(spans, key=lambda x: x["start"])

    # Merge overlapping or adjacent spans
    merged_spans = []
    current_span = spans[0]

    for span in spans[1:]:
        # Check if spans overlap or are adjacent (with 1-2 char gap)
        if span["start"] <= current_span["end"] + 2:
            # Merge them
            current_span["end"] = max(current_span["end"], span["end"])
        else:
            merged_spans.append(current_span)
            current_span = span

    merged_spans.append(current_span)

    # Ensure spans start and end with alphanumeric characters
    refined_spans = []
    for span in merged_spans:
        start = span["start"]
        end = span["end"]

        # Adjust start to first alphanumeric character
        while start < end and start < len(text) and not text[start].isalnum():
            start += 1

        # Adjust end to last alphanumeric character
        while end > start and end > 0 and not text[end-1].isalnum():
            end -= 1

        # Expand to include quotation marks if applicable
        if start > 0 and text[start-1] == '"':
            start -= 1
        if end < len(text) and text[end] == '"':
            end += 1

        # Only keep span if it has meaningful length
        if end - start >= 2:  # Filter spans of length 1
            refined_spans.append({
                "start": start,
                "end": end,
                "text": text[start:end]
            })

    return refined_spans



model_path = "./propaganda_detection_model.pt"  # Path to saved model
test_file_path = "./propaganda_val.tsv"         # Path to test TSV file
output_excel_path = "./propaganda_results.xlsx" # Path to save results

# Run evaluation
results = evaluate_test_data(model_path, test_file_path, output_excel_path)

    