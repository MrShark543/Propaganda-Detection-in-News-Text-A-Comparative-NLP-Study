import pandas as pd
import torch
import re
import contractions
from torch.utils.data import Dataset, DataLoader
from transformers import RobertaTokenizer, RobertaForSequenceClassification
from tqdm import tqdm
from sklearn.metrics import classification_report
import warnings
warnings.filterwarnings('ignore')

# Define abbreviation mappings
ABB_MEANING = {
    'U.S.': "United States of America",
    'F.B.I.': "Federal Bureau of Investigation",
    'C.I.A.': "Central Intelligence Agency",
    'D.C.': "District of Columbia",
    'i.e.': "that is",
    'U.N.': "United Nations",
    'e.g.': "for example",

}

# Process the abbreviations for regex pattern
ESCAPED_KEYS = sorted(ABB_MEANING.keys(), key=len, reverse=True)
PATTERN = re.compile(
    r'(?<!\w)(' + '|'.join(map(re.escape, ESCAPED_KEYS)) + r')(?!\w)'
)

def expand_abbreviation(m):
    """Return the expanded form of an abbreviation."""
    return ABB_MEANING[m.group(1)]

def preprocess_text(text):
    """
    Preprocess the text by:
    1. Handling non-string inputs
    2. Expanding contractions
    3. Replacing abbreviations
    4. Converting to lowercase
    5. Removing extra whitespace
    """
    # Handle non-string inputs
    if not isinstance(text, str):
        if pd.isna(text):  # Handle NaN values
            return ""
        text = str(text)  # Convert numbers or other types to string

    # Fix contractions
    text = contractions.fix(text)

    # Replace abbreviations
    text = re.sub(PATTERN, expand_abbreviation, text)

    # Convert to lowercase
    text = text.lower()

    # Remove extra whitespace
    text = ' '.join(text.split())

    return text


class PropagandaInferenceDataset(Dataset):
    """
    A PyTorch Dataset for preparing text data for propaganda technique inference.

    This dataset is designed for use with transformer-based models like RoBERTa.
    It tokenizes preprocessed text samples and optionally includes labels for evaluation.

    Args:
        texts (List[str]): List of preprocessed input texts to be tokenized.
        labels (Optional[List[int]]): List of integer-encoded labels corresponding to the texts.
                                      If None, the dataset is used purely for inference.
        tokenizer (transformers.PreTrainedTokenizer): Tokenizer from Hugging Face Transformers.
        max_length (int): Maximum sequence length for tokenization (default: 120).

    Returns:
        Each item is a dictionary containing:
            - 'input_ids': Tensor of token IDs (size: max_length)
            - 'attention_mask': Tensor indicating real tokens vs padding
            - 'labels': (Optional) Tensor of label if provided
    """

    def __init__(self, texts, labels=None, tokenizer=None, max_length=120):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.has_labels = labels is not None

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])

        # Tokenize the text
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )

        item = {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
        }

        # If labels are provided, add them to the batch
        if self.has_labels:
            item['labels'] = torch.tensor(self.labels[idx], dtype=torch.long)

        return item

def load_and_preprocess_inference_data(file_path, span_column, label_column=None, tag_format='BOS_EOS'):
    """
    Load data from file and preprocess it for inference.

    Args:
        file_path: Path to the data file
        span_column: Column name containing the text spans
        label_column: Column name containing the gold labels (if any)
        tag_format: Format of tags used in the data ('BOS_EOS' or 'none')

    Returns:
        DataFrame with preprocessed data
    """
    # Determine file type and load accordingly
    if file_path.endswith('.csv'):
        df = pd.read_csv(file_path)
    elif file_path.endswith('.tsv'):
        df = pd.read_csv(file_path, sep='\t')
    elif file_path.endswith('.xlsx'):
        df = pd.read_excel(file_path)
    else:
        raise ValueError(f"Unsupported file format: {file_path}")

    # Check for required columns
    if span_column not in df.columns:
        raise ValueError(f"Required column '{span_column}' not found in {file_path}")

    df['span_text'] = df[span_column]

    # Preprocess the spans
    df['processed_text'] = df['span_text'].apply(preprocess_text)

    return df

def run_inference(model, data_loader, device, propaganda_types):
    """
    Run inference using the trained model.

    Args:
        model: The trained model
        data_loader: DataLoader with inference data
        device: Device to run inference on
        propaganda_types: List of propaganda technique names

    Returns:
        Tuple of (predicted_indices, predictions)
    """
    model.eval()
    all_preds = []
    all_pred_indices = []
    all_labels = []

    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Running inference"):
            # Move batch to device
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            
            # If we have labels in the batch, collect them
            if 'labels' in batch:
                all_labels.extend(batch['labels'].cpu().numpy())

            # Forward pass
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )

            # Get predictions
            logits = outputs.logits
            pred_indices = torch.argmax(logits, dim=1).cpu().numpy()
            all_pred_indices.extend(pred_indices)

            # Convert numeric predictions to label names
            pred_labels = [propaganda_types[pred] for pred in pred_indices]
            all_preds.extend(pred_labels)

    return all_pred_indices, all_preds, all_labels

def main_inference(model_path, file_path, span_column, label_column=None, tag_format='BOS_EOS'):
    """
    Main function to run inference and print classification report.

    Args:
        model_path: Path to the saved model
        file_path: Path to the data file
        span_column: Column name containing the text spans
        label_column: Column name containing the gold labels (if any)
        tag_format: Format of tags used in the data ('BOS_EOS' or 'none')
    """
    # Define propaganda types in the same order as during training
    propaganda_types = [
        "flag_waving",
        "appeal_to_fear_prejudice",
        "causal_oversimplification",
        "doubt",
        "exaggeration,minimisation",
        "loaded_language",
        "name_calling,labeling",
        "repetition",
        "not_propaganda"
    ]

    # Create label mapping
    label_mapping = {label: idx for idx, label in enumerate(propaganda_types)}

    # Load and preprocess data
    print("Loading and preprocessing inference data...")
    inference_df = load_and_preprocess_inference_data(
        file_path,
        span_column,
        label_column,
        tag_format
    )

    # Load model and tokenizer
    print("Loading model and tokenizer...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    try:
        tokenizer = RobertaTokenizer.from_pretrained('roberta-large')
        model = RobertaForSequenceClassification.from_pretrained(
            'roberta-large',
            num_labels=len(propaganda_types)
        )

        # Load the trained model weights
        model.load_state_dict(torch.load(model_path, map_location=device))
        model = model.to(device)

    except Exception as e:
        print(f"Error loading model: {e}")
        return None

    # Prepare for inference and evaluation
    texts = inference_df['processed_text'].tolist()

    # If we have gold labels, map them to indices
    gold_indices = None

    if label_column and label_column in inference_df.columns:
        gold_labels = inference_df[label_column].tolist()

        # Map string labels to indices
        gold_indices = []
        for label in gold_labels:
            if label in label_mapping:
                gold_indices.append(label_mapping[label])
            else:
                print(f"Warning: Label '{label}' not found in mapping. Using -1.")
                gold_indices.append(-1)  # Invalid index to be filtered out later

    # Create dataset and dataloader
    print("Creating inference data loader...")
    inference_dataset = PropagandaInferenceDataset(
        texts=texts,
        labels=gold_indices if gold_indices else None,
        tokenizer=tokenizer,
        max_length=120
    )

    inference_loader = DataLoader(
        inference_dataset,
        batch_size=8,
        shuffle=False
    )

    # Run inference
    print("Running inference...")
    pred_indices, predictions, gold_labels = run_inference(
        model,
        inference_loader,
        device,
        propaganda_types
    )

    # Print classification report if we have gold labels
    if gold_indices and len(gold_indices) > 0:
        # Filter out invalid labels (-1)
        valid_indices = [i for i, idx in enumerate(gold_indices) if idx != -1]
        
        if valid_indices:
            valid_gold = [gold_indices[i] for i in valid_indices]
            valid_pred = [pred_indices[i] for i in valid_indices]
            
            print("\n===== CLASSIFICATION REPORT =====")
            report = classification_report(
                valid_gold,
                valid_pred,
                target_names=propaganda_types,
                digits=4
            )
            print(report)
        else:
            print("No valid gold labels for evaluation.")
    else:
        print("No gold labels available for classification report.")

    # Return predictions and add to DataFrame
    inference_df['predicted_technique'] = predictions
    inference_df.to_csv(file_path.replace('.', '_with_predictions.'), index=False)
    
    print(f"Predictions saved to DataFrame. Total predictions: {len(predictions)}")
    
    return inference_df




MODEL_PATH = "/content/best_propaganda_model.pth"  # Path to trained model
FILE_PATH = "/content/results2.csv"  # Path to test dataset
SPAN_COLUMN = "predicted_span"  # Column containing text spans
LABEL_COLUMN = "Label"  # Column containing gold labels

df_with_predictions = main_inference(
    MODEL_PATH,
    FILE_PATH,
    SPAN_COLUMN,
    LABEL_COLUMN

)