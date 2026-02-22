import numpy as np
import pandas as pd
import re
import contractions
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import RobertaTokenizer, RobertaForSequenceClassification, Adafactor
from transformers import get_linear_schedule_with_warmup
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score, classification_report
from tqdm import tqdm
import warnings
import os
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
    'N.A.T.O.': "North Atlantic Treaty Organization",
    'W.H.O.': "World Health Organization"
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
    1. Expanding contractions
    2. Replacing abbreviations
    3. Converting to lowercase
    4. Removing extra whitespace
    """
    # Fix contractions
    text = contractions.fix(text)

    # Replace abbreviations
    text = re.sub(PATTERN, expand_abbreviation, text)

    # Convert to lowercase
    text = text.lower()

    # Remove extra whitespace
    text = ' '.join(text.split())

    return text

def extract_propaganda_span(text):
    """
    Extract the propaganda span marked by <BOS> and <EOS> tags.

    Args:
        text (str): The input string containing optional <BOS> and <EOS> tags.

    Returns:
        str: The substring between <BOS> and <EOS> if both are present;
             otherwise, returns the original text.

    """

    match = re.search(r'<BOS>\s*(.*?)\s*<EOS>', text)
    if match:
        return match.group(1)
    return text  # If no tags found, return the original text

def load_and_preprocess_data(file_path, label_mapping=None):
    """
    Load data from tsv file and preprocess it.

    Args:
        file_path: Path to the TSV file
        label_mapping: Dictionary mapping label names to numeric indices

    Returns:
        DataFrame with preprocessed data
    """
    # Read the TSV file
    try:
        df = pd.read_csv(file_path, sep='\t', quotechar='|')
    except Exception as e:
        print(f"Error reading file {file_path}: {e}")
        print("Attempting to read with different parameters...")
        try:
            df = pd.read_csv(file_path, sep='\t')
        except Exception as e2:
            print(f"Second attempt failed: {e2}")
            raise

    # Print column names to debug
    print(f"Columns in the loaded data: {df.columns.tolist()}")

    # Check for required columns
    if 'tagged_in_context' not in df.columns:
        raise ValueError(f"Required column 'tagged_in_context' not found in {file_path}")
    if 'label' not in df.columns:
        raise ValueError(f"Required column 'label' not found in {file_path}")

    # Extract propaganda spans
    df['span_text'] = df['tagged_in_context'].apply(extract_propaganda_span)

    # Preprocess the spans
    df['processed_text'] = df['span_text'].apply(preprocess_text)

    # Map labels to indices if mapping is provided
    if label_mapping is not None:
        # Check for labels that aren't in the mapping
        unknown_labels = set(df['label'].unique()) - set(label_mapping.keys())
        if unknown_labels:
            print(f"Warning: Found {len(unknown_labels)} unknown labels: {unknown_labels}")
            print("These will be assigned NaN values.")

        df['label_idx'] = df['label'].map(label_mapping)

        # Print label distribution
        print("\nLabel distribution:")
        label_counts = df['label'].value_counts()
        for label, count in label_counts.items():
            print(f"  {label}: {count}")

    return df

class PropagandaDataset(Dataset):
    """
    A custom PyTorch Dataset for loading propaganda classification data.

    This dataset is designed to work with Hugging Face's tokenizer and prepares 
    the text-label pairs for input into a transformer model like RoBERTa.

    Args:
        texts (List[str]): A list of input texts (already preprocessed).
        labels (List[int]): A list of integer labels corresponding to the texts.
        tokenizer (transformers.PreTrainedTokenizer): A Hugging Face tokenizer to tokenize the input texts.
        max_length (int): Maximum length of tokenized sequences (default is 120).

    Returns:
        A dictionary per sample with:
            - 'input_ids': Tensor of token IDs padded/truncated to max_length.
            - 'attention_mask': Tensor indicating which tokens are padding.
            - 'labels': Tensor containing the label as an integer.

    """

    def __init__(self, texts, labels, tokenizer, max_length=120):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]

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

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

def create_data_loaders(df, tokenizer, batch_size=4, max_length=120, train_size=0.9):
    """
    Create train and validation DataLoaders from a DataFrame.

    Args:
        df: DataFrame with preprocessed data
        tokenizer: Tokenizer to convert texts to token IDs
        batch_size: Batch size for DataLoaders
        max_length: Maximum sequence length (with padding)
        train_size: Proportion of data to use for training

    Returns:
        train_loader, val_loader: DataLoaders for training and validation data
    """
    # Split into training and validation sets
    train_df, val_df = train_test_split(
        df,
        train_size=train_size,
        random_state=42,
        stratify=df['label_idx'] if 'label_idx' in df.columns else None
    )

    # Create datasets
    train_dataset = PropagandaDataset(
        texts=train_df['processed_text'].tolist(),
        labels=train_df['label_idx'].tolist(),
        tokenizer=tokenizer,
        max_length=max_length
    )

    val_dataset = PropagandaDataset(
        texts=val_df['processed_text'].tolist(),
        labels=val_df['label_idx'].tolist(),
        tokenizer=tokenizer,
        max_length=max_length
    )

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False
    )

    return train_loader, val_loader

def create_test_loader(df, tokenizer, batch_size=4, max_length=120):
    """
    Create a DataLoader for test data to be used during evaluation.

    This function wraps the given preprocessed DataFrame into a 
    PropagandaDataset and returns a PyTorch DataLoader that can be used 
    for model evaluation.

    Args:
        df (pd.DataFrame): DataFrame containing the processed text and label indices.
        tokenizer (transformers.PreTrainedTokenizer): Tokenizer to convert texts into token IDs.
        batch_size (int): Number of samples per batch (default is 4).
        max_length (int): Maximum length of tokenized input sequences (default is 120).

    Returns:
        torch.utils.data.DataLoader: A DataLoader for the test dataset.
    
    """
    test_dataset = PropagandaDataset(
        texts=df['processed_text'].tolist(),
        labels=df['label_idx'].tolist(),
        tokenizer=tokenizer,
        max_length=max_length
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False
    )

    return test_loader

def train_model(model, train_loader, val_loader, epochs=4, learning_rate=1e-5, patience=2): 
    """
    Train the model using the specified dataloaders.

    Args:
        model: The model to train
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        epochs: Number of training epochs
        learning_rate: Learning rate for optimizer
        patience: Number of epochs with no improvement after which training will be stopped

    Returns:
        Trained model
    """
    # Set up device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    model = model.to(device)

    # Set up optimizer - Using AdaFactor as specified in the paper
    optimizer = Adafactor(
        model.parameters(),
        lr=learning_rate,
        eps=(1e-30, 1e-3),
        clip_threshold=1.0,
        decay_rate=-0.8,
        beta1=None,
        weight_decay=0.0,
        relative_step=False,
        scale_parameter=False,
        warmup_init=False
    )

    # Total number of training steps
    total_steps = len(train_loader) * epochs

    # Set up learning rate scheduler
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,
        num_training_steps=total_steps
    )

    # For early stopping
    best_val_f1 = 0
    no_improve_epochs = 0

    # Training loop
    for epoch in range(epochs):
        print(f'\nEpoch {epoch + 1}/{epochs}')

        # Training phase
        model.train()
        train_loss = 0

        progress_bar = tqdm(train_loader, desc="Training")
        for batch in progress_bar:
            # Move batch to device
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            # Clear gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )

            loss = outputs.loss
            train_loss += loss.item()

            # Backward pass
            loss.backward()

            # Update parameters
            optimizer.step()

            # Update scheduler
            scheduler.step()

            # Update progress bar
            progress_bar.set_postfix({'loss': loss.item()})

        avg_train_loss = train_loss / len(train_loader)
        print(f'Average training loss: {avg_train_loss:.4f}')

        # Validation phase
        val_metrics = evaluate_model(model, val_loader, device)
        val_f1_micro = val_metrics['f1_micro']

        print(f'Validation Accuracy: {val_metrics["accuracy"]:.4f}')
        print(f'Validation F1 Micro: {val_f1_micro:.4f}')
        print(f'Validation F1 Macro: {val_metrics["f1_macro"]:.4f}')

        # Check for improvement for early stopping
        if val_f1_micro > best_val_f1:
            best_val_f1 = val_f1_micro
            no_improve_epochs = 0

            # Save the model
            torch.save(model.state_dict(), 'best_propaganda_model.pth')
            print(f'Model saved with F1 Micro: {val_f1_micro:.4f}')
        else:
            no_improve_epochs += 1
            print(f'No improvement for {no_improve_epochs} epochs.')

            if no_improve_epochs >= patience:
                print(f'Early stopping after {epoch + 1} epochs.')
                break

    # Load the best model
    model.load_state_dict(torch.load('best_propaganda_model.pth'))

    return model

def evaluate_model(model, data_loader, device):
    """
    Evaluate the model on the specified data.

    Args:
        model: The model to evaluate
        data_loader: DataLoader with evaluation data
        device: Device to run evaluation on

    Returns:
        Dictionary with evaluation metrics
    """
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Evaluating"):
            # Move batch to device
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            # Forward pass
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )

            # Get predictions
            logits = outputs.logits
            predictions = torch.argmax(logits, dim=1)

            # Add to lists
            all_preds.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_preds)
    f1_micro = f1_score(all_labels, all_preds, average='micro')
    f1_macro = f1_score(all_labels, all_preds, average='macro')

    return {
        'accuracy': accuracy,
        'f1_micro': f1_micro,
        'f1_macro': f1_macro,
        'predictions': all_preds,
        'labels': all_labels
    }


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

label_mapping = {label: idx for idx, label in enumerate(propaganda_types)}

#  Load and preprocess data
print("Loading and preprocessing training data...")
train_df = load_and_preprocess_data('/content/propaganda_train.tsv', label_mapping)

#  Load model and tokenizer
print("Loading model and tokenizer...")
try:
    tokenizer = RobertaTokenizer.from_pretrained('roberta-large')
    model = RobertaForSequenceClassification.from_pretrained(
        'roberta-large',
        num_labels=len(propaganda_types)
    )
except Exception as e:
    print(f"Error loading roberta-large: {e}")
    print("Falling back to roberta-base...")
    tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
    model = RobertaForSequenceClassification.from_pretrained(
        'roberta-base',
        num_labels=len(propaganda_types)
    )

#  Split data into training (90%) and validation (10%) sets
print("Splitting data into training and validation sets...")

# Check for NaN values in label_idx
if 'label_idx' in train_df.columns:
    nan_count = train_df['label_idx'].isna().sum()
    if nan_count > 0:
        print(f"Warning: Found {nan_count} NaN values in labels. Dropping these rows.")
        train_df = train_df.dropna(subset=['label_idx'])

    # Also ensure label_idx is an integer type
    train_df['label_idx'] = train_df['label_idx'].astype(int)

    stratify_column = train_df['label_idx']
else:
    stratify_column = None
    print("Warning: No 'label_idx' column found, stratification disabled.")

try:
    train_data, val_data = train_test_split(
        train_df,
        test_size=0.1,  # 10% for validation
        random_state=42,
        stratify=stratify_column
    )
except ValueError as e:
    print(f"Error during stratification: {e}")
    print("Attempting split without stratification...")
    train_data, val_data = train_test_split(
        train_df,
        test_size=0.1,
        random_state=42
    )

print(f"Training data size: {len(train_data)}")
print(f"Validation data size: {len(val_data)}")

# Create data loaders from the splits
print("Creating data loaders...")
train_dataset = PropagandaDataset(
    texts=train_data['processed_text'].tolist(),
    labels=train_data['label_idx'].tolist(),
    tokenizer=tokenizer,
    max_length=120
)

val_dataset = PropagandaDataset(
    texts=val_data['processed_text'].tolist(),
    labels=val_data['label_idx'].tolist(),
    tokenizer=tokenizer,
    max_length=120
)

train_loader = DataLoader(
    train_dataset,
    batch_size=4,
    shuffle=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=4,
    shuffle=False
)

#  Train the model
print("Training model...")
trained_model = train_model(
    model,
    train_loader,
    val_loader,
    epochs=4,
    learning_rate=2e-5, #Changed learning rate to 2e-5(Model B) from 1e-5 (Model A)
    patience=2
)

#Final evaluation on validation set
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print("Performing final evaluation on validation data...")
val_metrics = evaluate_model(trained_model, val_loader, device)

print("\nValidation Results:")
print(f"Accuracy: {val_metrics['accuracy']:.4f}")
print(f"F1 Micro: {val_metrics['f1_micro']:.4f}")
print(f"F1 Macro: {val_metrics['f1_macro']:.4f}")

# Generate detailed classification report
# Get the unique classes that actually appear in the predictions and labels
unique_classes = sorted(list(set(val_metrics['labels']) | set(val_metrics['predictions'])))
class_names = [propaganda_types[i] for i in unique_classes]

report = classification_report(
    val_metrics['labels'],
    val_metrics['predictions'],
    labels=unique_classes,  # Specify which labels to include in the report
    target_names=class_names,  # Provide names only for labels that exist
    digits=4
)

print("\nDetailed Classification Report:")
print(report)

# Also print full classification mapping for reference
print("\nFull Class Mapping:")
for idx, class_name in enumerate(propaganda_types):
    count = (val_data['label_idx'] == idx).sum() if 'label_idx' in val_data.columns else 0
    print(f"  {idx}: {class_name} - {count} examples in validation set")

# Save predictions to file
val_data_copy = val_data.copy()
val_data_copy['predicted_label_idx'] = val_metrics['predictions']
val_data_copy['predicted_label'] = val_data_copy['predicted_label_idx'].apply(
    lambda idx: propaganda_types[idx]
)

val_data_copy[['tagged_in_context', 'label', 'predicted_label']].to_csv(
    'propaganda_validation_predictions.csv', index=False
)

print("Validation predictions saved to 'propaganda_validation_predictions.csv'")


#  Optional: Evaluate on actual test data if available
print("Loading and preprocessing test data...")
test_df = load_and_preprocess_data('/content/propaganda_val.tsv', label_mapping)

print("Creating test data loader...")
test_loader = create_test_loader(
    test_df,
    tokenizer,
    batch_size=4,
    max_length=120
)

print("Evaluating model on test data...")
test_metrics = evaluate_model(trained_model, test_loader, device)

print("\nTest Results:")
print(f"Accuracy: {test_metrics['accuracy']:.4f}")
print(f"F1 Micro: {test_metrics['f1_micro']:.4f}")
print(f"F1 Macro: {test_metrics['f1_macro']:.4f}")

# Generate detailed test classification report
test_report = classification_report(
    test_metrics['labels'],
    test_metrics['predictions'],
    target_names=propaganda_types,
    digits=4
)

print("\nDetailed Test Classification Report:")
print(test_report)

# Save test predictions
test_df['predicted_label_idx'] = test_metrics['predictions']
test_df['predicted_label'] = test_df['predicted_label_idx'].apply(
    lambda idx: propaganda_types[idx]
)

test_df[['tagged_in_context', 'label', 'predicted_label']].to_csv(
    'propaganda_test_predictions.csv', index=False
) #To Check the predictions



