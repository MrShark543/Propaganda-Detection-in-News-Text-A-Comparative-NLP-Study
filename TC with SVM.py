import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, train_test_split, StratifiedKFold
from sklearn.metrics import classification_report, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
import pickle


# Loading data
print("Loading and preprocessing data...")
train_df = pd.read_csv("propaganda_train.tsv", sep="\t")
test_df = pd.read_csv("propaganda_val.tsv", sep="\t")

# Extract text without BOS and EOS tags
train_df["text"] = train_df["tagged_in_context"].str.replace("<BOS>", "").str.replace("<EOS>", "")
test_df["text"] = test_df["tagged_in_context"].str.replace("<BOS>", "").str.replace("<EOS>", "")


# Split training data for development
X_train, X_dev, y_train, y_dev = train_test_split(
    train_df["text"], train_df["label"],
    test_size=0.20,
    stratify=train_df["label"],
    random_state=42
)

# Define the pipeline with TF-IDF and LinearSVC
pipe = Pipeline([
    ("tfidf", TfidfVectorizer(
        ngram_range=(1, 2),
        max_features=10000,
        min_df=3,  # Minimum document frequency
        token_pattern=r"(?u)\b\w+\b",
        use_idf=True,
        sublinear_tf=True  # Applying sublinear tf scaling (1 + log(tf))
    )),
    ("svm", LinearSVC(
        class_weight="balanced",
        dual=False,  # For efficiency
        max_iter=5000,
        random_state=42
    )),
])

# Set up k-fold cross-validation
print("\nPerforming 5-fold cross-validation...")
param_grid = {
    "tfidf__ngram_range": [(1, 1), (1, 2), (1, 3)],
    "tfidf__max_features": [5000, 10000, 15000],
    "svm__C": [0.01, 0.1, 1.0, 10.0]
}

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
search = GridSearchCV(
    pipe,
    param_grid,
    cv=skf,
    scoring="f1_macro",  # Using macro-averaged F1 score
    n_jobs=-1,  # Use all available cores
    verbose=1
)

# Train the model with grid search
search.fit(X_train, y_train)
print(f"Best parameters: {search.best_params_}")
print(f"Best cross-validation score: {search.best_score_:.4f}")

# Evaluate on development set
print("\nEvaluating on development set:")
y_dev_pred = search.predict(X_dev)
dev_report = classification_report(y_dev, y_dev_pred, digits=4)
print(dev_report)

# Training on full training data and evaluate on test set
print("\nTraining final model on all training data...")
best_pipe = search.best_estimator_
best_pipe.fit(train_df["text"], train_df["label"])

print("Evaluating on test set:")
y_test = test_df["label"]
y_test_pred = best_pipe.predict(test_df["text"])
test_report = classification_report(y_test, y_test_pred, digits=4)
print(test_report)

# Visualize confusion matrix on test set
def plot_confusion_matrix(y_true, y_pred, labels):
    """
    Plots and saves a confusion matrix as a heatmap image.

    This function compares the true labels with the predicted labels,
    visualizes the confusion matrix using a heatmap, and saves it as 'confusion_matrix.png'.

    Args:
        y_true (List or Array): True labels from the dataset.
        y_pred (List or Array): Predicted labels from the model.
        labels (List[str]): List of all unique label names for axis ticks.
    """
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    plt.figure(figsize=(14, 12))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png')
    print("Saved confusion matrix visualization to 'confusion_matrix.png'")

unique_labels = sorted(train_df["label"].unique())
plot_confusion_matrix(y_test, y_test_pred, unique_labels)

# Feature importance analysis
def get_top_features(vectorizer, model, class_names, top_n=10):
    """
    Extracts the top N most important features (words or n-grams) for each class from a trained SVM model.

    Args:
        vectorizer: The fitted TfidfVectorizer used in the pipeline.
        model: The trained LinearSVC model.
        class_names (List[str]): List of class labels.
        top_n (int): Number of top features to extract per class.

    Returns:
        dict: A dictionary where each key is a class name and the value is a list of tuples 
              (feature, importance score), sorted by importance.
    """
    feature_names = vectorizer.get_feature_names_out()
    feature_importance = {}

    for i, class_name in enumerate(class_names):
        # For each class, get the coefficients from the model
        coefficients = model.coef_[i]
        top_indices = coefficients.argsort()[-top_n:][::-1]
        top_features = [(feature_names[j], coefficients[j]) for j in top_indices]
        feature_importance[class_name] = top_features

    return feature_importance

# Extract components from the pipeline
tfidf = best_pipe.named_steps['tfidf']
svm = best_pipe.named_steps['svm']
class_names = unique_labels  

feature_importance = get_top_features(tfidf, svm, class_names)

# Print top features for each class
print("\nTop features for each propaganda technique:")
for class_name, features in feature_importance.items():
    print(f"\n{class_name}:")
    for feature, importance in features:
        print(f"  {feature}: {importance:.4f}")

# Save results and model


# Save test set predictions
result_df = pd.DataFrame({
    "text": test_df["text"],
    "true_label": y_test,
    "predicted_label": y_test_pred
})
result_df.to_csv("propaganda_test_predictions.csv", index=False)
print("\nSaved test predictions to 'propaganda_test_predictions.csv'")

# Save the model
with open('propaganda_classifier.pkl', 'wb') as f:
    pickle.dump(best_pipe, f)
print("Saved model to 'propaganda_classifier.pkl'")

# Report overall F1 score
macro_f1 = f1_score(y_test, y_test_pred, average='macro')
micro_f1 = f1_score(y_test, y_test_pred, average='micro')
print(f"\nFinal test set performance:")
print(f"Macro-averaged F1 score: {macro_f1:.4f}")
print(f"Micro-averaged F1 score: {micro_f1:.4f}")