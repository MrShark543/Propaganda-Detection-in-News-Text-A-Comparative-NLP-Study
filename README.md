# Propaganda Detection in News Text: A Comparative NLP Study

> Detecting and classifying propaganda techniques in news snippets using fine-tuned transformer models and classical ML — covering both technique classification and span identification.

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org)
[![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers-yellow.svg)](https://huggingface.co)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## Overview

Propaganda in digital news is subtle, targeted, and increasingly hard to distinguish from legitimate journalism. This project tackles automated propaganda detection across two tasks:

- **Task 1 — Technique Classification:** Given a sentence with a marked propaganda span, identify which of 8 propaganda techniques is being used (or classify it as non-propaganda).
- **Task 2 — Span Identification + Classification:** Given a raw sentence, locate the propaganda span and classify its technique.

For the full technical report including methodology details and error analysis, see [`Anlp_report_final.pdf`](Anlp_report_final.pdf).

Three approaches are implemented for Task 1 and two for Task 2, enabling a direct comparison of transformer-based models against classical ML baselines.

---

## Propaganda Techniques

The project targets 8 propaganda techniques drawn from the SemEval-2020 Task 11 dataset:

| Technique | Description |
|---|---|
| Flag Waving | Exploiting patriotism or group identity |
| Appeal to Fear/Prejudice | Triggering fear to manipulate opinion |
| Causal Oversimplification | Attributing complex events to a single cause |
| Doubt | Questioning credibility without evidence |
| Exaggeration / Minimisation | Distorting the scale of events |
| Loaded Language | Emotionally charged words to influence |
| Name Calling / Labeling | Attaching negative labels to a person/group |
| Repetition | Reinforcing a message through repeated exposure |

A `not_propaganda` class is also included for non-propaganda spans.

---

## Project Structure

```
.
├── TC_with_roberta.py                      # Task 1: Technique classification with RoBERTa-large
├── TC_with_T5.py                           # Task 1: Technique classification with T5-base
├── TC_with_SVM.py                          # Task 1: Technique classification with SVM + TF-IDF
├── SC_with_roberta.py                      # Task 2: Span identification with RoBERTa BIO tagging
├── SC_with_roberta_and_negative_sampling.py # Task 2: Span identification with negative sampling
└── Running_test_on_identifiedSpans.py      # Inference: classify techniques on predicted spans
```

---

## Approaches

### Task 1: Technique Classification

#### RoBERTa-large (Best performer)
Fine-tuned `roberta-large` (355M parameters) on propaganda spans marked with `<BOS>` and `<EOS>` tags. Includes a preprocessing pipeline that expands contractions and abbreviations, converts to lowercase, and removes noise. The `[CLS]` token's final hidden state is used for classification.

**Key hyperparameters:** Adafactor optimizer, learning rate 2e-5, 4 epochs, max sequence length 120, cross-entropy loss.

#### T5-base
Reformulates classification as a text generation task — the model generates the technique name as free text given a structured prompt listing all possible options. Weighted random sampling is used to address class imbalance.

**Key hyperparameters:** AdamW optimizer, learning rate 3e-5, 3 epochs, max input 150 tokens, max output 25 tokens.

#### SVM + TF-IDF (Baseline)
A classical pipeline: TF-IDF vectorization (n-grams 1–2, 10k features, sublinear TF scaling) feeding into a Linear SVM with balanced class weights. Grid search over n-gram range, feature count, and regularization strength C via 5-fold cross-validation.

---

### Task 2: Span Identification

#### RoBERTa with BIO Tagging
A token-level sequence labeling approach using `roberta-base`. Each token is assigned a BIO label: `B-PROP` (start of span), `I-PROP` (continuation), or `O` (outside). A post-processing pipeline merges fragmented predictions, refines span boundaries, and filters noise. The best predicted span is then passed to the Task 1 RoBERTa classifier for technique labeling.

**Key hyperparameters:** AdamW, learning rate 2e-5, 3 epochs, max sequence length 256, 80/20 train/val split.

#### RoBERTa with Negative Sampling
Frames span detection as binary classification over candidate spans. For each positive (gold propaganda span), 5 negative spans are sampled from the same sentence. At inference time, all possible spans up to 10 words are scored, the `not_propaganda` class is masked out, and the span with the highest propaganda probability is selected.

**Key hyperparameters:** AdamW with linear scheduler, learning rate 2e-5, weight decay 0.01, 3 epochs, max sequence length 128.

---

## Results

### Task 1: Technique Classification

| Model | Macro F1 | Micro F1 | Accuracy |
|---|---|---|---|
| **RoBERTa-large** | **0.6694** | **0.7812** | **0.7812** |
| SVM + TF-IDF | 0.2966 | 0.5172 | — |
| T5-base | 0.0151 | 0.0724 | 0.07 |

#### RoBERTa-large Per-Class F1 (Best Model)

| Technique | F1 |
|---|---|
| Not Propaganda | 0.9116 |
| Flag Waving | 0.7955 |
| Causal Oversimplification | 0.7059 |
| Doubt | 0.6582 |
| Appeal to Fear/Prejudice | 0.6316 |
| Loaded Language | 0.5556 |
| Name Calling / Labeling | 0.6349 |
| Repetition | 0.5135 |
| Exaggeration / Minimisation | 0.6176 |

### Task 2: Span Identification + Classification

| Approach | Span Precision | Span Recall | Span F1 | Classification Accuracy |
|---|---|---|---|---|
| **RoBERTa BIO Tagging** | **0.9914** | **0.9914** | **0.9914** | **0.7190** |
| RoBERTa Negative Sampling | 1.0000 | 0.2100 | 0.3400 | 0.4672 |

---

## Key Findings

**RoBERTa dominates across the board.** Its transformer architecture captures the contextual nuances that TF-IDF simply cannot — most evident in `loaded language` (F1=0.0 for SVM vs. 0.5556 for RoBERTa) and `repetition`, which requires broader document-level context.

**T5 collapsed to a single-class predictor.** Despite weighted sampling, the model converged to predicting `appeal to fear/prejudice` for almost everything (recall=0.98, precision=0.07). Reformulating multi-class classification as text generation is ill-suited for this task — the model exploits statistical patterns in the prompt rather than learning discriminative features.

**BIO tagging vs. negative sampling is a precision-recall tradeoff.** BIO tagging produces comprehensive, high-recall span predictions (F1=0.99). Negative sampling achieves perfect precision (1.00) but dramatically low recall (0.21) — it's highly conservative and misses most spans, constrained by its 10-word span limit and the difficulty of learning from exhaustive candidate enumeration.

**Class imbalance is the dominant challenge.** The `not_propaganda` class (301 of 580 test instances) systematically inflates metrics. Techniques like `repetition` suffer most, partly because detecting repetition inherently requires document-level context beyond individual spans.

---

## Setup

### Requirements

```bash
pip install torch transformers datasets scikit-learn pandas numpy tqdm contractions matplotlib seaborn
```

### Data Format

The scripts expect `.tsv` files with two columns:

```
tagged_in_context                                    label
"The <BOS>enemy is everywhere<EOS> and must..."      appeal_to_fear_prejudice
```

Place your files as `propaganda_train.tsv` and `propaganda_val.tsv` in the working directory (or update the paths in each script).

---

## Usage

### Task 1: Technique Classification

```bash
# RoBERTa-large (recommended)
python TC_with_roberta.py

# T5-base
python TC_with_T5.py

# SVM + TF-IDF baseline
python TC_with_SVM.py
```

### Task 2: Span Identification

```bash
# BIO tagging (recommended)
python SC_with_roberta.py

# Negative sampling approach
python SC_with_roberta_and_negative_sampling.py
```

### End-to-End Inference (Span → Technique)

Once you have predicted spans from Task 2, run the combined inference script to classify their techniques:

```bash
python Running_test_on_identifiedSpans.py
```

Update `MODEL_PATH`, `FILE_PATH`, `SPAN_COLUMN`, and `LABEL_COLUMN` at the bottom of the script to point to your saved model and results file.

---

## References

- Martino et al. (2020). [SemEval-2020 Task 11: Detection of Propaganda Techniques in News Articles](https://arxiv.org/abs/2009.02696)
- Liu et al. (2019). [RoBERTa: A Robustly Optimized BERT Pretraining Approach](https://arxiv.org/abs/1907.11692)
- Raffel et al. (2020). [Exploring the Limits of Transfer Learning with T5](https://arxiv.org/abs/1910.10683)
- Abdullah et al. (2022). Detecting Propaganda Techniques in English News Articles Using Pre-trained Transformers. ICICS 2022.
- Chernyavskiy et al. (2020). [Aschern at SemEval-2020 Task 11: RoBERTa, CRF, and Transfer Learning](https://arxiv.org/abs/2008.02837)
- Ahmad et al. (2025). Hierarchical Graph-based Integration Network for Propaganda Detection. Scientific Reports.
