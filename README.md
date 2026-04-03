# CMPE 346 - Natural Language Processing: Assignment 01

**Author:** Özge Meltem İnan
**Student ID:** 123200086

## 📌 Project Overview
This repository contains a highly optimized machine learning pipeline for binary sentiment analysis. Complying with the assignment rules, the entire architecture is built strictly using `scikit-learn`. The model processes raw textual data and classifies it into **Positive (1)** or **Negative (0)** sentiments.

Due to the high-dimensional nature of natural language, a `LinearSVC` model paired with a hybrid `TF-IDF` embedding strategy was implemented to maximize the macro F1-Score.

## 🧠 Architecture & Implementation Details

The pipeline is modularized into three intelligent components: Tokenizer, Embedder, and Classifier.

### 1. Advanced Tokenizer 
Standard punctuation removal destroys valuable sentiment signals. The `Tokenizer` class in `preprocessing.py` preserves contextual cues before cleaning:
* **Regex-Based Custom Tagging:** * Emoticons (`<SMILE>`, `<SAD>`)
  * Capitalized shouting (`<ALLCAPS>`)
  * Elongated words (e.g., "sooo" -> `<ELONGATED>`)
  * Punctuation of interest (`<EXCLAMATION>`, `<QUESTION>`)
* **Scope-Based Negation Handling (`_handle_negation`):** Implements an algorithm to detect negators (e.g., "not", "didn't", "never"). Once triggered, subsequent tokens are prefixed with `NOT_` until a scope-breaker (like `<exclamation>` or `<sad>`) resets the flag. This ensures "not happy" and "happy" are treated as distinct mathematical entities.

### 2. Hybrid Embedder 
To capture both semantic context and morphological robustness, the `Embedder` merges two `TfidfVectorizer` representations using `FeatureUnion`:
* **Word-Level TF-IDF (Weight: 1.0):** Extracts 1 to 3 n-grams (up to 200,000 features). Tuned with `min_df=3` to filter out rare noise.
* **Character-Level TF-IDF (Weight: 0.8):** Extracts 3 to 6 n-grams within word boundaries (`char_wb`, up to 100,000 features). This makes the model highly resilient to typos, internet slang, and unexpected suffixes.
* **Logarithmic Scaling:** `sublinear_tf=True` is applied to both to prevent high-frequency words from disproportionately dominating the sparse matrix.

### 3. Model Selection & Hyperparameter Optimization
The classification is handled by `LinearSVC`, which is mathematically optimal for finding hyperplanes in massive TF-IDF vector spaces. Optimization was done in `main.py` using `GridSearchCV`:
* **Cross-Validation:** 5-fold CV to ensure generalizability.
* **Scoring Metric:** `f1_macro` was explicitly targeted to ensure balanced performance across both classes.
* **Parameter Space Explored:**
  * **Regularization (`C`):** `[0.25, 0.27, 0.28, 0.29, 0.30]` - Focused around low C values to enforce heavier regularization and prevent overfitting on the massive feature set.
  * **Class Weights (`class_weight`):** Tested various imbalanced penalizations for the positive class (e.g., `1.03`, `1.05`, `1.08`) to combat slight dataset skewness.

---

## 📂 Directory Structure
The repository strictly follows the required directory tree:

```text
.
├── data/                  # Contains train.csv, valid.csv, test.csv
├── saved_objects/         # Contains serialized objects (.pkl)
│   ├── tokenizer.pkl
│   ├── embedder.pkl
│   └── model.pkl
├── main.py                # Script for training and hyperparameter tuning
├── model.py               # LinearSVC model wrapper
├── preprocessing.py       # Tokenizer and Embedder implementations
├── run_test.py            # Evaluation script provided by the instructor
└── README.md              # This documentation file