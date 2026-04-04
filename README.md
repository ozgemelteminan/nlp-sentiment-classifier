# CMPE 346 - Natural Language Processing: Assignment 01

**Author:** Özge Meltem İnan
**Student ID:** 123200086

## 📌 Project Overview
This repository contains a highly optimized machine learning pipeline for binary sentiment analysis. Complying with the assignment rules, the entire architecture is built strictly using `scikit-learn`. The model processes raw textual data and classifies it into **Positive (1)** or **Negative (0)** sentiments.

Due to the high-dimensional nature of natural language, a `LinearSVC` model paired with a hybrid `TF-IDF` embedding strategy was implemented to maximize the macro F1-Score.

## 🔬 Experimental History & Architectural Decisions

Reaching the `0.917` F1-score was an intensive, iterative process. Before finalizing the architecture, the model went through several milestones and optimization barriers. The table below summarizes the score progression:

| Stage / Experiment Version | Train F1 Score | Valid F1 Score | Notes |
| :--- | :--- | :--- | :--- |
| **Course Baseline** | - | **0.8475** | The minimum target required for the assignment. |
| **SVD (Dimensionality Reduction)** | - | **0.8579** | Degraded performance due to the loss of fine-grained signals (e.g., char n-grams). |
| **Optimization V1** | - | **0.9152** | Solid initial score after establishing the TF-IDF and FeatureUnion architecture. |
| **Optimization V2** | - | **0.9156** | Slight improvement by filtering noise via `min_df` and `max_df` parameter tuning. |
| **Optimization V3 (Previous Peak)**| - | **0.9166** | A resistance point reached by tweaking the `C` parameter and `class_weight`. |
| **🏆 Champion Model (Final)** | **0.9895** | **0.9170** | The absolute maximum limit reached by `LinearSVC`, breaking the 0.9166 barrier. |

To achieve this progression, I conducted **10 major experiments** spanning preprocessing, feature extraction, and algorithm selection:

### Phase 1: Text Preprocessing & Normalization
* **Experiment 1: Standard Punctuation Removal vs. Custom Tagging**
    * *Approach:* Initially stripped all punctuation. Later, I replaced emoticons, repeated letters, and specific punctuation with regex tags (`<SMILE>`, `<ALLCAPS>`, `<ELONGATED>`).
    * *Result:* Preserving these emotional cues provided a significant performance boost.
* **Experiment 2: Scope-Based Negation Handling**
    * *Approach:* Implemented a custom `_handle_negation` function to prefix words with `NOT_` if they follow a negator (e.g., "didn't like" -> "didn't NOT_like") until a scope-breaker is reached.
    * *Result:* Dramatically reduced False Positives in reviews that contained sarcasm or mixed sentiments.
* **Experiment 3: Classical Stop-Word Removal**
    * *Approach:* Added `stop_words='english'` to reduce matrix sparsity.
    * *Result:* F1-Score dropped. Standard stop-word lists aggressively remove sentiment modifiers like *"not"*, *"very"*, and *"but"*. 
    * *Conclusion:* Abandoned predefined stop-words. Instead, I used mathematical frequency filtering (`max_df=0.90`) to dynamically eliminate uninformative corpus-specific noise.

### Phase 2: Feature Engineering & Embeddings
* **Experiment 4: Pure Unigrams (Baseline)**
    * *Approach:* `TfidfVectorizer` with `ngram_range=(1, 1)`.
    * *Result:* Failed to capture local context. 
* **Experiment 5: Minimum Document Frequency (`min_df`) Tuning**
    * *Approach:* Tuned `min_df` from 2 to 3 for word n-grams, and from 3 to 4 for character n-grams.
    * *Result:* Successfully filtered out extreme noise and misspellings that appeared only once or twice, preventing the model from overfitting to garbage tokens.
* **Experiment 6: FeatureUnion (Character + Word N-grams)**
    * *Approach:* Combined Word N-grams (1, 3) with Character N-grams (`char_wb`, 3, 6) using `FeatureUnion` with weighted importance.
    * *Result:* F1-Score leaped significantly. Character n-grams made the model highly resilient to typos and internet slang.

### Phase 3: Algorithm Selection & Hyperparameters
* **Experiment 7: Exploring Different Classifiers**
    * *Approach:* Tested `MultinomialNB`, `LogisticRegression`, and `LinearSVC` on the same TF-IDF matrix.
    * *Result:* `LinearSVC` dominated because it is mathematically superior at finding separating hyperplanes in extremely high-dimensional, sparse spaces.
* **Experiment 8: Neural Networks (`MLPClassifier`)**
    * *Approach:* Tested `sklearn.neural_network.MLPClassifier` to capture non-linear relationships.
    * *Result:* Overfitted the training data and took too long to converge. Linear models proved far more stable.
* **Experiment 9: Dimensionality Reduction (TruncatedSVD)**
    * *Approach:* Pipelined `TruncatedSVD(n_components=500)` before `LinearSVC` to compress features into a dense matrix.
    * *Result:* F1-Score degraded to **`0.8579`**. Compression destroyed the fine-grained, rare signals (like custom tags and negated words) that the linear model relied on.
* **Experiment 10: Grid Search Fine-Tuning**
    * *Approach:* Exhaustive `GridSearchCV` on `LinearSVC` covering Regularization (`C`: 0.25 to 0.30) and `class_weight` adjustments.
    * *Result:* Found the optimal balance to combat slight dataset skewness without penalizing the majority class too heavily.

---

## 🧠 Final Architecture Implementation

The pipeline is modularized into three intelligent components: Tokenizer, Embedder, and Classifier. Deep learning frameworks were strictly avoided to comply with the assignment rules.

### 1. Advanced Tokenizer
* **Custom Regex Tagging:** Maps emoticons, capitalized shouting, elongated words, and key punctuation to explicit tokens.
* **Scope-Based Negation Handling:** Implements an algorithm to detect negators and prefixes subsequent tokens with `NOT_` until a scope-breaker resets the flag. 

### 2. Hybrid Embedder
The `Embedder` merges two representations using `FeatureUnion`:
* **Word-Level TF-IDF (Weight: 1.0):** Extracts 1 to 3 n-grams.
* **Character-Level TF-IDF (Weight: 0.8):** Extracts 3 to 6 n-grams within word boundaries (`char_wb`).

### 3. Model & Hyperparameters
Classification is handled by `LinearSVC`. Optimization was done using `GridSearchCV`:
* **Cross-Validation:** 5-fold CV.
* **Scoring Metric:** `f1_macro` was explicitly targeted to ensure balanced performance.

## 📂 Directory Structure
```text

├── data/                  # Contains train.csv, valid.csv, test.csv
├── saved_objects/         # Contains serialized objects (.pkl)
│   ├── tokenizer.pkl
│   ├── embedder.pkl
│   └── model.pkl
├── main.py                # Script for model training and Grid Search
├── model.py               # LinearSVC model wrapper
├── preprocessing.py       # Custom Tokenizer and Embedder logic
├── run_test.py            # Evaluation script provided in assignment
└── README.md              # Documentation