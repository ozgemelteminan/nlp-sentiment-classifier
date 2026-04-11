# CMPE 346 - Natural Language Processing: Assignment 01

**Author:** Özge Meltem İnan  
**Student ID:** 123200086

## 📂 Directory Structure

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

## 📌 Project Overview
This repository contains a highly optimized machine learning pipeline for binary sentiment analysis. Complying with the assignment rules, the entire architecture is built strictly using `scikit-learn`. The model processes raw textual data and classifies it into **Positive (1)** or **Negative (0)** sentiments.

Due to the high-dimensional nature of natural language, a `LinearSVC` model paired with a hybrid `TF-IDF` embedding strategy was implemented to maximize the macro F1-Score.

## 🔬 Experimental History & Architectural Decisions

Reaching the `0.9200` F1-score was an intensive, iterative process. Before finalizing the architecture, the model went through several milestones and optimization barriers. The table below summarizes the score progression:

| Stage / Experiment Version | Train F1 Score | Valid F1 Score | Notes |
| :--- | :--- | :--- | :--- |
| **Course Baseline** | - | **0.8475** | The minimum target required for the assignment. |
| **SVD (Dimensionality Reduction)** | - | **0.8579** | Degraded performance due to the loss of fine-grained signals (e.g., char n-grams). |
| **Optimization V1** | - | **0.9152** | Solid initial score after establishing the TF-IDF and FeatureUnion architecture. |
| **Optimization V2** | - | **0.9156** | Slight improvement by filtering noise via `min_df` and `max_df` parameter tuning. |
| **Optimization V3 (Previous Peak)**| - | **0.9170** | A resistance point reached by tweaking the `C` parameter and negative sentiment tagging. |
| **🏆 Champion Model (Final)** | - | **0.9208** | The absolute maximum limit reached by introducing `<pos_signal>`, rating extractors, and optimizing `LinearSVC` with balanced weights. |

To achieve this progression, I conducted **major experiments** spanning preprocessing, feature extraction, and algorithm selection:

### Phase 1: Text Preprocessing & Normalization
* **Experiment 1: Advanced Regex Tagging (Positive & Negative Signals)**
    * *Approach:* Instead of merely stripping punctuation, I replaced emoticons, specific punctuation, and strong sentiment phrases with custom regex tags (`<SMILE>`, `<ALLCAPS>`, `<ELONGATED>`). Furthermore, strong subjective phrases ("waste of time", "masterpiece", "highly recommend") were explicitly grouped under `<neg_signal>` and `<pos_signal>`. Explicit numerical ratings (e.g., "10/10", "5 stars") were unified under a `<rating>` tag.
    * *Result:* Preserving these emotional cues and unifying multi-word sentiment bombs provided a massive performance boost and helped break the 0.92 barrier.
* **Experiment 2: Scope-Based Negation Handling**
    * *Approach:* Implemented a custom `_handle_negation` function to prefix words with `NOT_` if they follow a negator (e.g., "didn't like" -> "didn't NOT_like") until a scope-breaker is reached.
    * *Result:* Dramatically reduced False Positives in reviews that contained sarcasm or mixed sentiments.
* **Experiment 3: Classical Stop-Word Removal**
    * *Approach:* Added `stop_words='english'` to reduce matrix sparsity.
    * *Result:* F1-Score dropped. Standard stop-word lists aggressively remove sentiment modifiers like *"not"*, *"very"*, and *"but"*. 
    * *Conclusion:* Abandoned predefined stop-words. Instead, I used mathematical frequency filtering (`max_df=0.85` to `0.90`) to dynamically eliminate uninformative corpus-specific noise.

### Phase 2: Feature Engineering & Embeddings
* **Experiment 4: Pure Unigrams (Baseline)**
    * *Approach:* `TfidfVectorizer` with `ngram_range=(1, 1)`.
    * *Result:* Failed to capture local context. 
* **Experiment 5: Minimum Document Frequency (`min_df`) Tuning**
    * *Approach:* Tuned `min_df` to 4 for word n-grams, and 5 for character n-grams.
    * *Result:* Successfully filtered out extreme noise and misspellings that appeared only rarely, preventing the model from overfitting to garbage tokens.
* **Experiment 6: FeatureUnion (Character + Word N-grams)**
    * *Approach:* Combined Word N-grams (1, 3) with Character N-grams (`char_wb`, 3, 5) using `FeatureUnion` with explicitly weighted importance. Word TF-IDF was given full weight (1.0), while Character TF-IDF was scaled down (0.5) to act as a supportive signal for typos and internet slang without overwhelming the primary semantic meaning.

### Phase 3: Algorithm Selection & Hyperparameters
* **Experiment 7: Exploring Different Classifiers**
    * *Approach:* Tested `MultinomialNB`, `LogisticRegression`, and `LinearSVC` on the same TF-IDF matrix.
    * *Result:* `LinearSVC` dominated because it is mathematically superior at finding separating hyperplanes in extremely high-dimensional, sparse spaces (totaling ~180,000 features).
* **Experiment 8: Neural Networks (`MLPClassifier`)**
    * *Approach:* Tested `sklearn.neural_network.MLPClassifier` to capture non-linear relationships.
    * *Result:* Overfitted the training data and took too long to converge. Linear models proved far more stable.
* **Experiment 9: Grid Search Fine-Tuning**
    * *Approach:* Exhaustive `GridSearchCV` on `LinearSVC` covering Regularization (`C`: 0.30 to 0.48) and `class_weight` adjustments.
    * *Result:* Found the optimal balance (`C` around 0.30 to 0.45 and `class_weight='balanced'`) to combat slight dataset skewness and push the F1-Score to the 0.9200 mark.

---

## 🧠 Final Architecture Implementation

The pipeline is modularized into three intelligent components: Tokenizer, Embedder, and Classifier. Deep learning frameworks were strictly avoided to comply with the assignment rules.

### 1. Advanced Tokenizer
* **Custom Regex Tagging:** Maps emoticons, capitalized shouting, ratings, and explicit positive/negative signals to unified tokens.
* **Scope-Based Negation Handling:** Implements an algorithm to detect negators and prefixes subsequent tokens with `NOT_` until a scope-breaker resets the flag. 

### 2. Hybrid Embedder
The `Embedder` merges two representations using `FeatureUnion`:
* **Word-Level TF-IDF (Weight: 1.0):** Extracts 1 to 3 n-grams (max 100,000 features).
* **Character-Level TF-IDF (Weight: 0.5):** Extracts 3 to 5 n-grams within word boundaries (`char_wb`) (max 80,000 features).

### 3. Model & Hyperparameters
Classification is handled by `LinearSVC`. Optimization was done using `GridSearchCV`:
* **Cross-Validation:** 5-fold CV.
* **Scoring Metric:** `f1_weighted` was explicitly targeted to ensure balanced performance. 