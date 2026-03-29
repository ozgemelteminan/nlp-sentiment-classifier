# XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
# ! This is a sample file for your experiments. You can change
# !     this file as you wish.
# ! Strategy: Optimizing both C and class_weight to maximize F1.
# XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
from preprocessing import Preprocessor, Tokenizer, Embedder
from model import Model

import pandas as pd
from sklearn.svm import LinearSVC
from sklearn.model_selection import RandomizedSearchCV

# =====================================================================
# 1. DATA LOADING
# =====================================================================
print("Loading datasets...")
train_data = pd.read_csv("data/train.csv")
valid_data = pd.read_csv("data/valid.csv")

# =====================================================================
# 2. INITIALIZING & TRAINING PREPROCESSING OBJECTS
# =====================================================================
print("Initializing Tokenizer and Embedder...")
tokenizer = Tokenizer()
embedder = Embedder()

# Extraction of texts for the Embedder training phase
train_texts = train_data["text"].tolist()
train_tokens = [tokenizer.tokenize(text) for text in train_texts]

print("Training the Embedder (TF-IDF FeatureUnion)...")
# Note: Embedder trains on a list of tokens but joins them internally for TF-IDF
embedder.train(train_tokens)

# Saving preprocessors to the mandatory saved_objects/ directory
print("Saving Tokenizer and Embedder...")
tokenizer.save("saved_objects/tokenizer.pkl")
embedder.save("saved_objects/embedder.pkl")

# =====================================================================
# 3. PREPARING VECTORS (FEATURE EXTRACTION)
# =====================================================================
print("Transforming texts into high-dimensional sparse matrices...")
preprocessor = Preprocessor(tokenizer=tokenizer, embedder=embedder)

# Prepare training and validation data (X: sparse matrices, Y: integer labels)
trainX, trainY = preprocessor.prepare_data(train_data)
validX, validY = preprocessor.prepare_data(valid_data)

# =====================================================================
# 4. GRANULAR HYPERPARAMETER TUNING
# =====================================================================
print("Starting Precision Tuning using RandomizedSearchCV...")

# when ROC AUC is already high. It shifts the decision threshold.
param_distributions = {
    'C': [0.08, 0.1, 0.12, 0.14, 0.16, 0.18, 0.2], 
    'class_weight': [
        {0: 1, 1: 1.12}, 
        {0: 1, 1: 1.15}, 
        {0: 1, 1: 1.18}, 
        {0: 1, 1: 1.20},
        {0: 1, 1: 1.22}
    ]
}

# Initializing a temporary LinearSVC for the search process
temp_model = LinearSVC(max_iter=10000, dual="auto", random_state=42)

search = RandomizedSearchCV(
    estimator=temp_model, 
    param_distributions=param_distributions, 
    n_iter=35,               # Searching 15 combinations for a better "sweet spot"
    scoring='f1_weighted', 
    cv=5,                    # 5-fold cross-validation for maximum robustness
    verbose=1, 
    n_jobs=1                 # Safest for Mac/Python 3.13 environments
)

search.fit(trainX, trainY)
best_params = search.best_params_
print(f"Optimal Configuration Found: {best_params}")

# =====================================================================
# 5. FINAL MODEL TRAINING & EVALUATION
# =====================================================================
print("Training the Final Model with Best Parameters...")

# We pass all best_params (C and class_weight) to our Model class
model = Model(**best_params)
model.train(trainX, trainY)

# Mandatory save path for assignment evaluation
model.save("saved_objects/model.pkl")

print("\n" + "="*40)
print("FINAL EVALUATION ON VALIDATION SET:")
print("="*40)
# Final check to see the F1 improvement
print(model.evaluate(validX, validY))

# Eğitim seti üzerindeki performansı ölç
train_results = model.evaluate(trainX, trainY)
print(f"Train F1: {train_results['f1']}")

# Zaten bastırdığın valid skorunu hatırla
print(f"Valid F1: {model.evaluate(validX, validY)['f1']}")