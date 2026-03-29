# XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
# ! This is a sample file for your experiments. You can change
# !     this file as you wish.
# ! Strategy: Exhaustive Grid Search with Regularization for 0.92+ F1
# XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
from preprocessing import Preprocessor, Tokenizer, Embedder
from model import Model

import pandas as pd
import os
from sklearn.svm import LinearSVC
from sklearn.model_selection import GridSearchCV

# =====================================================================
# 1. DATA LOADING & DIRECTORY SETUP
# =====================================================================
print("Loading datasets...")
train_data = pd.read_csv("data/train.csv")
valid_data = pd.read_csv("data/valid.csv")

# Ensure the directory exists to avoid FileNotFoundError
if not os.path.exists("saved_objects"):
    os.makedirs("saved_objects")

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
# 4. EXHAUSTIVE HYPERPARAMETER TUNING
# =====================================================================
print("Starting Exhaustive Grid Search with Regularization focus...")


param_grid = {
    'C': [0.05, 0.08, 0.1, 0.12, 0.15], 
    'class_weight': [
        {0: 1.0, 1: 0.90}, 
        'balanced', 
        {0: 1.0, 1: 1.10}, 
        {0: 1.0, 1: 1.20}
    ]
}

temp_model = LinearSVC(max_iter=10000, dual="auto", random_state=42)

search = GridSearchCV(
    estimator=temp_model, 
    param_grid=param_grid, 
    scoring='f1_weighted', 
    cv=5,                    
    verbose=1, 
    n_jobs=1                 
)

search.fit(trainX, trainY)
best_params = search.best_params_
print(f"Optimal Configuration Found: {best_params}")

# =====================================================================
# 5. FINAL MODEL TRAINING & EVALUATION
# =====================================================================
print("Training the Final Model with Best Parameters...")

model = Model(**best_params)
model.train(trainX, trainY)

# Mandatory save path for assignment evaluation
model.save("saved_objects/model.pkl")

print("\n" + "="*40)
print("FINAL EVALUATION ON VALIDATION SET:")
print("="*40)
print(model.evaluate(validX, validY))

# Overfitting kontrolü için Train ve Valid skorlarını karşılaştır
train_results = model.evaluate(trainX, trainY)
print(f"Train F1: {train_results['f1']}")
print(f"Valid F1: {model.evaluate(validX, validY)['f1']}")