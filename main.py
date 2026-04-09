from preprocessing import Preprocessor, Tokenizer, Embedder
from model import Model
import pandas as pd
import os
from sklearn.svm import LinearSVC
from sklearn.model_selection import GridSearchCV

def main():
    print("Loading datasets...")
    train_data = pd.read_csv("data/train.csv")
    valid_data = pd.read_csv("data/valid.csv")

    if not os.path.exists("saved_objects"):
        os.makedirs("saved_objects")

    print("Initializing Preprocessors...")
    tokenizer = Tokenizer()
    embedder = Embedder()

    train_texts = train_data["text"].tolist()
    train_tokens = [tokenizer.tokenize(text) for text in train_texts]

    print("Training Embedder...")
    embedder.train(train_tokens)
    tokenizer.save("saved_objects/tokenizer.pkl")
    embedder.save("saved_objects/embedder.pkl")

    preprocessor = Preprocessor(tokenizer=tokenizer, embedder=embedder)
    trainX, trainY = preprocessor.prepare_data(train_data)
    validX, validY = preprocessor.prepare_data(valid_data)

    print("Starting Grid Search...")
    # C'yi 0.25 civarında çok daha hassas tarayacağız
    param_grid = {
        'C': [0.15, 0.20, 0.25, 0.30, 0.35],
        'class_weight': [
            None,                
            {0: 1.0, 1: 1.03},
            {0: 1.0, 1: 1.05},    
            {0: 1.0, 1: 1.08}
        ]
    }

    temp_model = LinearSVC(max_iter=10000, dual="auto", random_state=42)

    search = GridSearchCV(
        estimator=temp_model, 
        param_grid=param_grid, 
        scoring='f1_weighted',      
        cv=5,                    
        verbose=1, 
        n_jobs=1  # Kırmızı Mac hatasını önlemek için 1'e geri aldık
    )

    search.fit(trainX, trainY)
    best_params = search.best_params_
    print(f"Best Params: {best_params}")

    print("Training Final Model...")
    model = Model(**best_params)
    model.train(trainX, trainY)
    model.save("saved_objects/model.pkl")

    print("\nFINAL RESULTS:")
    print(model.evaluate(validX, validY))
    print(f"Train F1: {model.evaluate(trainX, trainY)['f1']}")
    print(f"Valid F1: {model.evaluate(validX, validY)['f1']}")

if __name__ == '__main__':
    main()