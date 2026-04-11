# XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
# ! This is a sample file for your experiments. You can change
# !     this file as you wish.
# XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
from preprocessing import Preprocessor, Tokenizer, Embedder
from model import Model
import pandas as pd
import os

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

    print("Training Final Model...")
    model = Model(C=0.5, class_weight=None, loss='squared_hinge')
    model.train(trainX, trainY)
    model.save("saved_objects/model.pkl")

    print("\nFINAL RESULTS:")
    print(model.evaluate(validX, validY))
    print(f"Train F1: {model.evaluate(trainX, trainY)['f1']}")
    print(f"Valid F1: {model.evaluate(validX, validY)['f1']}")

if __name__ == '__main__':
    main()