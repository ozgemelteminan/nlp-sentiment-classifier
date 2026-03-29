# XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
# ! Do not change this file. This is a sample file that
# !    will be run during evaluation of your submission. It
# !    is shared with you so that you can test your implementation
# !    locally. If you get any errors when running this file with
# !    `python run_test.py --valid`, it is highly likely that you
# !    will receive errors during evaluation.
# !
# ! Note that recieving errors will result in a zero score for this
# !    assignment. Therefore, make sure that your implementation is
# !    correct by testing this file locally.
# XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX

from preprocessing import Tokenizer, Embedder, Preprocessor
from model import Model
import pandas as pd
import argparse
import os

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Running experiments for the pretrained model")
    parser.add_argument("--valid", action="store_true", help="Evaluate validation data")
    parser.add_argument("--test",  action="store_true", help="Evaluate test data")
    args = parser.parse_args()

    if args.test and not os.path.exists("data/test.csv"):
        raise FileNotFoundError("Test data not found. Try it with --valid flag")

    if not os.path.exists("saved_objects/tokenizer.pkl"):
        raise FileNotFoundError("Either tokenizer is not found or named wrong. Save your tokenizer object first and name it tokenizer.pkl")

    if not os.path.exists("saved_objects/embedder.pkl"):
        raise FileNotFoundError("Either embedder is not found or named wrong. Save your embedder object first and name it embedder.pkl")

    if not os.path.exists("saved_objects/model.pkl"):
        raise FileNotFoundError("Either model is not found or named wrong. Save your model object first and name it model.pkl")

    tokenizer = Tokenizer(pretrained_path="saved_objects/tokenizer.pkl")
    embedder = Embedder(pretrained_path="saved_objects/embedder.pkl")

    preprocessor = Preprocessor(tokenizer=tokenizer, embedder=embedder)

    if args.valid and not args.test:
        data = pd.read_csv("data/valid.csv")
    elif args.test:
        data = pd.read_csv("data/test.csv")
    else:
        raise ValueError("Please provide either --valid or --test flag")

    X, y = preprocessor.prepare_data(data)

    model = Model(pretrained_path="saved_objects/model.pkl")
    print("Evaluation result: ", model.evaluate(X, y))
