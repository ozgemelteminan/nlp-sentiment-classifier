# XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
# ! This is the file for the model class.
# !     Do not change predict, evaluate, save, load methods.
# XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX

from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.svm import LinearSVC
from typing import List, Dict
import numpy as np
import pickle

class Model:
    def __init__(self, pretrained_path: str = None, **kwargs):
        self.pretrained_path = pretrained_path
        self.__dict__.update(kwargs)
        if pretrained_path != None:
            self.__dict__.update(self.load(self.pretrained_path).__dict__)
        else:
            self.init_model(**kwargs)
            if not kwargs:
                print(f"!! {type(self).__name__} is not trained.")
             
    def init_model(self, **kwargs):
        c_value = kwargs.get('C', 0.1) 
        cw = kwargs.get('class_weight', 'balanced')
        
        # LinearSVC: Yüksek boyutlu TF-IDF verileri için en optimize çözüm.
        self.model = LinearSVC(
            C=c_value, 
            class_weight=cw, 
            max_iter=10000, 
            dual="auto", 
            tol=1e-4, 
            random_state=42
        )

    def train(self, x: List[List[float]], y: List[int]):
        # Model eğitimi
        self.model.fit(x, y)

    # =================================================================
    # ! DO NOT CHANGE THESE METHODS 
    # =================================================================
    def predict(self, x: List[List[float]]) -> List[int]:
        return self.model.predict(x)

    def evaluate(self, x: List[List[float]], y: List[int]) -> Dict:
        prediction = self.predict(x)
        scores = self.model.decision_function(x)
        return {
            "accuracy": accuracy_score(y, prediction),
            "f1": f1_score(y, prediction, average="weighted"),
            "roc_auc": roc_auc_score(y, scores, average="weighted")
        }

    def save(self, path: str):
        with open(path, 'wb') as f:
            pickle.dump(self, f)

    def load(self, path: str):
        with open(path, 'rb') as f:
            return pickle.load(f)