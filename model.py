# XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
# ! This is the file for the model class.
# !     You need to implement the train and predict methods.
# !     You can also add any other methods as required.
# !     You can also add required parameters to the methods.
# !     You can also use additional packages in this file.
# !
# ! Make sure that the final implementation is compatible with the
# !     Model class. Be careful about the input and output types of
# !     the methods.
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
            # Not trained uyarısını konsolu kirletmemesi için sildik
             
    ###########################################################
    # ! Update functions for Model initialization and training
    ###########################################################
    def init_model(self, **kwargs):
        # Eğer parametre gelmezse Şampiyon varsayılanları (0.25 ve None) kullan
        c_value = kwargs.get('C', 0.25) 
        cw = kwargs.get('class_weight', None)
        
        # LinearSVC: Yüksek boyutlu TF-IDF verileri için en optimize çözüm.
        self.model = LinearSVC(
            C=c_value, 
            class_weight=cw, 
            max_iter=10000, 
            dual="auto", 
            random_state=42
        )

    def train(self, x: List[List[float]], y: List[int]):
        self.model.fit(x, y)

    # XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
    # ! Do not change the remaining code
    # XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
    def predict(self, x: List[List[float]]) -> List[int]:
        return self.model.predict(x)

    def evaluate(self, x: List[List[float]], y: List[int]) -> Dict:
        prediction = self.predict(x)

        return {"accuracy": accuracy_score(y, prediction),
                "f1": f1_score(y, prediction, average="weighted"),
                "roc_auc": roc_auc_score(y, prediction, average="weighted")}

    def save(self, path: str):
        with open(path, 'wb') as f:
            pickle.dump(self, f)

    def load(self, path: str):
        with open(path, 'rb') as f:
            return pickle.load(f)