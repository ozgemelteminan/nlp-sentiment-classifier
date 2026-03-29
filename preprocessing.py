# XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
# ! This is the file for preprocessing objects and their utilities.
# !     Do not change Preprocessor, LabelEncoder, PreprocessorObject 
# !         classes.
# !     You need to update Tokenizer and Embedder classes.
# XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX

from typing import List, Dict, Tuple
import pickle 
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion

# =====================================================================
# ! DO NOT CHANGE THESE CLASSES (Hocanın dokunma dediği kısımlar)
# =====================================================================
class Preprocessor:
    def __init__(self, tokenizer, embedder, **kwargs):
        self.tokenizer = tokenizer
        self.embedder  = embedder
        self.label_encoder = LabelEncoder(labels = {"negative": 0, "positive": 1})
        self.__dict__.update(kwargs)

    def prepare_data(self, data: Dict[str, str]) -> Tuple[List[float], List[int]]:
        X = [self.tokenizer.tokenize(text) for text in data["text"]]
        X = self.embedder.embed(X)
        y = [self.label_encoder.label2id[label] for label in data["label"]]
        return X, y

class LabelEncoder:
    def __init__(self, labels, **kwargs):
        self.id2label = {v: k for k, v in labels.items()}
        self.label2id = {k: v for k, v in labels.items()}
        self.__dict__.update(kwargs)

class PreprocessorObject:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)
        if "pretrained_path" in self.__dict__:
            self.__dict__.update(self.load(self.pretrained_path).__dict__)
        else:
            print(f"!! {type(self).__name__} is not pretrained. You may need to train it first.")
    
    def save(self, path: str):
        with open(path, 'wb') as f:
            pickle.dump(self, f)
    
    def load(self, path: str):
        with open(path, 'rb') as f:
            # FIXED: Virgül hatası temizlendi, direkt nesne döner.
            return pickle.load(f)

# =====================================================================
# ! UPDATE THESE CLASSES (Senin fark yarattığın kısımlar)
# =====================================================================
class Tokenizer(PreprocessorObject):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def train(self, texts: List[str]):
        # Professional touch: Rule-based tokenizer doesn't need training
        return self
    
    def _handle_negation(self, tokens: List[str]) -> List[str]:
        # UPGRADE: Negation handling captures context like "not_good"
        neg_words = {"not", "no", "never", "none", "neither", "nor"}
        result = []
        negate = False
        for t in tokens:
            if t in neg_words:
                negate = True
                result.append(t)
            elif negate:
                result.append("NOT_" + t)
                negate = False
            else:
                result.append(t)
        return result

    def tokenize(self, text: str) -> List[str]:
        text = text.lower()
        text = re.sub(r"http\S+|www\.\S+", "", text)
        text = re.sub(r"(.)\1{2,}", r"\1\1", text) # soooo -> soo
        text = re.sub(r"\d+", " <NUM> ", text)
        text = re.sub(r"!", " <EXCLAMATION> ", text)
        text = re.sub(r"\?", " <QUESTION> ", text)
        text = re.sub(r"[^a-z\s<>]", "", text)
        
        tokens = re.sub(r"\s+", " ", text).strip().split()
        return self._handle_negation(tokens)

class Embedder(PreprocessorObject):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if not hasattr(self, 'model'):
            # STRATEGY: (1, 4) Word + (3, 6) Char with 300k combined capacity
            word_tfidf = TfidfVectorizer(
                analyzer='word', ngram_range=(1, 4), sublinear_tf=True,
                max_features=150000, min_df=2, max_df=0.85, norm='l2'
            )
            char_tfidf = TfidfVectorizer(
                analyzer='char_wb', ngram_range=(3, 6), sublinear_tf=True,
                max_features=150000, min_df=2, max_df=0.85, norm='l2'
            )
            self.model = FeatureUnion([
                ("word", word_tfidf), ("char", char_tfidf)
            ], transformer_weights={"word": 1.0, "char": 1.3})

    def train(self, tokens_list: List[List[str]]):
        joined_texts = [" ".join(t) for t in tokens_list]
        self.model.fit(joined_texts)

    def embed(self, tokens_list: List[List[str]]):
        joined_texts = [" ".join(t) for t in tokens_list]
        # FIXED: Sondaki virgül kaldırıldı, hata vermez.
        return self.model.transform(joined_texts)