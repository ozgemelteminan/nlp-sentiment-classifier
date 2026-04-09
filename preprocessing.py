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
            return pickle.load(f)

class Tokenizer(PreprocessorObject):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def train(self, texts: List[str]):
        return self
    
    def _handle_negation(self, tokens: List[str]) -> List[str]:
        neg_words = {
            "not", "no", "never", "none", "neither", "nor", "cannot",
            "dont", "isnt", "wasnt", "didnt", "wouldnt", "couldnt", 
            "shouldnt", "havent", "hasnt", "hadnt", "arent", "werent", 
            "aint", "wont", "cant"
        }
        scope_breakers = {
            "<exclamation>", "<multiexclaim>",
            "<question>", "<multiquestion>",
            "<smile>", "<sad>"
        }
        result = []
        negate_count = 0 
        for t in tokens:
            if t in neg_words:
                negate_count = 4 
                result.append(t)
            elif t in scope_breakers:
                negate_count = 0
                result.append(t)
            elif negate_count > 0:
                result.append("NOT_" + t)
                negate_count -= 1
            else:
                result.append(t)
        return result

    def tokenize(self, text: str) -> List[str]:
        # 1. HTML VE GÜRÜLTÜ TEMİZLİĞİ
        text = re.sub(r"<br\s*/?>", " ", text)
        text = re.sub(r"<[^>]+>", " ", text)         # BUG FIX: <b>,<i>,<p> vs. → <> token'ı üretiyordu (1553 kez!)
        text = re.sub(r"&\w+;", " ", text)            # HTML entity'leri (&amp;, &quot;)
        text = re.sub(r"(\w)-(\w)", r"\1 \2", text)   # Tireli kelimeler (well-made → well made)

        # 2. İNTERNET JARGONU VE GÜNLÜK DİL
        text = re.sub(r"\bomg\b", "oh my god", text, flags=re.IGNORECASE)
        text = re.sub(r"\blol\b", " <smile> ", text, flags=re.IGNORECASE)
        text = re.sub(r"\bwtf\b", "what the heck", text, flags=re.IGNORECASE)
        text = re.sub(r"\baka\b", "also known as", text, flags=re.IGNORECASE)
        text = re.sub(r"\bgonna\b", "going to", text, flags=re.IGNORECASE)
        text = re.sub(r"\bwanna\b", "want to", text, flags=re.IGNORECASE)
        text = re.sub(r"\bgotta\b", "got to", text, flags=re.IGNORECASE)
        text = re.sub(r"\by'all\b", "you all", text, flags=re.IGNORECASE)

        # 3. EKSİK KESME İŞARETLİ VE GİZLİ NEGATİFLER
        # BUG FIX: can't ve won't ÖNCE explicit yakalanmalı —
        # aksi halde n't pattern'ı "ca" ve "wo" artığı bırakıyordu (52 + 17 kez!)
        text = re.sub(r"\bcan't\b", "can not", text, flags=re.IGNORECASE)
        text = re.sub(r"\bwon't\b", "will not", text, flags=re.IGNORECASE)

        text = re.sub(r"\bdidnt\b", "did not", text, flags=re.IGNORECASE)
        text = re.sub(r"\bdont\b", "do not", text, flags=re.IGNORECASE)
        text = re.sub(r"\bdoesnt\b", "does not", text, flags=re.IGNORECASE)
        text = re.sub(r"\bisnt\b", "is not", text, flags=re.IGNORECASE)
        text = re.sub(r"\bwasnt\b", "was not", text, flags=re.IGNORECASE)
        text = re.sub(r"\bwouldnt\b", "would not", text, flags=re.IGNORECASE)
        text = re.sub(r"\bcouldnt\b", "could not", text, flags=re.IGNORECASE)
        text = re.sub(r"\bshouldnt\b", "should not", text, flags=re.IGNORECASE)
        text = re.sub(r"\bcant\b", "can not", text, flags=re.IGNORECASE)
        text = re.sub(r"\bwont\b", "will not", text, flags=re.IGNORECASE)
        text = re.sub(r"\bain't\b|\baint\b", "is not", text, flags=re.IGNORECASE)
        text = re.sub(r"\bcannot\b", "can not", text, flags=re.IGNORECASE)

        # 4. STANDART KISALTMALAR
        text = re.sub(r"n't\b", " not", text, flags=re.IGNORECASE)
        text = re.sub(r"'re\b", " are", text, flags=re.IGNORECASE)
        text = re.sub(r"'ve\b", " have", text, flags=re.IGNORECASE)
        text = re.sub(r"'ll\b", " will", text, flags=re.IGNORECASE)
        text = re.sub(r"'d\b",  " would", text, flags=re.IGNORECASE)
        text = re.sub(r"'m\b",  " am", text, flags=re.IGNORECASE)
        text = re.sub(r"'s\b",  " is", text, flags=re.IGNORECASE)

        # 5. BÜYÜK HARF, EMOJİ, URL (lowercase bu adımda)
        text = re.sub(r'\b([A-Z]{2,})\b', r'\1 <ALLCAPS>', text)
        text = text.lower()   # ← bundan sonra eklenen token'lar lowercase olmalı!

        text = re.sub(r"(:\)|:-\)|=\)|:d|:-d|<3)", " <smile> ", text)
        text = re.sub(r"(:\(|:-\(|=\(|:'\()", " <sad> ", text)
        text = re.sub(r"http\S+|www\.\S+", " <url> ", text)
        text = re.sub(r"([a-z])\1{2,}", r"\1\1 <elongated> ", text)
        text = re.sub(r"\d+", " <num> ", text)

        # 6. NOKTALAMA İŞARETLERİ
        # BUG FIX: lowercase'den SONRA eklenen token'lar küçük harfli olmalı,
        # yoksa [^a-z\s<>] temizliği büyük harfleri silip <> bırakıyordu!
        text = re.sub(r"\.{2,}", " <ellipsis> ", text)
        text = re.sub(r"!{2,}", " <multiexclaim> ", text)   # önce çoklu
        text = re.sub(r"!", " <exclamation> ", text)         # sonra tekli
        text = re.sub(r"\?{2,}", " <multiquestion> ", text)  # önce çoklu
        text = re.sub(r"\?", " <question> ", text)           # sonra tekli

        text = re.sub(r"[^a-z\s<>]", "", text)

        tokens = re.sub(r"\s+", " ", text).strip().split()
        return self._handle_negation(tokens)


class Embedder(PreprocessorObject):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if not hasattr(self, 'model'):
            word_tfidf = TfidfVectorizer(
                analyzer='word', 
                token_pattern=r"\S+",       
                ngram_range=(1, 3),         
                sublinear_tf=True,          
                max_features=250000,        
                min_df=2,                   
                max_df=0.90,                
                norm='l2'
            )
            char_tfidf = TfidfVectorizer(
                analyzer='char_wb', 
                ngram_range=(3, 6),         
                sublinear_tf=True,
                max_features=150000,        
                min_df=3,                   
                max_df=0.90,
                norm='l2'
            )
            self.model = FeatureUnion([
                ("word", word_tfidf), ("char", char_tfidf)
            ], transformer_weights={"word": 1.0, "char": 0.8})

    def train(self, tokens_list: List[List[str]]):
        joined_texts = [" ".join(t) for t in tokens_list]
        self.model.fit(joined_texts)

    def embed(self, tokens_list: List[List[str]]):
        joined_texts = [" ".join(t) for t in tokens_list]
        return self.model.transform(joined_texts)