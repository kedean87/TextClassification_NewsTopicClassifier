# Text Classification (AG News Dataset)

from datasets import load_dataset
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import re, string

import nltk
nltk.download("punkt", quiet=True)
nltk.download("stopwords", quiet=True)

class NTC():
    def __init__(self):
        self.dataset = None
        self.df = None
        self.text = None
        
        # Model Datasets
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        
        # vectorized datasets
        self.X_train_vec = None
        self.X_test_vec = None
        
        # Word Vectorizers
        self.vectorizer_tfidf = TfidfVectorizer(max_features=5000, stop_words="english")
        
        # Classifiers
        self.model_LogReg = LogisticRegression(max_iter=300)
        
        # Model To Use
        self.model = None
    
    def download_dataset(self, use_nltk=False):
        # download from hugging face
        self.dataset = load_dataset("ag_news")
        self.df = self.dataset["train"].to_pandas()
        
        # Inspect available columns
        print("Columns:", self.df.columns.tolist())
        
        # Handle both possible schema versions
        if "text" in self.df.columns:
            self.df["combined_text"] = self.df["text"]
        elif all(col in df.columns for col in ["title", "description"]):
            self.df["combined_text"] = self.df["title"] + " " + self.df["description"]
        else:
            raise ValueError("Unexpected column names in AG News dataset.")
        
        # USE NLTK
        def clean_text_nltk(text):
            tokens = word_tokenize(text)
            
            stop_words = set(stopwords.words("english"))
            
            # Keeps words/numbers only and no stopwords
            tokens = [w for w in tokens if w.isalnum() and w not in stop_words]
            clean_text = " ".join(tokens)
            return clean_text
        
        # use re and punctuation
        def clean_text_re(text):
            text = text.lower()
            text = re.sub(f"[{re.escape(string.punctuation)}]", "", text)
            return text
        
        # select which text processing method to use
        cleaner = clean_text_nltk if use_nltk else clean_text_re
        self.df["clean_text"] = self.df["combined_text"].apply(cleaner)
        
    def load_model_datasets(self):
        # Split into train/test
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.df["clean_text"], self.df["label"], test_size=0.2, random_state=42
        )
        
        self.y_train = self.y_train.astype(int)
        self.y_test = self.y_test.astype(int)
    
    def vectorize(self, vectorizer):
        self.X_train_vec = vectorizer.fit_transform(self.X_train)
        self.X_test_vec = vectorizer.transform(self.X_test)
    
    def load_model(self, model):
        self.model = model
    
    def train(self):
        self.model.fit(self.X_train_vec, self.y_train)
    
    def predict_and_evaluate(self):
        # Predict and evaluate
        y_pred = self.model.predict(self.X_test_vec)

        # Label mapping
        label_map = {0: "World", 1: "Sports", 2: "Business", 3: "Sci/Tech"}

        print("Classification Report:\n", classification_report(self.y_test, y_pred, target_names=label_map.values()))
        print("Confusion Matrix:\n", confusion_matrix(self.y_test, y_pred))

    def run(self, use_nltk=False):
        self.download_dataset(use_nltk)
        self.load_model_datasets()
        self.vectorize(self.vectorizer_tfidf)
        self.load_model(self.model_LogReg)
        self.train()
        self.predict_and_evaluate()

if __name__ == "__main__":
    ntc = NTC()
    ntc.run(use_nltk=False)
