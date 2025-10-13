from datasets import load_dataset
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, HashingVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import re, string
import itertools
import joblib

import nltk
nltk.download("punkt", quiet=True)
nltk.download("stopwords", quiet=True)

import seaborn as sns
import matplotlib.pyplot as plt

class NTC():
    def __init__(self):
        self.dataset = None
        self.df = None
        
        # Model Datasets
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        
        # vectorized datasets
        self.X_train_vec = None
        self.X_test_vec = None
        
        # Word Vectorizers
        # self.vectorizer_tfidf = TfidfVectorizer(max_features=5000, stop_words="english")
        self.vectorizers = {
            "Count": CountVectorizer(max_features=5000, stop_words='english'),
            "TF-IDF": TfidfVectorizer(max_features=5000, stop_words='english'),
            "Hashing": HashingVectorizer(n_features=5000, alternate_sign=False)
            }
        
        # Classifiers
        # self.model_LogReg = LogisticRegression(max_iter=300)
        self.models = {
            "Logistic Regression": LogisticRegression(max_iter=1000),
            "Naive Bayes": MultinomialNB(),
            "Linear SVC": LinearSVC()
            }
        
        # Model To Use
        self.model = None
        
        # vectorizer to use
        self.vectorizer = None
    
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
        self.vectorizer = vectorizer
        
        self.X_test_vec = vectorizer.transform(self.X_test)
    
    def load_model(self, model):
        self.model = model
    
    def train(self):
        self.model.fit(self.X_train_vec, self.y_train)
    
    def predict_and_evaluate(self, plot_confusion_matrix=False):
        # Predict and evaluate
        y_pred = self.model.predict(self.X_test_vec)

        # Label mapping
        label_map = {0: "World", 1: "Sports", 2: "Business", 3: "Sci/Tech"}

        print("Classification Report:\n", classification_report(self.y_test, y_pred, target_names=label_map.values()))
        print("Confusion Matrix:\n", confusion_matrix(self.y_test, y_pred))
        
        if plot_confusion_matrix:
            cm = confusion_matrix(self.y_test, y_pred, labels=[0,1,2,3])

            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                        xticklabels=label_map.values(),
                        yticklabels=label_map.values())
            plt.xlabel("Predicted")
            plt.ylabel("True")
            plt.title("AG News Confusion Matrix")
            plt.show()
        
        acc = accuracy_score(self.y_test, y_pred)
        return acc

    def run(self, use_nltk=False, plot_confusion_matrix=False):
        self.download_dataset(use_nltk)
        self.load_model_datasets()
        
        results = []

        # Iterate over all combinations
        for vec_name, model_name in itertools.product(self.vectorizers.keys(), self.models.keys()):
            print("\n", vec_name + ",", model_name, "\n")
            self.vectorize(self.vectorizers[vec_name])
            self.load_model(self.models[model_name])
            self.train()
            
            acc = self.predict_and_evaluate(plot_confusion_matrix)
            results.append((vec_name, model_name, acc))
        
        df_results = pd.DataFrame(results, columns=["Vectorizer", "Model", "Accuracy"])
        print(df_results.sort_values(by="Accuracy", ascending=False).reset_index(drop=True))
        
        best_row = df_results.sort_values(by="Accuracy", ascending=False).iloc[0]
        best_vec_name = best_row["Vectorizer"]
        best_model_name = best_row["Model"]
        best_accuracy = best_row["Accuracy"]

        print(f"\nBest combination: {best_vec_name} + {best_model_name} (Accuracy: {best_accuracy:.4f})")

        # Re-train the best model fully
        self.vectorizer = self.vectorizers[best_vec_name]
        X_train_vec = self.vectorizer.fit_transform(self.X_train)
        joblib.dump(self.vectorizer, "best_vectorizer.pkl")
        
        self.load_model(self.models[best_model_name])
        self.train()
        # Save both model and vectorizer
        joblib.dump(self.model, "best_model.pkl")
        

if __name__ == "__main__":
    ntc = NTC()
    ntc.run(use_nltk=False, plot_confusion_matrix=False)
