# Text Classification Example (AG News Dataset, Fully Compatible)
# Works with the current Hugging Face 'datasets' API

from datasets import load_dataset
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
import re, string

# Load the dataset
dataset = load_dataset("ag_news")

# Convert to pandas DataFrame
df = dataset["train"].to_pandas()

# Inspect available columns
print("Columns:", df.columns.tolist())

# Handle both possible schema versions
if "text" in df.columns:
    df["combined_text"] = df["text"]
elif all(col in df.columns for col in ["title", "description"]):
    df["combined_text"] = df["title"] + " " + df["description"]
else:
    raise ValueError("Unexpected column names in AG News dataset.")

# Clean text
def clean_text(text):
    text = text.lower()
    text = re.sub(f"[{re.escape(string.punctuation)}]", "", text)
    return text

df["clean_text"] = df["combined_text"].apply(clean_text)

# Split into train/test
X_train, X_test, y_train, y_test = train_test_split(
    df["clean_text"], df["label"], test_size=0.2, random_state=42
)

# TF-IDF Vectorization
vectorizer = TfidfVectorizer(max_features=5000, stop_words="english")
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Train model
model = LogisticRegression(max_iter=300)
model.fit(X_train_vec, y_train)

# Predict and evaluate
y_pred = model.predict(X_test_vec)

# Label mapping
label_map = {0: "World", 1: "Sports", 2: "Business", 3: "Sci/Tech"}

print("Classification Report:\n", classification_report(y_test, y_pred, target_names=label_map.values()))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
