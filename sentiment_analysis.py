#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import TreebankWordTokenizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score

# --- 1. SETUP & INITIALIZATION ---
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()
tokenizer = TreebankWordTokenizer()

print("Loading data...")
df = pd.read_csv("quotes_dataset.csv", encoding="latin-1")

# Remove missing values just in case
df = df.dropna(subset=["quote", "label"]).reset_index(drop=True)

# Make sure quote column is text
df["quote"] = df["quote"].astype(str)

X_train, X_test, y_train, y_test = train_test_split(
    df["quote"], df["label"], test_size=0.2, random_state=42
)

# --- 2. TEXT CLEANING FUNCTION ---
def clean_text(text):
    text = text.lower().strip()
    tokens = tokenizer.tokenize(text)
    clean_tokens = [
        lemmatizer.lemmatize(word) for word in tokens if word.lower() not in stop_words
    ]
    return " ".join(clean_tokens)

# --- 3. PROGRESS TRACKER SETUP ---
class ProgressCleaner:
    def __init__(self, total_rows, dataset_name):
        self.count = 0
        self.total = total_rows
        self.dataset_name = dataset_name

    def clean_and_count(self, text):
        self.count += 1
        if self.count % 200 == 0:
            print(f"[{self.dataset_name}] Cleaning row {self.count} out of {self.total}")
        return clean_text(text)

# --- 4. APPLY CLEANING WITH PROGRESS ---
print("\nOperating on training data...")
train_tracker = ProgressCleaner(len(X_train), "Training Data")
X_train_clean = X_train.apply(train_tracker.clean_and_count)

print("\nOperating on test data...")
test_tracker = ProgressCleaner(len(X_test), "Test Data")
X_test_clean = X_test.apply(test_tracker.clean_and_count)

# --- 5. PIPELINE ---
print("\nBuilding and Training the Pipeline...")

sentiment_pipeline = Pipeline([
    ('vectorizer', TfidfVectorizer(analyzer="word", ngram_range=(1, 2), max_features=10000)),
    ('classifier', LogisticRegression(C=100, random_state=0, solver='liblinear'))
])

sentiment_pipeline.fit(X_train_clean, y_train)

# --- 6. EVALUATION ---
print("\nEvaluating model on test data...")
predictions = sentiment_pipeline.predict(X_test_clean)

accuracy = accuracy_score(y_test, predictions) * 100
print(f"The accuracy of the model is {accuracy:.0f}%")

# --- 7. INTERPRETATION ---
print("\nFirst 20 predictions:")
print(predictions[:20])

print("\nPrediction ratio:")
print(pd.Series(predictions).value_counts(normalize=True))

print("\nConfusion matrix:")
print(pd.crosstab(y_test, predictions))

feature_names = sentiment_pipeline.named_steps['vectorizer'].get_feature_names_out()
coefficients = sentiment_pipeline.named_steps['classifier'].coef_[0]

top_positive = sorted(zip(coefficients, feature_names), reverse=True)[:10]
top_negative = sorted(zip(coefficients, feature_names))[:10]

print("\nTop words for class 1:")
print(top_positive)

print("\nTop words for class 0:")
print(top_negative)
