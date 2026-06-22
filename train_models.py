import pandas as pd
import numpy as np
import string
import re
import pickle
import time
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier

def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"www\S+", "", text)
    text = re.sub(r"<.*?>", "", text)
    text = re.sub(r"[%s]" % re.escape(string.punctuation), " ", text)
    text = re.sub(r"\d+", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def main():
    print("Loading datasets...")
    start_time = time.time()
    
    # Load Fake and True datasets
    fake_df = pd.read_csv("Fake.csv")
    true_df = pd.read_csv("True.csv")
    
    print(f"Fake dataset shape: {fake_df.shape}")
    print(f"True dataset shape: {true_df.shape}")
    
    # Label datasets (Fake -> 0, True -> 1)
    fake_df["label"] = 0
    true_df["label"] = 1
    
    # Combine datasets
    df = pd.concat([fake_df, true_df], ignore_index=True)
    
    # Shuffle dataset
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    print(f"Combined dataset shape: {df.shape}")
    
    # Combine title and text
    print("Combining title and text...")
    df["content"] = df["title"].fillna("") + " " + df["text"].fillna("")
    
    # Clean text content
    print("Cleaning text (this may take a few seconds)...")
    df["content"] = df["content"].apply(clean_text)
    
    # Define features and target
    X = df["content"]
    y = df["label"]
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    
    print(f"Training samples: {len(X_train)}")
    print(f"Testing samples: {len(X_test)}")
    
    # Fit TF-IDF Vectorizer
    print("Fitting TF-IDF Vectorizer...")
    tfidf = TfidfVectorizer(stop_words="english", max_df=0.7, min_df=2)
    X_train_tfidf = tfidf.fit_transform(X_train)
    X_test_tfidf = tfidf.transform(X_test)
    
    print(f"TF-IDF vocabulary size: {X_train_tfidf.shape[1]}")
    
    # Save TF-IDF Vectorizer
    print("Saving TF-IDF Vectorizer...")
    with open("tfidf_vectorizer.pkl", "wb") as f:
        pickle.dump(tfidf, f, protocol=pickle.HIGHEST_PROTOCOL)
        
    # Fit Random Forest Classifier
    print("Training Random Forest Classifier (n_estimators=100, n_jobs=-1)...")
    rf_model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        n_jobs=-1
    )
    
    rf_start = time.time()
    rf_model.fit(X_train_tfidf, y_train)
    print(f"Random Forest trained in {time.time() - rf_start:.2f} seconds.")
    
    # Save Random Forest Model
    print("Saving Random Forest model...")
    with open("fake_news_random_forest.pkl", "wb") as f:
        pickle.dump(rf_model, f, protocol=pickle.HIGHEST_PROTOCOL)
        
    print(f"All done! Total execution time: {time.time() - start_time:.2f} seconds.")

if __name__ == "__main__":
    main()
