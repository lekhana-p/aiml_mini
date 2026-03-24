"""Train a machine learning model for addiction risk prediction.

Steps:
1. Load dataset from CSV.
2. Split into train/test sets.
3. Train a RandomForest classifier.
4. Print model performance.
5. Save model to model.pkl.
"""

from __future__ import annotations

import pickle

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split


DATASET_FILE = "digital_wellbeing/usage_dataset.csv"
MODEL_FILE = "digital_wellbeing/model.pkl"



def train_and_save_model() -> None:
    """Train model and store it as model.pkl."""

    df = pd.read_csv(DATASET_FILE)

    # Input features used for prediction.
    feature_columns = [
        "daily_screen_time",
        "phone_unlocks",
        "social_media_usage",
        "night_usage",
        "avg_session_length",
    ]

    X = df[feature_columns]
    y = df["addiction_level"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Random Forest is robust and easy for students to use.
    model = RandomForestClassifier(n_estimators=150, random_state=42)
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)

    print(f"Model accuracy: {accuracy:.2%}")
    print("\nClassification Report:")
    print(classification_report(y_test, predictions))

    with open(MODEL_FILE, "wb") as model_file:
        pickle.dump(model, model_file)

    print(f"\nModel saved at: {MODEL_FILE}")


if __name__ == "__main__":
    train_and_save_model()
