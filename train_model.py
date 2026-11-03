
import pandas as pd
import joblib
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

DATA_PATH = "data/processed_reviews.csv"
MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)

print("Loading processed reviews...")
df = pd.read_csv(DATA_PATH)

sample_size = min(100_000, len(df))

df = (
    df.groupby("Sentiment", group_keys=False)
    .apply(
        lambda group: group.sample(
            n=max(
                1,
                round(sample_size * len(group) / len(df))
            ),
            random_state=42
        )
    )
)

df = df.sample(frac=1, random_state=42).reset_index(drop=True)

df = df.dropna(subset=["ReviewText", "Sentiment"])
df = df[df["ReviewText"].str.strip() != ""]

X = df["ReviewText"]
y = df["Sentiment"]

print("Reviews selected:", len(df))

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.30,
    random_state=42,
    stratify=y
)

print("Training reviews:", len(X_train))
print("Testing reviews:", len(X_test))

model = Pipeline([
    ("tfidf", TfidfVectorizer(
        max_features=30_000,
        stop_words="english",
        ngram_range=(1, 2),
        min_df=2
    )),
    ("classifier", LogisticRegression(
        max_iter=1000,
        class_weight="balanced"
    ))
])

print("\nTraining model...")
model.fit(X_train, y_train)

print("\nEvaluating model...")
predictions = model.predict(X_test)

print("Accuracy:", round(accuracy_score(y_test, predictions), 4))
print("\nClassification report:")
print(classification_report(y_test, predictions, zero_division=0))

print("Confusion matrix:")
print(confusion_matrix(
    y_test,
    predictions,
    labels=["Negative", "Neutral", "Positive"]
))

joblib.dump(model, MODEL_DIR / "sentiment_model_70_30.joblib")

print("\nModel saved to models/sentiment_model.joblib")
