
import pandas as pd
from pathlib import Path
import html
import re

input_file = "data/Reviews.csv"
output_file = "data/processed_reviews.csv"

print("Loading customer reviews...")

df = pd.read_csv(
    input_file,
    usecols=["Id", "ProductId", "Score", "Time", "Summary", "Text"]
)

print("Original rows:", len(df))

df = df.dropna(subset=["Text"]).copy()

df = df.drop_duplicates(subset=["Id"]).copy()


def clean_text(value):
    if pd.isna(value):
        return ""

    text = html.unescape(str(value))

    text = re.sub(r"(?i)<br\s*/?>", " ", text)

    text = re.sub(r"<[^>]+>", " ", text)

    text = re.sub(r"\s+", " ", text).strip()

    return text


df["Text"] = df["Text"].apply(clean_text)
df["Summary"] = df["Summary"].apply(clean_text)

df = df[df["Text"] != ""].copy()

print("Rows after cleaning:", len(df))


df["ReviewDate"] = pd.to_datetime(
    df["Time"],
    unit="s",
    errors="coerce"
)

def label_sentiment(score):
    if score <= 2:
        return "Negative"
    elif score == 3:
        return "Neutral"
    else:
        return "Positive"


df["Sentiment"] = df["Score"].apply(label_sentiment)


df["ReviewText"] = (
    df["Summary"] + ". " + df["Text"]
).str.strip(" .")


Path("data").mkdir(exist_ok=True)

df.to_csv(output_file, index=False)

print("\nPreprocessing completed!")
print("Processed rows:", len(df))

print("\nSentiment distribution:")
print(df["Sentiment"].value_counts())

print("\nMissing values in important columns:")
print(df[["ReviewText", "ReviewDate", "Sentiment"]].isna().sum())

print("\nSample cleaned review:")
print(df["ReviewText"].iloc[0][:300])

print("\nSaved to:", output_file)
