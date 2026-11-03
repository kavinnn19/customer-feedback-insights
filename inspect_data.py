
import pandas as pd

file_path = "data/Reviews.csv"

print("Loading dataset...")
df = pd.read_csv(file_path)

print("\nDataset shape:")
print(f"Rows: {df.shape[0]}")
print(f"Columns: {df.shape[1]}")

print("\nMissing values:")
print(df[["Text", "Summary", "Score", "Time"]].isnull().sum())

print("\nRating distribution:")
print(df["Score"].value_counts().sort_index())

print("\nReview date range:")
df["ReviewDate"] = pd.to_datetime(
    df["Time"], unit="s", errors="coerce"
)
print("Start:", df["ReviewDate"].min())
print("End:", df["ReviewDate"].max())
