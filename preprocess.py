import pandas as pd

file_path = "data/meajor/meajor_cleaned_preprocessed.csv"

df = pd.read_csv(file_path)

print("Original shape:", df.shape)

print("\nRows with missing label:")
print(df[df["label"].isna()])

df = df.dropna(subset=["label"])

print("\nShape after removing unlabeled rows:", df.shape)

print("\nMissing text values:")
print(df[["subject", "body"]].isna().sum())

df["subject"] = df["subject"].fillna("")
df["body"] = df["body"].fillna("")
print("\nMissing text values after filling:")
print(df[["subject", "body"]].isna().sum())

df["text"] = df["subject"] + " " + df["body"]
print("\nCombined text examples:")
print(df["text"].head(3))

print("\nCombined text length:")
print(df["text"].str.len().describe())

print("\nFinal label distribution:")
print(df["label"].value_counts())
print("\nFinal dataset shape:")
print(df.shape)