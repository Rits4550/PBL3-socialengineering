import pandas as pd
file_path="data/meajor/meajor_cleaned_preprocessed.csv"
df= pd.read_csv(file_path)

print("Shape:")
print(df.shape)
print("Info:")
print(df.info())
print("\nLABEL COUNTS")
print(df["label"].value_counts(dropna=False))

print("\nLABEL PERCENTAGES")
print(df["label"].value_counts(normalize=True, dropna=False) * 100)

print("\nSOURCES ")
print(df["source"].value_counts(dropna=False))

print("\nSAMPLE LEGITIMATE EMAIL ")
print(
    df[df["label"] == 0][["subject", "body", "source"]]
    .head(1)
    .to_string(index=False)
)

print("\nSAMPLE PHISHING EMAIL")
print(
    df[df["label"] == 1][["subject", "body", "source"]]
    .head(1)
    .to_string(index=False)
)