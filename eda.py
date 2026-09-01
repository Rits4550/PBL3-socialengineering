import pandas as pd
df= pd.read_csv("data/meajor/meajor_cleaned_preprocessed.csv")
'''print (df.shape)
print("Missing values")
print(df.isnull().sum())

print("\nLABEL COUNTS")
print(df["label"].value_counts(dropna=False))

print("\nLABEL PERCENTAGES")
print(df["label"].value_counts(normalize=True, dropna=False) * 100)  

print("\nSOURCE COUNTS")
print(df["source"].value_counts(dropna=False))

print("\nSOURCE PERCENTAGES")
print(df["source"].value_counts(normalize=True, dropna=False) * 100)

subject_length = df["subject"].fillna("").astype(str).str.len()

print("\nSUBJECT LENGTH")
print(subject_length.describe())
body_length = df["body"].fillna("").astype(str).str.len()

print("\nBODY LENGTH")
print(body_length.describe())'''

symbol_count = (
    df["subject"].fillna("").astype(str).str.count("<|SIMBOL|>")
    + df["body"].fillna("").astype(str).str.count("<|SIMBOL|>")
)

print("\n<|SIMBOL|> FREQUENCY")
print("Rows containing marker:", (symbol_count > 0).sum())
print("Total occurrences:", symbol_count.sum())

emoji_count = (
    df["subject"].fillna("").astype(str).str.count("<|EMOJI|>")
    + df["body"].fillna("").astype(str).str.count("<|EMOJI|>")
)

print("\n<|EMOJI|> FREQUENCY")
print("Rows containing marker:", (emoji_count > 0).sum())
print("Total occurrences:", emoji_count.sum()) 

print("\nSIMBOL BY LABEL")

print(
    df.assign(symbol_count= symbol_count)
      .groupby("label")["symbol_count"]
      .agg(["count", "sum", "mean"])
)
print("\nEMOJI BY LABEL")

print(
    df.assign(emoji_count=emoji_count)
      .groupby("label")["emoji_count"]
      .agg(["count", "sum", "mean"])
)

print("\nSIMBOL BY SOURCE")

print(
    df.assign(symbol_count=symbol_count)
      .groupby("source")["symbol_count"]
      .agg(["count", "sum", "mean"])
)
print("\nEMOJI BY SOURCE")

print(
    df.assign(emoji_count=emoji_count)
      .groupby("source")["emoji_count"]
      .agg(["count", "sum", "mean"])
)

print("\nMARKER OVERLAP")

symbol_present = symbol_count > 0
emoji_present = emoji_count > 0

print("Both markers:", (symbol_present & emoji_present).sum())
print("SIMBOL only:", (symbol_present & ~emoji_present).sum())
print("EMOJI only:", (~symbol_present & emoji_present).sum())
print("Neither:", (~symbol_present & ~emoji_present).sum())

print("\nMARKER PRESENCE BY LABEL")

print(
    df.assign(marker_present=symbol_present)
      .groupby("label")["marker_present"]
      .agg(["count", "sum", "mean"])
)
print("\nMARKER PRESENCE BY SOURCE")

print(
    df.assign(marker_present=symbol_present)
      .groupby("source")["marker_present"]
      .agg(["count", "sum", "mean"])
)