import pandas as pd
file_path="data/meajor/meajor_cleaned_preprocessed.csv"
df= pd.read_csv(file_path)
print("First 5 rows:")
print(df.head())
print("Columns:")
print(df.columns)
print("Shape:")
print(df.shape)
print("Info:")
print(df.info())
