import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

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


X = df["text"]
y = df["label"]
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)
print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))

print("\nTraining label distribution:")
print(y_train.value_counts(normalize=True))

print("\nTesting label distribution:")
print(y_test.value_counts(normalize=True))

vectorizer = TfidfVectorizer(
    max_features=50000,
    ngram_range=(1, 2)
)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)
print("\nTF-IDF training shape:", X_train_tfidf.shape)
print("TF-IDF testing shape:", X_test_tfidf.shape)

model = LogisticRegression(
    max_iter=1000,
    random_state=42
)
model.fit(X_train_tfidf, y_train)
y_pred = model.predict(X_test_tfidf)

print("\nMODEL EVALUATION")

print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
print("F1 Score:", f1_score(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nFirst 20 predictions:")
print(y_pred[:20])

print("\nFirst 20 actual labels:")
print(y_test.iloc[:20].values)
