import joblib
import pandas as pd
from sklearn.model_selection import train_test_split

from lime.lime_text import LimeTextExplainer
model = joblib.load("model.joblib")
vectorizer = joblib.load("vectorizer.joblib")

print("Model and vectorizer loaded successfully.")

def predict_proba(texts):
    transformed_text = vectorizer.transform(texts)
    return model.predict_proba(transformed_text)
explainer = LimeTextExplainer(
    class_names=["Legitimate", "Phishing"]
)

file_path = "data/meajor/meajor_cleaned_preprocessed.csv"

df = pd.read_csv(file_path)

df = df.dropna(subset=["label"])

df["subject"] = df["subject"].fillna("")
df["body"] = df["body"].fillna("")

df["text"] = df["subject"] + " " + df["body"]

X = df["text"]
y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)
sample_text = X_test.iloc[0]
sample_label = y_test.iloc[0]

print("\nSample email:")
print(sample_text)

print("\nActual label:", sample_label)

sample_tfidf = vectorizer.transform([sample_text])

prediction = model.predict(sample_tfidf)[0]
probabilities = model.predict_proba(sample_tfidf)[0]

print("\nModel prediction:", prediction)
print("Legitimate probability:", probabilities[0])
print("Phishing probability:", probabilities[1])

explanation = explainer.explain_instance(
    sample_text,
    predict_proba,
    labels=[1],
    num_features=10
)
print("\nLIME explanation:")

for feature, weight in explanation.as_list():
    print(f"{feature}: {weight:.4f}")