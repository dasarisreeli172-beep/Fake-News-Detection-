from pathlib import Path
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "news.csv"
MODEL_PATH = BASE_DIR / "model.joblib"

def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            "Dataset not found. Add data/news.csv with columns: text,label"
        )

    df = pd.read_csv(DATA_PATH)
    required = {"text", "label"}
    if not required.issubset(df.columns):
        raise ValueError("CSV must contain columns named 'text' and 'label'.")

    df = df.dropna(subset=["text", "label"])
    df["text"] = df["text"].astype(str).str.strip()
    df["label"] = df["label"].astype(str).str.strip().str.lower()
    df = df[df["text"] != ""]
    df = df[df["label"].isin(["real", "fake"])]

    if len(df) < 10 or df["label"].nunique() < 2:
        raise ValueError(
            "Please use a larger labeled dataset containing both 'real' and 'fake' rows."
        )

    X_train, X_test, y_train, y_test = train_test_split(
        df["text"], df["label"], test_size=0.25, random_state=42,
        stratify=df["label"]
    )

    model = Pipeline([
        ("tfidf", TfidfVectorizer(stop_words="english", max_features=20000,
                                  ngram_range=(1, 2))),
        ("classifier", LogisticRegression(max_iter=1000, class_weight="balanced"))
    ])
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    print(f"Test accuracy: {accuracy_score(y_test, predictions):.3f}")
    print("\nClassification report:")
    print(classification_report(y_test, predictions, zero_division=0))

    joblib.dump(model, MODEL_PATH)
    print(f"\nSaved trained model to: {MODEL_PATH}")
    print("Note: accuracy depends on dataset quality and does not prove a claim is true.")

if __name__ == "__main__":
    main()
