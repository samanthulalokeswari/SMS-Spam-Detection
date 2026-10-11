
from pathlib import Path
import json
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "SMSSpamCollection"


def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

    df = pd.read_csv(
        DATA_PATH,
        sep="\t",
        header=None,
        names=["label", "message"],
        encoding="utf-8"
    )

    df = df.dropna(subset=["label", "message"])
    df["label"] = df["label"].str.strip().str.lower()
    df["message"] = df["message"].astype(str).str.strip()
    df = df[df["label"].isin(["ham", "spam"])]
    df = df[df["message"] != ""]
    df = df.drop_duplicates(subset=["label", "message"])

    print("Dataset loaded:", len(df), "messages")
    print(df["label"].value_counts())

    X_train, X_test, y_train, y_test = train_test_split(
        df["message"],
        df["label"],
        test_size=0.20,
        random_state=42,
        stratify=df["label"]
    )

    model = Pipeline([
        ("tfidf", TfidfVectorizer(
            lowercase=True,
            ngram_range=(1, 2),
            sublinear_tf=True
        )),
        ("classifier", MultinomialNB(alpha=0.5))
    ])

    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    metrics = {
        "total_messages": int(len(df)),
        "accuracy": float(accuracy_score(y_test, predictions)),
        "spam_precision": float(precision_score(
            y_test, predictions, pos_label="spam", zero_division=0
        )),
        "spam_recall": float(recall_score(
            y_test, predictions, pos_label="spam", zero_division=0
        )),
        "spam_f1": float(f1_score(
            y_test, predictions, pos_label="spam", zero_division=0
        )),
        "confusion_matrix": confusion_matrix(
            y_test, predictions, labels=["ham", "spam"]
        ).tolist()
    }

    joblib.dump(model, BASE_DIR / "spam_model.pkl")

    with open(BASE_DIR / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=4)

    print("\nMODEL TRAINED SUCCESSFULLY")
    print(f"Accuracy:       {metrics['accuracy']:.4f}")
    print(f"Spam precision: {metrics['spam_precision']:.4f}")
    print(f"Spam recall:    {metrics['spam_recall']:.4f}")
    print(f"Spam F1-score:  {metrics['spam_f1']:.4f}")

    print("\nClassification report:")
    print(classification_report(
        y_test,
        predictions,
        labels=["ham", "spam"],
        target_names=["Not Spam", "Spam"],
        zero_division=0
    ))

    print("\nConfusion matrix (actual rows, predicted columns):")
    print(confusion_matrix(y_test, predictions, labels=["ham", "spam"]))


if __name__ == "__main__":
    main()
