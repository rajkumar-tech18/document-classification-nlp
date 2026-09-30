"""
train.py
--------
Trains and compares two classical NLP text-classification pipelines on
the news-articles-by-topic dataset:

    1. TF-IDF + Multinomial Naive Bayes
    2. TF-IDF + Linear SVM (LinearSVC)

Steps:
    1. Load data/news_dataset.csv
    2. Clean text (preprocess.clean_corpus)
    3. Train/test split (stratified)
    4. Vectorize with TF-IDF (unigrams + bigrams, English stopwords removed)
    5. Train both models, evaluate accuracy / precision / recall / F1
    6. Save a confusion matrix and classification report for each model
    7. Save the best-performing model + the fitted vectorizer with joblib

Run:
    python src/train.py
"""

import sys
import time
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")  # headless-safe backend
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC

sys.path.append(str(Path(__file__).resolve().parent))
from preprocess import clean_corpus  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "news_dataset.csv"
MODELS_DIR = ROOT / "models"
OUTPUTS_DIR = ROOT / "outputs"

RANDOM_STATE = 42
TEST_SIZE = 0.2


def load_data():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"{DATA_PATH} not found. Run `python src/generate_dataset.py` first."
        )
    df = pd.read_csv(DATA_PATH)
    df = df.dropna(subset=["text", "category"])
    return df


def evaluate_model(name, model, X_test_vec, y_test, labels):
    preds = model.predict(X_test_vec)
    acc = accuracy_score(y_test, preds)
    report = classification_report(y_test, preds, labels=labels, zero_division=0)

    print(f"\n=== {name} ===")
    print(f"Accuracy: {acc:.4f}")
    print(report)

    # Save classification report to a text file
    report_path = OUTPUTS_DIR / f"classification_report_{name.replace(' ', '_').lower()}.txt"
    report_path.write_text(f"{name}\nAccuracy: {acc:.4f}\n\n{report}")

    # Save confusion matrix plot
    cm = confusion_matrix(y_test, preds, labels=labels)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    fig, ax = plt.subplots(figsize=(6, 6))
    disp.plot(ax=ax, xticks_rotation=45, colorbar=False, cmap="Blues")
    ax.set_title(f"Confusion Matrix — {name}")
    fig.tight_layout()
    fig_path = OUTPUTS_DIR / f"confusion_matrix_{name.replace(' ', '_').lower()}.png"
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)

    return acc


def main():
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading data...")
    df = load_data()
    print(f"Loaded {len(df)} documents across {df['category'].nunique()} categories:")
    print(df["category"].value_counts().to_string())

    print("\nCleaning text...")
    df["clean_text"] = clean_corpus(df["text"])

    X_train, X_test, y_train, y_test = train_test_split(
        df["clean_text"],
        df["category"],
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=df["category"],
    )
    labels = sorted(df["category"].unique())

    print("\nVectorizing text with TF-IDF...")
    vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
        max_features=20000,
        min_df=2,
    )
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
    print(f"Vocabulary size: {len(vectorizer.vocabulary_)}")

    results = {}

    print("\nTraining Naive Bayes...")
    t0 = time.time()
    nb_model = MultinomialNB()
    nb_model.fit(X_train_vec, y_train)
    print(f"  done in {time.time() - t0:.2f}s")
    results["Naive Bayes"] = (nb_model, evaluate_model("Naive Bayes", nb_model, X_test_vec, y_test, labels))

    print("\nTraining Linear SVM...")
    t0 = time.time()
    svm_model = LinearSVC(random_state=RANDOM_STATE)
    svm_model.fit(X_train_vec, y_train)
    print(f"  done in {time.time() - t0:.2f}s")
    results["Linear SVM"] = (svm_model, evaluate_model("Linear SVM", svm_model, X_test_vec, y_test, labels))

    # Pick the best model by test accuracy
    best_name, (best_model, best_acc) = max(results.items(), key=lambda kv: kv[1][1])
    print(f"\nBest model: {best_name} (accuracy = {best_acc:.4f})")

    joblib.dump(best_model, MODELS_DIR / "best_model.joblib")
    joblib.dump(vectorizer, MODELS_DIR / "vectorizer.joblib")
    (MODELS_DIR / "best_model_name.txt").write_text(best_name)

    # Save a simple comparison summary
    summary_lines = ["Model comparison (test accuracy):"]
    for name, (_, acc) in results.items():
        summary_lines.append(f"  {name}: {acc:.4f}")
    summary_lines.append(f"\nSelected best model: {best_name}")
    (OUTPUTS_DIR / "model_comparison.txt").write_text("\n".join(summary_lines))

    print(f"\nSaved model      -> {MODELS_DIR / 'best_model.joblib'}")
    print(f"Saved vectorizer -> {MODELS_DIR / 'vectorizer.joblib'}")
    print(f"Saved reports/plots -> {OUTPUTS_DIR}/")


if __name__ == "__main__":
    main()
