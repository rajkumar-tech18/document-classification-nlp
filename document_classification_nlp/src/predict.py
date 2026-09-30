"""
predict.py
----------
Loads the trained model + TF-IDF vectorizer and classifies new,
unseen text into one of the five news categories.

Usage:
    # Classify one string passed on the command line
    python src/predict.py "The striker scored twice as the team won the final."

    # Or run with no arguments for an interactive prompt
    python src/predict.py
"""

import sys
from pathlib import Path

import joblib

sys.path.append(str(Path(__file__).resolve().parent))
from preprocess import clean_text  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "models"


def load_artifacts():
    model_path = MODELS_DIR / "best_model.joblib"
    vec_path = MODELS_DIR / "vectorizer.joblib"
    if not model_path.exists() or not vec_path.exists():
        raise FileNotFoundError(
            "Trained model not found. Run `python src/train.py` first."
        )
    model = joblib.load(model_path)
    vectorizer = joblib.load(vec_path)
    return model, vectorizer


def predict(text: str, model, vectorizer):
    cleaned = clean_text(text)
    vec = vectorizer.transform([cleaned])
    label = model.predict(vec)[0]

    # Not every classifier exposes predict_proba (LinearSVC does not by
    # default), so fall back to decision_function-based confidence when
    # probabilities aren't available.
    confidence = None
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(vec)[0]
        confidence = max(proba)
    elif hasattr(model, "decision_function"):
        scores = model.decision_function(vec)[0]
        # Softmax over the decision scores gives a comparable "confidence"
        import numpy as np
        exp_scores = np.exp(scores - np.max(scores))
        probs = exp_scores / exp_scores.sum()
        confidence = max(probs)

    return label, confidence


def main():
    model, vectorizer = load_artifacts()

    if len(sys.argv) > 1:
        text = " ".join(sys.argv[1:])
        label, confidence = predict(text, model, vectorizer)
        if confidence is not None:
            print(f"Predicted category: {label}  (confidence: {confidence:.2%})")
        else:
            print(f"Predicted category: {label}")
        return

    print("Document Classification — interactive mode")
    print("Type a sentence or short paragraph and press Enter (Ctrl+C to quit).\n")
    while True:
        try:
            text = input(">> ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nExiting.")
            break
        if not text:
            continue
        label, confidence = predict(text, model, vectorizer)
        if confidence is not None:
            print(f"Predicted category: {label}  (confidence: {confidence:.2%})\n")
        else:
            print(f"Predicted category: {label}\n")


if __name__ == "__main__":
    main()
