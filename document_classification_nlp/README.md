# Document Classification using NLP Techniques

A machine learning project that classifies short news articles into one of
five topic categories — **Sports, Politics, Technology, Business,
Entertainment** — using classical NLP feature engineering (TF-IDF) and two
classical machine learning classifiers (Naive Bayes and Linear SVM).

## Project Structure

```
document_classification_nlp/
├── README.md
├── requirements.txt
├── data/
│   └── news_dataset.csv          # labeled dataset (text, category)
├── models/                       # created by train.py
│   ├── best_model.joblib
│   ├── vectorizer.joblib
│   └── best_model_name.txt
├── outputs/                      # created by train.py
│   ├── classification_report_naive_bayes.txt
│   ├── classification_report_linear_svm.txt
│   ├── confusion_matrix_naive_bayes.png
│   ├── confusion_matrix_linear_svm.png
│   └── model_comparison.txt
└── src/
    ├── generate_dataset.py       # builds data/news_dataset.csv
    ├── preprocess.py             # text cleaning shared by train/predict
    ├── train.py                  # trains, evaluates, saves the best model
    └── predict.py                # classifies new text with the saved model
```

## Setup

```bash
pip install -r requirements.txt
```

Requires Python 3.9+. No external corpus downloads (e.g. NLTK data) are
needed — stopword removal is handled by scikit-learn's built-in English
stopword list.

## How to Run

### 1. Generate the dataset

```bash
python src/generate_dataset.py
```

This writes `data/news_dataset.csv`. A copy is already included, so this
step is only needed if you want to regenerate it (e.g. after changing the
templates or the random seed).

### 2. Train and evaluate the models

```bash
python src/train.py
```

This will:
- Load and clean the dataset
- Split it into train/test sets (80/20, stratified by category)
- Vectorize the text with **TF-IDF** (unigrams + bigrams, English stopwords
  removed, max 20,000 features)
- Train a **Multinomial Naive Bayes** classifier and a **Linear SVM**
  (`LinearSVC`)
- Print accuracy, precision, recall, and F1-score for both models
- Save a confusion matrix image and classification report for each model to
  `outputs/`
- Save whichever model scored higher on the test set to `models/`

### 3. Classify new text

```bash
# One-off classification
python src/predict.py "The central bank raised interest rates this week."

# Interactive mode
python src/predict.py
>> The striker scored twice as the home team won the final.
Predicted category: Sports  (confidence: 85.25%)
```

## Approach

### Why a synthetic dataset?

Standard benchmark corpora for topic classification (e.g. the 20-Newsgroups
dataset) require downloading data from the internet at run time, which makes
a project fragile to run on an offline or restricted machine (including a
classroom lab or exam setting). Instead, `generate_dataset.py` builds an
equivalent labeled dataset locally:

- Each category has several **sentence templates** with slots (e.g. a
  company, an action, a metric) filled from category-specific word lists.
  Filling templates randomly — rather than picking from a fixed list of
  whole sentences — produces thousands of distinct sentence combinations,
  so the same exact sentence rarely appears in both the train and test
  split.
- To keep the task **realistic rather than trivially easy**, ~25% of a
  document's sentences are occasionally borrowed from a topically related
  category (e.g. a Business article may include a Technology-flavored
  sentence, mirroring how real "tech earnings" stories span both sections),
  and ~5% of documents are given a randomly wrong label to mimic
  real-world annotation noise. This is why the model's accuracy is in the
  mid-to-high 80s rather than a suspicious 100%.
- The result: 1,100 labeled documents (220 per category) in
  `data/news_dataset.csv`, with realistic class confusion (mainly between
  Business and Technology, and between Sports and Entertainment).

### Pipeline

1. **Preprocessing** (`preprocess.py`): lowercase, strip URLs/HTML/
   non-alphabetic characters, collapse whitespace.
2. **Feature extraction**: `TfidfVectorizer(stop_words="english",
   ngram_range=(1,2), max_features=20000, min_df=2)` — converts cleaned
   text into weighted term-frequency vectors, including bigrams to capture
   short phrases (e.g. "interest rates").
3. **Modeling**: two classical algorithms are trained and compared —
   - **Multinomial Naive Bayes** — a strong, fast baseline for text
     classification that models word-count likelihoods per class.
   - **Linear SVM (`LinearSVC`)** — typically the strongest classical
     linear model for high-dimensional sparse text features.
4. **Evaluation**: accuracy, per-class precision/recall/F1, and a confusion
   matrix for each model on a held-out 20% test split.
5. **Model selection**: the higher-scoring model on the test set is saved
   for use in `predict.py`.

### Results (on the included dataset)

| Model         | Test Accuracy |
|---------------|---------------|
| Naive Bayes   | ~0.85         |
| Linear SVM    | ~0.85         |

Both classical models perform similarly. The confusion matrices in
`outputs/` show most errors occur between **Business ↔ Technology** and
**Sports ↔ Entertainment** — categories that intentionally share
overlapping vocabulary, which mirrors real-world ambiguity in news
categorization.

## Possible Extensions

- Swap in a real corpus (e.g. `sklearn.datasets.fetch_20newsgroups` or the
  BBC News dataset) when internet access is available — `preprocess.py`,
  `train.py`, and `predict.py` all work unchanged as long as the CSV has
  `text` and `category` columns.
- Add TF-IDF + Logistic Regression or a Random Forest as a third baseline.
- Replace TF-IDF with word embeddings (Word2Vec/GloVe) or a transformer
  (e.g. `bert-base-uncased` via `sentence-transformers`) for a deep-learning
  comparison.
- Add k-fold cross-validation and a grid search over `max_features`,
  `ngram_range`, and the Naive Bayes/SVM hyperparameters.
- Build a small Flask/Streamlit UI around `predict.py` for a live demo.
