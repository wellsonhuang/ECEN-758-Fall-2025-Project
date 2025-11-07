import os
import re
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.calibration import CalibratedClassifierCV
from joblib import dump
from sklearn.neighbors import KNeighborsClassifier


ROOT = os.path.dirname(__file__)
DATA_DIR = os.path.join(ROOT, "..", "data")
MODEL_DIR = os.path.join(ROOT, "..", "models")
os.makedirs(MODEL_DIR, exist_ok=True)


def normalize_text(s: str) -> str:
    if not isinstance(s, str):
        return ""
    
    # remove URLs
    s = re.sub(r'https?://\S+|www\.\S+', ' ', s)

    # collapse whitespace
    s = re.sub(r'\s+', ' ', s).strip()

    return s


# combine title and content into a single string
def build_text_column(df: pd.DataFrame, duplicate_title: bool = True, max_len: int = 3000) -> pd.Series:
    title = df["title"].fillna("").map(normalize_text)
    content = df["content"].fillna("").map(normalize_text)

    # to give more weight to the title which could be the strongest topic cue
    if duplicate_title:
        text = title + " [SEP] " + title + " " + content
    else:
        text = title + " [SEP] " + content

    return text.str.slice(0, max_len)


def main():
    train_path = os.path.join(DATA_DIR, "sogou_train.csv")
    test_path = os.path.join(DATA_DIR, "sogou_test.csv")

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    train_texts = build_text_column(train_df, duplicate_title=True, max_len=3000)
    test_texts = build_text_column(test_df, duplicate_title=True, max_len=3000)

    y_train_full = train_df["label"].astype(int).values
    y_test = test_df["label"].astype(int).values

    X_train, X_val, y_train, y_val = train_test_split(
        train_texts,
        y_train_full,
        test_size=0.1,
        stratify=y_train_full,
        random_state=42,
    )

    # TF-IDF + Linear SVM pipeline and TF-IDF + Multinomial Naive Bayes pipeline
    base_clf = {
        "LinearSVC": LinearSVC(C=1.0),
        "NaiveBayes" : MultinomialNB(),
        "KNN": KNeighborsClassifier(n_neighbors=7, metric="cosine", n_jobs=-1)
    }


    # for calibrated probabilities:
    # base_clf = CalibratedClassifierCV(LinearSVC(C=1.0), method="isotonic", cv=3)
    for model_name, classifier in base_clf.items():
        print("\n" + "=" * 40)
        print(f"Training and evaluating: {model_name}")
        print("=" * 40 + "\n")

        pipe = Pipeline([
            ("tfidf", TfidfVectorizer(
                analyzer="char",
                ngram_range=(1, 2),   # TODO: try (1,3), (2,3), (1,4)
                min_df=3,
                max_df=0.9,
                sublinear_tf=True,
                lowercase=False,
                dtype=np.float32,
            )),
            ("clf", classifier),
        ])

        # fit TF-IDF + LinearSVC and TF-IDF + MultinomialNB on train split
        pipe.fit(X_train, y_train)

        # validation metrics
        print("\n--- Validation Metrics ---")
        y_val_pred = pipe.predict(X_val)
        print(classification_report(y_val, y_val_pred))
        print(confusion_matrix(y_val, y_val_pred))

        # test metrics
        print("\n--- Test Metrics ---")
        y_test_pred = pipe.predict(test_texts)
        print(classification_report(y_test, y_test_pred))
        print(confusion_matrix(y_test, y_test_pred))

        # save model
        model_filename = f"tfidf_char12_{model_name.lower()}.joblib"
        model_path = os.path.join(MODEL_DIR, model_filename)
        dump(pipe, model_path)


if __name__ == "__main__":
    main()