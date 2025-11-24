import os
import re
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
from joblib import dump
from train_tfidf import build_text_column
from skopt import BayesSearchCV
from skopt.space import Real

ROOT = os.path.dirname(__file__)
DATA_DIR = os.path.join(ROOT, "..", "data")
MODEL_DIR = os.path.join(ROOT, "..", "models")
os.makedirs(MODEL_DIR, exist_ok=True)


train_path = os.path.join(DATA_DIR, "sogou_train.csv")
test_path = os.path.join(DATA_DIR, "sogou_test.csv")

train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)

train_texts = build_text_column(train_df, duplicate_title=True, max_len=3000)
test_texts = build_text_column(test_df, duplicate_title=True, max_len=3000)

y_train_full = train_df["label"].astype(int).values
y_test = test_df["label"].astype(int).values

X_train, X_val, y_train, y_val = train_test_split(
        train_texts, y_train_full, test_size=0.1, stratify=y_train_full, random_state=42
    )

pipe = Pipeline([
        ("tfidf", TfidfVectorizer(
            analyzer="word",
            ngram_range=(1, 3),
            min_df=2,
            max_df=0.8,
            sublinear_tf=True,
            lowercase=False,
            dtype=np.float32,
        )),
        ("clf", LinearSVC(C=1.0)),
    ])
range_of_c={
    "clf__C": Real(1e-3, 10, prior="log-uniform")
}
    # BayesSearchCV
opt = BayesSearchCV(
    estimator=pipe,
    search_spaces=range_of_c,
    n_iter=20,         
    cv=3,             
    scoring="accuracy",
    n_jobs=1,
    random_state=42,
    verbose=1
)

opt.fit(X_train, y_train)


pipe = opt.best_estimator_
print("\nC for the model:", opt.best_params_)
print("\n--- Validation Metrics ---")
y_val_pred = pipe.predict(X_val)
print(classification_report(y_val, y_val_pred))
print(confusion_matrix(y_val, y_val_pred))

print("\n--- Test Metrics ---")
y_test_pred = pipe.predict(test_texts)
print(classification_report(y_test, y_test_pred))
print(confusion_matrix(y_test, y_test_pred))

model_path = os.path.join(MODEL_DIR, "linearsvc_final.joblib")
dump(pipe, model_path)