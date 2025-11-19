import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
import numpy as np
from train_tfidf import build_text_column
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV


ROOT = os.path.dirname(__file__)
DATA_DIR = os.path.join(ROOT, "..", "data")
MODEL_DIR = os.path.join(ROOT, "..", "models")

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

pipeline = Pipeline([
    ("tfidf", TfidfVectorizer(
        analyzer="word",
        ngram_range=(1, 3),
        sublinear_tf=True,
        lowercase=False,
        dtype=np.float32,
    )),
    ("clf", LinearSVC(C=1))
])

random_param = {
    "tfidf__min_df": list(range(2,6)),
    "tfidf__max_df": [0.5,0.75,1.0]
}

random_search = RandomizedSearchCV(
    estimator=pipeline,
    param_distributions=random_param,
    n_iter=8,    
    scoring="accuracy",
    n_jobs=1,
    random_state=42,
    verbose=1
)

print("\n--- Random Search ---")
random_search.fit(X_train, y_train)
print("Best Score:", random_search.best_score_)
print("Best Params:", random_search.best_params_)

best_random_min_df = random_search.best_params_['tfidf__min_df']
best_random_max_df = random_search.best_params_['tfidf__max_df']

param_grid = {
    "tfidf__min_df": [best_random_min_df-1, best_random_min_df, best_random_min_df+1],
    "tfidf__max_df": np.linspace(max(0.5, best_random_max_df - 0.1), min(1.0, best_random_max_df + 0.1), num=5).tolist()
}

grid_search = GridSearchCV(
    estimator=random_search.best_estimator_,
    param_grid=param_grid,
    scoring="accuracy",
    n_jobs=1,
    verbose=1
)

print("\n--- Grid Search---")
grid_search.fit(X_train, y_train)
print("Best Score:", grid_search.best_score_)
print("Best Params:", grid_search.best_params_)

