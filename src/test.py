import os
import pandas as pd
from joblib import load
from train_tfidf import build_text_column
import gdown
from sklearn.metrics import classification_report, confusion_matrix

ROOT = os.path.dirname(__file__)
DATA_DIR = os.path.join(ROOT, "..", "data")

MODEL_DIR = os.path.join(ROOT, "..", "models")
os.makedirs(MODEL_DIR, exist_ok=True)
model_path = os.path.join(MODEL_DIR, "linearsvc_final.joblib")
GDRIVE_URL = "https://drive.google.com/uc?id=1sKB1KjB2gCAJtwa-VyQD9QeC3wguPa6c"

if not os.path.exists(model_path):
    print("Model not found")
    gdown.download(GDRIVE_URL, model_path, quiet=False)
    print("Download")

model = load(model_path)

test_path = os.path.join(DATA_DIR, "sogou_test.csv")
test_df = pd.read_csv(test_path)

test_texts = build_text_column(test_df, duplicate_title=True, max_len=3000)
y_test = test_df["label"].astype(int).values
y_pred = model.predict(test_texts)

print("\n--- Evaluation metrics ---\n")
print(classification_report(y_test, y_pred))
print("\n--- Confusion matrix ---\n")
print(confusion_matrix(y_test, y_pred))