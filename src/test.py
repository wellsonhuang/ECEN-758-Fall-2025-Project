import os
import pandas as pd
from joblib import load
from train_tfidf import build_text_column
import gdown
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# --- SETUP PATHS ---
ROOT = os.path.dirname(__file__)
DATA_DIR = os.path.join(ROOT, "..", "data")
MODEL_DIR = os.path.join(ROOT, "..", "models")
PLOTS_DIR = os.path.join(ROOT, "..", "plots")

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)

model_path = os.path.join(MODEL_DIR, "linearsvc_final.joblib")
GDRIVE_URL = "https://drive.google.com/uc?id=1sKB1KjB2gCAJtwa-VyQD9QeC3wguPa6c"

if not os.path.exists(model_path):
    print("Model not found")
    gdown.download(GDRIVE_URL, model_path, quiet=False)
    print("Download")

# --- LOADING THE LinearSVC Model ---
model = load(model_path)

test_path = os.path.join(DATA_DIR, "sogou_test.csv")
test_df = pd.read_csv(test_path)

# --- EVALUATING ON TEST DATA ---
test_texts = build_text_column(test_df, duplicate_title=True, max_len=3000)
y_test = test_df["label"].astype(int).values
y_pred = model.predict(test_texts)

print("\n--- Evaluation Metrics ---\n")
report = classification_report(y_test, y_pred)
print(report)

print("\n--- Generating Confusion Matrix ---\n")
cm = confusion_matrix(y_test, y_pred)
# --- PRINT CONFUSION MATRIX ---
print(cm)

# --- CONFUSION MATRIX (PLOT) ---
plt.figure(figsize=(10, 7))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix')

# --- SAVING THE CONFUSION MATRIX PLOT ---
cm_path = os.path.join(PLOTS_DIR, "confusion_matrix.png")
plt.savefig(cm_path)
print(f"Confusion Matrix Plot saved to: {cm_path}")
plt.close()