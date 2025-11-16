import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.calibration import CalibratedClassifierCV
from joblib import dump
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import TruncatedSVD
from sklearn.manifold import TSNE



ROOT = os.path.dirname(__file__)
DATA_DIR = os.path.join(ROOT, "..", "data")
MODEL_DIR = os.path.join(ROOT, "..", "models")
PLOTS_DIR = os.path.join(ROOT, "..", "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

max_len = 3000

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


def plot_class_distribution(df: pd.DataFrame, save_path=None):
    """Plotting the class distribution."""
    print("Analyzing class distribution...")
    plt.figure(figsize=(10, 6))
    # Assuming 'label' is an integer.
    sns.countplot(x='label', data=df)
    plt.title('Class Distribution (Training Set)')
    plt.xlabel('Category Label')
    plt.ylabel('Number of Articles')
    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=300)
    plt.show()


def plot_text_length_statistics(texts: pd.Series, save_path=None):
    """ Plotting the distribution of article lengths."""
    print("Analyzing text lengths...")
    lengths = texts.str.len()
    print("Text Length Descriptive Statistics:")
    print(lengths.describe())  # Prints mean, median, min, max, etc.

    plt.figure(figsize=(10, 6))
    sns.histplot(lengths, bins=50, kde=True)
    plt.title('Distribution of Article Lengths (Characters)')
    plt.xlabel('Length')
    plt.ylabel('Frequency')
    plt.xlim(0, max_len)  # Use the max_len from build_text_column
    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=300)
    plt.show()

def show_class_examples(df: pd.DataFrame, num_examples_per_class : int = 2):
    print("Showing examples for each class")

    labels = df['label'].unique()
    labels.sort()

    for label in labels:
        print("\n" + "=" * 50)
        print(f"Examples for label :{label}")
        print("=" * 50)

        #Getting random sample of examples for this label
        sample_df = df[df['label']==label].sample(n=num_examples_per_class, random_state=42)

        for i, (index, row) in enumerate(sample_df.iterrows()):
            print(f"\n ----- Example {i+1} (Index: {index}) ------")
            title = normalize_text(row['title'])
            print(f" Title : {title}")

            content = normalize_text(row['content'])
            print(f" Content : {content}")

def plot_dimensionality_reduction(texts : pd.Series, labels : np.ndarray, save_path=None):
    print("Performing Dimensionality Reduction")

    # ---- Calculating sample_size ----
    total_size = len(texts)
    sample_size = min(10000, total_size)
    # --Calculating proportion -----
    train_proportion = sample_size / total_size

    sample_texts, sample_texts_unused, sample_labels, sample_labels_unused = train_test_split(
        texts, labels, train_size=train_proportion, stratify=labels, random_state=42)

    # -----Creating TF_IDF vectorizer ---
    tfidf = TfidfVectorizer(
        analyzer="char",
        ngram_range=(1,2),
        min_df=3,
        max_df=0.9,
        sublinear_tf=True,
        lowercase=False,
        dtype=np.float32,
    )
    print("Fitting TF-IDF for dimensionality reduction")
    features = tfidf.fit_transform(sample_texts)

    reducer = TruncatedSVD(n_components=2, random_state=42)

    print("Reducing dimensions")
    reduced_features = reducer.fit_transform(features)

    # ----- Plotting the PCA ----
    plot_df = pd.DataFrame({
        'x' : reduced_features[:, 0],
        'y' : reduced_features[:, 1],
        'label' : sample_labels
    })

    plt.figure(figsize=(8, 6))
    sns.scatterplot(data = plot_df,
                    x = 'x',
                    y = 'y',
                    hue='label',
                    palette = sns.color_palette("hls", n_colors=len(plot_df['label'].unique())),
                    legend = "full",
                    alpha = 0.5
    )
    plt.title("Dimensionality Reduction of TF-IDF Features")
    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=300)
    plt.show()

def main():
    train_path = os.path.join(DATA_DIR, "sogou_train.csv")
    test_path = os.path.join(DATA_DIR, "sogou_test.csv")

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    train_texts = build_text_column(train_df, duplicate_title=True, max_len=3000)
    test_texts = build_text_column(test_df, duplicate_title=True, max_len=3000)

    y_train_full = train_df["label"].astype(int).values
    y_test = test_df["label"].astype(int).values

    RUN_EDA = True
    if RUN_EDA:
        print("\n" + "=" * 40)
        print("Running Exploratory Data Analysis (EDA)")
        print("=" * 40 + "\n")

        # (2a) Descriptive statistics: Class Distribution
        plot_class_distribution(train_df, save_path = os.path.join(PLOTS_DIR, "class_distribution.png"))

        # (2a) Descriptive statistics: Text Length
        plot_text_length_statistics(train_texts, save_path = os.path.join(PLOTS_DIR, "text_length_statistics.png"))

        # (2b) Data Visualization: Show examples from classes
        show_class_examples(train_df, num_examples_per_class=2)

        # (2b) Data visualization: Dimensionality Reduction
        plot_dimensionality_reduction(train_texts, y_train_full, save_path = os.path.join(PLOTS_DIR, "dim_reduction.png"))

    X_train, X_val, y_train, y_val = train_test_split(
        train_texts,
        y_train_full,
        test_size=0.1,
        stratify=y_train_full,
        random_state=42,
    )

    # TF-IDF + Linear SVM pipeline and TF-IDF + Multinomial Naive Bayes pipeline
    base_clf = {
        "NaiveBayes": MultinomialNB(),
        "KNN": KNeighborsClassifier(n_neighbors=7, metric="cosine", n_jobs=-1),
        "LinearSVC": LinearSVC(C=1.0)

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