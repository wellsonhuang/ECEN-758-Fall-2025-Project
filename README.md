# SogouNews Text Classification (ECEN 758 - Group 6)

## 📌 Project Overview
This repository contains the implementation for the **ECEN 758 (Fall 2025)** Group 6 project. The goal of this project is to classify news articles from the **SogouNews dataset** into various topic categories (Sports, Finance, Entertainment, Technology).

We implemented and evaluated several text categorization models to benchmark performance, including:
* **Mutinomial Naive Bayes**
* **KNN**
* **LinearSVC**
* **1D CNN**
* **TextCNN**

  ## 📂 Project Structure
```text
ECEN-758-Fall-2025-Project/
├── data/                      # Training and testing datasets
│   ├── sogou_test.csv
│   └── sogou_train.csv
├── models/                    # Serialized trained models (.joblib)
│   ├── linearsvc_final.joblib # Best performing LinearSVC model
│   └── ...                    # Baseline models (KNN, Naive Bayes)
├── plots/                     # Generated visualizations
│   ├── class_distribution.png
│   ├── confusion_matrix.png
│   ├── dim_reduction.png
│   └── text_length_statistics.png
├── src/                       # Source code
│   ├── Data_Mining_project.ipynb # 1D CNN and TextCNN notebook
│   ├── model_tuning.py        # Hyperparameter tuning script for LinearSVC
│   ├── preprocess_sogou.py    # Data cleaning and tokenization
│   ├── test.py                # Evaluation on test data
│   ├── tfidf_tuning.py        # TF-IDF vectorizer optimization
│   └── train_tfidf.py         # Main training pipeline
├── .gitignore                 # Files to exclude from git
├── requirements.txt           # Python dependencies
└── README.md                  # Project documentation
```

Follow the steps below to run and evaluate the final trained model.

## Step 1: Clone the repository
```bash
git clone https://github.com/wellsonhuang/ECEN-758-Fall-2025-Project.git
cd ECEN-758-Fall-2025-Project
```

## **Step 2 — Install requirements**
pip install -r requirements.txt

## **Step 3 — import dataset and preprocess**
python src/preprocess_sogou.py

## **Step 4 — run model evaluation**
python src/test.py
