# SogouNews Text Classification (ECEN 758 - Group 6)

## 📌 Project Overview
This repository contains the implementation for the **ECEN 758 (Fall 2025)** Group 6 project. The goal of this project is to classify news articles from the **SogouNews dataset** into various topic categories (Sports, Finance, Entertainment, Technology). We executed and assessed various text categorization models to determine their effectiveness.

We implemented and evaluated several text categorization models to benchmark performance, including:
* **Mutinomial Naive Bayes**
* **KNN**
* **LinearSVC**
* **1D CNN**
* **TextCNN**

Follow the steps below to run and evaluate the final trained model.

## **Step 1 - Clone the repository

git clone [https://github.com/wellsonhuang/ECEN-758-Fall-2025-Project.git](https://github.com/wellsonhuang/ECEN-758-Fall-2025-Project.git)
cd ECEN-758-Fall-2025-Projectbash

## **Step 2 — Install requirements**
pip install -r requirements.txt

## **Step 3 — import dataset and preprocess**
python src/preprocess_sogou.py

## **Step 4 — run model evaluation**
python src/test.py
