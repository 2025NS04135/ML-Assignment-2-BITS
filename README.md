# Breast Cancer Classification & Evaluation App

## a. Problem Statement
The objective of this project is to build, evaluate, and benchmark five distinct machine learning classification algorithms to predict whether a breast tissue mass sample is **Malignant** (`0`) or **Benign** (`1`). The solution is integrated into an interactive Streamlit web dashboard for real-time model evaluation.

## b. Dataset Description
* **Dataset**: Breast Cancer Wisconsin (Diagnostic) Dataset
* **Source**: UCI Machine Learning Repository / Scikit-Learn
* **Instances**: 569 samples
* **Features**: 30 continuous numerical attributes (including radius, texture, perimeter, area, smoothness, compactness, concavity, symmetry, and fractal dimension)
* **Target**: Binary classification (`0` = Malignant, `1` = Benign)

## c. GitHub Repository Link
`https://github.com/2025NS04135/ML-Assignment-2-BITS`

## d. Models Used & Comparison Table

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** | 0.9825 | 0.9954 | 0.9861 | 0.9861 | 0.9861 | 0.9623 |
| **Decision Tree** | 0.9123 | 0.9157 | 0.9559 | 0.9028 | 0.9286 | 0.8174 |
| **KNN** | 0.9561 | 0.9788 | 0.9589 | 0.9722 | 0.9655 | 0.9054 |
| **Naive Bayes** | 0.9386 | 0.9878 | 0.9452 | 0.9583 | 0.9517 | 0.8676 |
| **Support Vector Machine** | 0.9825 | 0.9950 | 0.9861 | 0.9861 | 0.9861 | 0.9623 |

## e. Observations on Model Performance

| ML Model Name | Observation about model performance |
| :--- | :--- |
| **Logistic Regression** | Performed exceptionally well. Standardized linear boundaries cleanly separate the high-dimensional feature space without overfitting, yielding peak F1 (0.9861) and MCC (0.9623). |
| **Decision Tree** | Underperformed relative to kernel and distance models due to rigid axis-aligned splits on continuous features, resulting in a lower recall of 0.9028. |
| **KNN** | Performed strongly after feature scaling, achieving 0.9561 accuracy and high recall (0.9722). |
| **Naive Bayes** | Delivered solid overall AUC (0.9878) due to continuous Gaussian feature distributions, though precision slightly lagged linear models. |
| **Support Vector Machine** | Tied for the top performance (Accuracy: 0.9825, MCC: 0.9623) due to effective hyper-plane separation via RBF kernel mapping on scaled attributes. |
| **Overall Winner for your dataset?** | **Logistic Regression & Support Vector Machine (Tie)**. Both models achieved identical peak Accuracy (98.25%), F1 Score (0.9861), and MCC (0.9623). Logistic Regression slightly leads in overall AUC (0.9954). |