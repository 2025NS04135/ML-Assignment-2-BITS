import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score, 
    recall_score, f1_score, matthews_corrcoef, 
    confusion_matrix, classification_report
)

# Set page layout and title
st.set_page_config(
    page_title="Breast Cancer ML Classification App",
    page_icon="🔬",
    layout="wide"
)

st.title("🔬 Breast Cancer Classification Model Evaluator")
st.markdown("### Interactive Model Evaluation Dashboard | BITS Pilani Machine Learning Assignment 2")
st.markdown("---")

# Sidebar Controls
st.sidebar.header("🕹️ Configuration Controls")

# Requirement (a): Dataset upload option (CSV)
uploaded_file = st.sidebar.file_uploader(
    "Upload Test Data (CSV)", 
    type=["csv"],
    help="Upload test_data.csv generated from training notebook."
)

# Requirement (b): Model selection dropdown
model_choice = st.sidebar.selectbox(
    "Select ML Classification Model",
    (
        "Logistic Regression",
        "Decision Tree",
        "KNN",
        "Naive Bayes",
        "Support Vector Machine"
    )
)

# Model Artifact Mapping
model_files = {
    "Logistic Regression": "model/logistic_regression.joblib",
    "Decision Tree": "model/decision_tree.joblib",
    "KNN": "model/knn.joblib",
    "Naive Bayes": "model/naive_bayes.joblib",
    "Support Vector Machine": "model/svm.joblib"
}

# Cache models to improve runtime performance
@st.cache_resource
def load_scaler_and_model(name):
    scaler = joblib.load("model/scaler.joblib")
    model = joblib.load(model_files[name])
    return scaler, model

# Main Content Execution
if uploaded_file is not None:
    # Read CSV
    df_test = pd.read_csv(uploaded_file)
    st.sidebar.success(f"Loaded {df_test.shape[0]} rows and {df_test.shape[1]} columns.")
    
    # Separate Features and Target
    if "target" in df_test.columns:
        X_test = df_test.drop(columns=["target"])
        y_test = df_test["target"]
    else:
        X_test = df_test.iloc[:, :-1]
        y_test = df_test.iloc[:, -1]

    # Load selected model and scaler
    scaler, model = load_scaler_and_model(model_choice)

    # Scale inputs for distance/gradient based algorithms
    if model_choice in ["Logistic Regression", "KNN", "Support Vector Machine"]:
        X_eval = scaler.transform(X_test)
    else:
        X_eval = X_test.values

    # Generate Predictions
    y_pred = model.predict(X_eval)
    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_eval)[:, 1]
    else:
        y_proba = y_pred

    # Compute Evaluation Metrics
    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    mcc = matthews_corrcoef(y_test, y_pred)

    # Requirement (c): Display of evaluation metrics
    st.subheader(f"📌 Evaluation Metrics: {model_choice}")
    
    col1, col2, col3, col4, col5, col6 = st.columns(6)
    col1.metric("Accuracy", f"{acc:.4f}")
    col2.metric("AUC Score", f"{auc:.4f}")
    col3.metric("Precision", f"{prec:.4f}")
    col4.metric("Recall", f"{rec:.4f}")
    col5.metric("F1 Score", f"{f1:.4f}")
    col6.metric("MCC Score", f"{mcc:.4f}")

    st.markdown("---")
    
    # Requirement (d): Confusion matrix and classification report
    col_left, col_right = st.columns([1, 1])

    with col_left:
        st.subheader("🎯 Confusion Matrix")
        cm = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots(figsize=(5, 4))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax,
                    xticklabels=["Malignant (0)", "Benign (1)"],
                    yticklabels=["Malignant (0)", "Benign (1)"])
        plt.xlabel("Predicted Label")
        plt.ylabel("True Label")
        st.pyplot(fig)

    with col_right:
        st.subheader("📋 Classification Report")
        report = classification_report(
            y_test, y_pred, 
            target_names=["Malignant (0)", "Benign (1)"]
        )
        st.code(report, language="text")

else:
    st.info("👈 Upload `test_data.csv` using the sidebar to view evaluation metrics, confusion matrix, and classification report.")