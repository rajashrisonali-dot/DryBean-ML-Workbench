
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
    classification_report
)

st.set_page_config(
    page_title="Dry Bean Classification",
    page_icon="🌱",
    layout="wide"
)

st.title("Dry Bean Multi-Class Classification")
st.write("Machine Learning Assignment 2 - Five ML Models")

FEATURES = [
    "Area", "Perimeter", "MajorAxisLength",
    "MinorAxisLength", "AspectRation", "Eccentricity",
    "ConvexArea", "EquivDiameter", "Extent", "Solidity",
    "roundness", "Compactness", "ShapeFactor1",
    "ShapeFactor2", "ShapeFactor3", "ShapeFactor4"
]

TARGET = "Class"

MODEL_FILES = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "K-Nearest Neighbors": "k_nearest_neighbors.joblib",
    "Gaussian Naive Bayes": "gaussian_naive_bayes.joblib",
    "Support Vector Machine": "support_vector_machine.joblib"
}

@st.cache_resource
def load_models():
    loaded = {}
    for name, filename in MODEL_FILES.items():
        path = os.path.join("model", filename)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing model: {path}")
        loaded[name] = joblib.load(path)
    return loaded

try:
    models = load_models()
    st.success("All five trained models loaded successfully!")
except Exception as error:
    st.error(str(error))
    st.stop()

selected_model = st.selectbox(
    "Select Machine Learning Model",
    list(MODEL_FILES.keys())
)

uploaded_file = st.file_uploader(
    "Upload Test CSV File",
    type=["csv"]
)

if uploaded_file is not None:
    try:
        data = pd.read_csv(uploaded_file)

        st.subheader("Dataset Preview")
        st.dataframe(data.head())

        missing = [c for c in FEATURES if c not in data.columns]

        if missing:
            st.error(f"Missing features: {missing}")
            st.stop()

        X_test = data[FEATURES]
        model = models[selected_model]
        y_pred = model.predict(X_test)

        st.subheader("Prediction Results")
        result = data.copy()
        result["Predicted_Class"] = y_pred

        if TARGET in data.columns:
            y_test = data[TARGET]

            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(
                y_test, y_pred, average="weighted", zero_division=0
            )
            recall = recall_score(
                y_test, y_pred, average="weighted", zero_division=0
            )
            f1 = f1_score(
                y_test, y_pred, average="weighted", zero_division=0
            )
            mcc = matthews_corrcoef(y_test, y_pred)

            try:
                scores = (
                    model.predict_proba(X_test)
                    if hasattr(model, "predict_proba")
                    else model.decision_function(X_test)
                )
                auc = roc_auc_score(
                    y_test, scores,
                    multi_class="ovr",
                    average="weighted",
                    labels=model.classes_
                )
            except Exception:
                auc = None

            st.subheader("Evaluation Metrics")

            c1, c2, c3 = st.columns(3)
            c1.metric("Accuracy", f"{accuracy:.4f}")
            c2.metric("AUC", f"{auc:.4f}" if auc is not None else "N/A")
            c3.metric("Precision", f"{precision:.4f}")

            c4, c5, c6 = st.columns(3)
            c4.metric("Recall", f"{recall:.4f}")
            c5.metric("F1 Score", f"{f1:.4f}")
            c6.metric("MCC", f"{mcc:.4f}")

            st.subheader("Confusion Matrix")
            labels = sorted(y_test.unique())
            cm = confusion_matrix(y_test, y_pred, labels=labels)

            fig, ax = plt.subplots(figsize=(9, 6))
            sns.heatmap(
                cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=labels, yticklabels=labels, ax=ax
            )
            ax.set_xlabel("Predicted Class")
            ax.set_ylabel("Actual Class")
            st.pyplot(fig)
            plt.close(fig)

            st.subheader("Classification Report")
            report = classification_report(
                y_test, y_pred, output_dict=True, zero_division=0
            )
            st.dataframe(pd.DataFrame(report).transpose())

            result["Prediction_Status"] = np.where(
                y_test == y_pred, "Correct", "Incorrect"
            )

        st.dataframe(result)

        st.download_button(
            "Download Prediction Results",
            result.to_csv(index=False).encode("utf-8"),
            "dry_bean_predictions.csv",
            "text/csv"
        )

    except Exception as error:
        st.error(f"Error: {error}")
else:
    st.info("Please upload test_data.csv to begin.")

st.sidebar.title("About Project")
st.sidebar.write("Dataset: UCI Dry Bean")
st.sidebar.write("Features: 16")
st.sidebar.write("Classes: 7")
st.sidebar.write("Classification Models: 5")
st.sidebar.write("Evaluation Metrics: 6")
