import streamlit as st
import os
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt

from eda import eda
from model.test import model_evaluate

st.title("Model Evaluator - Tarun Singhal - 2025aa05171")
st.markdown(
    "Load a test dataset and evaluate an already trained model. "
    "Select dataset via upload or filesystem path, choose the model, and click Evaluate."
)

# Model registry: map display name -> pkl path
MODEL_REGISTRY = {
    "Logistic Regression": "model/logistic_regression/logistic_regression_model.pkl",
}

# 1. Dataset input: upload or path
st.subheader("Test dataset")
uploaded_file = st.file_uploader("Upload CSV test file. Get it from gitrepo data/ObesityDataSet_test.csv", type=["csv"])
dataset_path = st.text_input("Or enter CSV filepath (use this value: data/ObesityDataSet_test.csv)")

dataset = None
if uploaded_file is not None:
    try:
        dataset = eda.Dataset(uploaded_file)
        st.success("Loaded dataset from uploaded file")
    except Exception as e:
        st.error(f"Could not read uploaded CSV: {e}")
elif dataset_path:
    if os.path.exists(dataset_path):
        try:
            dataset = eda.Dataset(dataset_path)
            st.success(f"Loaded dataset from `{dataset_path}`")
        except Exception as e:
            st.error(f"Failed to read CSV at `{dataset_path}`: {e}")
    else:
        st.info(f"File not found at `{dataset_path}`")

if dataset is not None:
    st.write("Preview of test data (first 5 rows):")
    st.dataframe(dataset.head())

# 2. Model selection from dropdown
st.subheader("Model selection")
model_name = st.selectbox("Select model to evaluate", list(MODEL_REGISTRY.keys()))
model_path = MODEL_REGISTRY.get(model_name)

st.markdown(f"Using model file: `{model_path}`")

target_col = dataset.get_target_column

# Evaluate button
if st.button("Evaluate"):
    if dataset is None:
        st.error("No dataset provided. Upload or specify a CSV path first.")
    else:
        if not os.path.exists(model_path):
            st.error(f"Model file not found at `{model_path}`. Make sure the path is correct.")
        else:
            model_evaluator = model_evaluate(model_path, dataset)
            if model_evaluator.is_valid():
                results, confusion_matrix = model_evaluator.evaluate()
                st.subheader("Metrics")
                st.json(results)
                disp = ConfusionMatrixDisplay(confusion_matrix=confusion_matrix)
                fig, ax = plt.subplots(figsize=(6, 6))
                disp.plot(ax=ax)
                st.subheader("Confusion Matrix")
                st.pyplot(fig)
            else:
                st.error("Model evaluation failed.")