import streamlit as st
import pandas as pd
import joblib
import os

# Page config setup
st.set_page_config(
    page_title="Indonesian E-Commerce Sentiment Analyzer",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ----------------------------------------------------------------------

# Cache resource loading
@st.cache_resource
def load_ml_pipeline():
    vectorizer_path = "models\sentiment_vectorizer.pkl"
    model_path = "models\sentiment_linearsvc_model.pkl"

    if not os.path.exists(vectorizer_path) or not os.path.exists(model_path):
        st.error("Required model file(s) are missing.")
        return None, None
    
    vectorizer = joblib.load(vectorizer_path)
    model = joblib.load(model_path)
    return vectorizer, model

# Invoking cache loader (on first run success, the models will be stored in RAM instead of disk. On every rerun, its just gonna retrieve the model from RAM)
vectorizer, model = load_ml_pipeline()

# ----------------------------------------------------------------------

# Setting up our main page
st.title("Indonesian E-Commerce Sentiment Analyzer")
st.caption("Dashboard Page")

# shows that model were loaded successfully (again)
if vectorizer and model:
    st.success("Vectorizer and Classification models successfully loaded")
else:
    st.warning("Failed to load required models, application may not function properly")

# ----------------------------------------------------------------------