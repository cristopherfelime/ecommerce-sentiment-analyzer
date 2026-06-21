# python packages
import streamlit as st

# software pipelines and dedicated tools
import app_utils

# streamlit tabs
from tabs import single_review, batch_analysis, database_bi

# Page config setup
st.set_page_config(
    page_title="Genesis",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ----------------------------------------------------------------------

# global variables
# ts for when variables are needed on next rerun
if "active_tab" not in st.session_state:
    st.session_state["active_tab"] = "Single Review Evaluator" # to preserve tab state on rerun
if "cleaned_review" not in st.session_state:
    st.session_state["cleaned_review"] = None # will store the cleaned version of the review submitted by user
if "review_vector" not in st.session_state:
    st.session_state["review_vector"] = None # will store the vectorised version of the review submitted by user
if "prediction_proba" not in st.session_state:
    st.session_state["prediction_proba"] = None # will store the prediction probability of the review submitted by user
if "sentiment_label" not in st.session_state:
    st.session_state["sentiment_label"] = None # will store the predicted sentiment label of the review submitted by user
if "sampled_5_rowids" not in st.session_state:
    st.session_state["sampled_5_rowids"] = None # will store the 5 sampled rowids (indices) from the dataset for single review evaluation
if "sampled_5_rowids_db" not in st.session_state:
    st.session_state["sampled_5_rowids_db"] = None # will store the 5 sampled rowids (indices) from the database for business intelligence tab, to preserve them
if "recent_uploaded_batch" not in st.session_state:
    st.session_state["recent_uploaded_batch"] = [] # will store the recently uploaded batch file, by using list and append
if "recent_temp_table_db" not in st.session_state:
    st.session_state["recent_temp_table_db"] = None # will store the recent temporary table from the database, by using string
if "old_avg_vws" not in st.session_state:
    st.session_state["old_avg_vws"] = 0 # will store the previous average volume-weighted sentiment for the delta thing
if "toast_queue" not in st.session_state:
    st.session_state["toast_queue"] = None # will store the toast message and icon, by using tuple

# for toast messages to keep after rerun (like a mini function)
def run_queue_toast():
    if st.session_state["toast_queue"] is not None:
        msg, icon = st.session_state["toast_queue"]
        st.toast(msg, icon=icon)
        st.session_state["toast_queue"] = None # immediate reset after use
run_queue_toast()

# invoking cache loader (on first run success, the models will be stored in RAM instead of disk. On every rerun, its just gonna retrieve the model from RAM)
vectorizer, model = app_utils.load_ml_pipeline() # this can stay up here since will be used for two tabs

# ----------------------------------------------------------------------

# title and caption
st.title("Genesis Reviewer")
st.caption("An E-Commerce Review Sentiment Analyzer and Dashboard.")

# ----------------------------------------------------------------------

# sidebar

with st.sidebar:
    st.header("Model Controls")
    st.caption("Options to fine-tune the ML models")

    # prediction threshold slider
    prediction_threshold = st.slider(
        label="Prediction Threshold", 
        min_value=0.0, 
        max_value=1.0, 
        value=0.5, 
        step=0.05,
        help="Adjust to change the prediction threshold of the ML model (applies to batch reviews as well, so keep in mind before uploading)"
    )

    st.divider()
    st.markdown("### Machine Learning Model Metadata")
    st.json({
        "Algorithm" : "Logistic Regression",
        "Text Vectorization" : "Term Frequency-Inverse Document Frequency",
        "Target Language" : "Informal Indonesian"
    }, expanded=False) # might add the hyperparameters here

    if st.button("Clear App Cache"):
        st.cache_data.clear()

# ----------------------------------------------------------------------

# tabs

tabs = st.tabs([
    "Single Review Evaluator",
    "Batch Review Analysis",
    "Database Business Intelligence & Insights"
], key="tab_storage") # using a key storage is to preserve the tab state upon reruns, crazy but it actually links to the active_tab global variable

# -----------------------------

# tab - single review

with tabs[0]:
    single_review.tab_content(vectorizer, model, prediction_threshold)

# -----------------------------

# tab - batch reviews

with tabs[1]:    
    batch_analysis.tab_content(vectorizer, model, prediction_threshold)

# -----------------------------

# tab - database business insights

with tabs[2]:
    database_bi.tab_content()

# ----------------------------------------------------------------------
