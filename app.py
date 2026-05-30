import streamlit as st
import numpy as np
import pandas as pd
import joblib
import os

import pipeline
import data_ingestion

# Page config setup
st.set_page_config(
    page_title="Indonesian E-Commerce Sentiment Analyzer",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ----------------------------------------------------------------------

# global variables
if "slang_dict" not in st.session_state:
    conn, _ = data_ingestion.initialize_db()
    try:
        slang_df = pd.read_sql_query("""
            SELECT slang, formal
            FROM lexicons;
        """, conn)
        st.session_state["slang_dict"] = dict(zip(slang_df["slang"], slang_df["formal"]))
    finally:
        conn.close()
if "cleaned_review" not in st.session_state:
    st.session_state["cleaned_review"] = None
if "review_vector" not in st.session_state:
    st.session_state["review_vector"] = None
if "prediction_proba" not in st.session_state:
    st.session_state["prediction_proba"] = None
if "sentiment_label" not in st.session_state:
    st.session_state["sentiment_label"] = None
if "sampled_5_rows" not in st.session_state:
    st.session_state["sampled_5_rows"] = None
if "recent_uploaded_batch" not in st.session_state:
    st.session_state["recent_uploaded_batch"] = []
if "toast_queue" not in st.session_state:
    st.session_state["toast_queue"] = None

# for toast messages to keep after rerun (like a mini function)
def run_queue_toast():
    if st.session_state["toast_queue"] is not None:
        msg, icon = st.session_state["toast_queue"]
        st.toast(msg, icon=icon)
        st.session_state["toast_queue"] = None # immediate reset after use
run_queue_toast()

# Cache resource loading
@st.cache_resource
def load_ml_pipeline():
    vectorizer_path = "models/sentiment_vectorizer.pkl"
    model_path = "models/sentiment_lr_model.pkl"

    if not os.path.exists(vectorizer_path) or not os.path.exists(model_path):
        st.error("Required model file(s) are missing.")
        return None, None
    
    vectorizer = joblib.load(vectorizer_path)
    model = joblib.load(model_path)
    return vectorizer, model

# invoking cache loader (on first run success, the models will be stored in RAM instead of disk. On every rerun, its just gonna retrieve the model from RAM)
vectorizer, model = load_ml_pipeline()

# Cache data loading
@st.cache_data(show_spinner="Processing batch file..", ttl="1h", max_entries=10) # ttl and max_entries prevents memory usage from getting too high yes
def process_batch_file(df, _vectorizer, _model, _slang_dict): # not used for single review, as its just gonna run once when its submitted, and it will retrieve values from the session state anyway
    # underscore is just to tell streamlit to "trust me bro", streamlit wont look up hashes for cached data (complex models like these dont tend to be able to be hashed regardless)
    # it is to tell streamlit to not input the specific argument passed in its cache (hash table), so it just uses whatver it gives you without looking in its hash table

    df = pd.read_csv(df)

    required_columns = {
        "review_text", "review_date", "review_id", 
        "product_name", "product_category", "product_variant", 
        "product_price", "product_url", "product_id", 
        "rating", "sold_count", "shop_id", "sentiment_label"
    }
    missing_cols = [i for i in required_columns if i not in df.columns]
    if missing_cols:
        return None, f"Missing required columns: {', '.join(missing_cols)}"
    
    df["cleaned_review"] = df["review_text"].apply(
        lambda x : pipeline.clean_review_text(text=x, data_dict=_slang_dict)
    )

    vectors = _vectorizer.transform(df["cleaned_review"])
    proba_matrix = _model.predict_proba(vectors) # should return matrix size of (n, 3)

    df["predict_negative"] = proba_matrix[:, 0]
    df["predict_neutral"] = proba_matrix[:, 1]
    df["predict_positive"] = proba_matrix[:, 2]

    return df, "Success"




# ----------------------------------------------------------------------

# title and caption
st.title("Indonesian E-Commerce Sentiment Analyzer")
st.caption("hi guys")

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
    }) # might add the hyperparameters here

    if st.button("Clear App Cache"):
        st.cache_data.clear()

# ----------------------------------------------------------------------

# tabs

tab_single_review, tab_batch_review = st.tabs([
    "Single Review Evaluator",
    "Batch Review Analysis"
])

# -----------------------------

# tab - single review

with tab_single_review:
    leader_index = None # to prevent error in the detailed prediction result json part
    st.subheader("Single Review Classification")

    layout_type = "A"
    user_review = st.text_input("Input your user review: ")
    if st.button("Submit Review"):
        if user_review is not None and user_review != "":
            st.write("Review was submitted.")
            layout_type = "B"
        else:
            st.write("Please input a valid review!")
            layout_type = "A"

    if user_review is not None and user_review != "":
        st.session_state["cleaned_review"] = pipeline.clean_review_text(
            text=user_review,
            data_dict=st.session_state["slang_dict"]
        )
        st.session_state["review_vector"] = vectorizer.transform([st.session_state["cleaned_review"]]) # transform() function from vectorizer expects a list of strings, not just a string
        st.session_state["prediction_proba"] = model.predict_proba(st.session_state["review_vector"])
        probs = st.session_state["prediction_proba"][0] # predict_proba always returns 2D array, ts to ensure that argmax to work on it
        leader_index = np.argmax(probs) # np.argmax takes the index of the max value
        leader_proba = probs[leader_index]
        if leader_proba >= prediction_threshold:
            st.session_state["sentiment_label"] = "Positive" if leader_index == 2 else "Neutral" if leader_index == 1 else "Negative" if leader_index == 0 else "Error"
        else:
            st.session_state["sentiment_label"] = f"Uncertain, Confidence: {leader_proba*100:.2f}% that it is " + (
                "Positive" if leader_index == 2 else
                "Neutral" if leader_index == 1 else
                "Negative" if leader_index == 0 else
                "Error"
            )
        

    container = st.container()
    with container:
        if layout_type == "A":
            st.divider()
            st.subheader("Result")
            st.metric(
                label="Sentiment Prediction:",
                value="Waiting for submission...",
            )
        
        elif layout_type == "B":
            st.divider()
            st.subheader("Result")
            if user_review is not None or user_review != "":
                st.metric(
                    label="Sentiment Prediction",
                    value=st.session_state["sentiment_label"],
                )
            else:
                st.metric(
                    label="Sentiment Prediction",
                    value="Error!",
                )

            with st.expander(label="Detailed Prediction Result", expanded=False):
                if st.session_state["prediction_proba"] is not None:
                    st.write(f"Negative: {st.session_state["prediction_proba"][0][0]*100:.2f}%")
                    st.write(f"Neutral: {st.session_state["prediction_proba"][0][1]*100:.2f}%")
                    st.write(f"Positive: {st.session_state["prediction_proba"][0][2]*100:.2f}%")
                    
                    st.json({
                        "Original Review" : user_review,
                        "Cleaned Review" : st.session_state["cleaned_review"],
                        "Sentiment Probabilities": {
                            "Negative" : st.session_state["prediction_proba"][0][0],
                            "Neutral" : st.session_state["prediction_proba"][0][1],
                            "Positive" : st.session_state["prediction_proba"][0][2],
                        },
                        "Predicted Sentiment" : "Positive" if leader_index == 2 else "Neutral" if leader_index == 1 else "Negative" if leader_index == 0 else "Error",
                    })

# -----------------------------

# tab - batch reviews

with tab_batch_review:
    st.subheader("Batch Review Analysis and Business Intelligence")
    st.write("A .csv file is expected. A thorough analysis will be performed and business insights will be given.")

    uploaded_file = st.file_uploader(
        label="Upload .csv file",
        type=["csv"],
        help="Expected features: review_text, review_date, review_id, product_name, product_category, product_variant, product_price, product_url, product_id, rating, sold_count, shop_id, sentiment_label"
    )
    
    if uploaded_file is not None:
        st.info(f"File {uploaded_file.name} has been successfully uploaded. Running back end operations..")

        result_df, status = process_batch_file(uploaded_file, vectorizer, model, st.session_state["slang_dict"])
        processed_df = result_df.copy()
        if status != "Success":
            st.error(status)
        elif processed_df is not None:
            st.success(f"Successfully processed {len(processed_df)} customer reviews")

            # similar to above, getting index and highest probability for each review
            # with this being here instead of the cache_data function, when user moves the threshold slider, it will update instantly due to the caching
            proba_columns = processed_df[["predict_negative", "predict_neutral", "predict_positive"]].values
            leader_indices = np.argmax(proba_columns, axis=1)
            leader_probs = np.max(proba_columns, axis=1)

            # getting final label based on threshold
            labeled_predictions = []
            for idx, prob in zip(leader_indices, leader_probs):
                if prob >= prediction_threshold:
                    labeled_predictions.append(
                        "Positive" if idx == 2 else
                        "Neutral" if idx == 1 else
                        "Negative" if idx == 0 else
                        "Error"
                    )
                else:
                    labeled_predictions.append(
                        f"Uncertain ({prob*100:.2f}%  " + (
                            "Positive)" if idx == 2 else
                            "Neutral)" if idx == 1 else
                            "Negative)" if idx == 0 else
                            "Error)"
                        )
                    )
            label_df = pd.DataFrame(labeled_predictions, columns=["final_sentiment"])
            processed_df = processed_df.join(label_df)
        
            # getting file id and checking if it was uploaded before
            df_id = f"{uploaded_file.name}_{uploaded_file.size}" # using file name and size, we can create a unique id for each file uploaded
            if df_id in st.session_state["recent_uploaded_batch"]:
                st.warning(f"Note: A similar dataset ({df_id}) has already been appended previously, be mindful before appending this one to the database")
                alr_appended = True
            else:
                alr_appended = False

            # database appending logic
            if st.button("Append to Database", type="primary", disabled=alr_appended): # placeholder for now
                st.session_state["recent_uploaded_batch"].append(df_id)
                if alr_appended:
                    st.session_state["toast_queue"] = ("Similar dataset already appended previously", "⚠️")
                else:
                    # mapping final_sentiment to original sentiment_label if not exist already
                    if "sentiment_label" not in processed_df.columns:
                        map_dict = {
                            "Positive": 2,
                            "Neutral": 1,
                            "Negative": 0,
                        }
                        processed_df["sentiment_label"] = processed_df["final_sentiment"].map(map_dict).fillna(value=-1).astype(int) # apparently pandas forces column type to be float if there are missing values, so we change it back to int just in case
                    # inserting into database
                    conn, c = data_ingestion.initialize_db()
                    try: # another way to prevent SQL injection
                        processed_df_list = processed_df[[
                            "review_text",
                            "review_date",
                            "review_id",
                            "product_name",
                            "product_category",
                            "product_variant",
                            "product_price",
                            "product_url",
                            "product_id",
                            "rating",
                            "sold_count",
                            "shop_id",
                            "sentiment_label"
                        ]].to_numpy().tolist()
                        c.executemany("""
                                INSERT INTO reviews (
                                    review_text,
                                    review_date,
                                    review_id,
                                    product_name,
                                    product_category,
                                    product_variant,
                                    product_price,
                                    product_url,
                                    product_id,
                                    rating,
                                    sold_count,
                                    shop_id,
                                    sentiment_label
                                )
                                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                            """,
                            processed_df_list
                            )
                        conn.commit()
                    finally:
                        conn.close()
                    st.session_state["toast_queue"] = ("Dataset successfully appended to the database", "✅")
                st.rerun() # for the button to be disabled right after clicking

            # previewing processed dataframe
            with st.expander("Preview Processed Dataframe", expanded=True):
                if st.button("Sample 5 rows"):
                    st.session_state["sampled_5_rows"] = processed_df[["review_text", "final_sentiment", "product_category", "product_price"]].sample(5)
                if st.session_state["sampled_5_rows"] is not None:
                    st.dataframe(st.session_state["sampled_5_rows"], use_container_width=True)
                st.dataframe(processed_df[["review_text", "final_sentiment", "product_category", "product_price"]].head(10), use_container_width=True)

                st.divider()
                st.subheader("Product Category Analysis")
            
            
