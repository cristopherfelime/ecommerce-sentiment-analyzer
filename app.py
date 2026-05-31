import streamlit as st
import numpy as np
import pandas as pd
import joblib
import os
import plotly.express as px

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
    except Exception as e:
        st.toast(f"Failed to load slang dictionary: {e}", icon="❌")
        st.session_state["slang_dict"] = {}
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
if "sampled_5_rowids" not in st.session_state:
    st.session_state["sampled_5_rowids"] = None
if "recent_uploaded_batch" not in st.session_state:
    st.session_state["recent_uploaded_batch"] = []
if "recent_temp_table_db" not in st.session_state:
    st.session_state["recent_temp_table_db"] = None
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

tab_single_review, tab_batch_review, tab_database_bi = st.tabs([
    "Single Review Evaluator",
    "Batch Review Analysis",
    "Database Business Intelligence & Insights"
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
    st.subheader("Batch Review Analysis")
    st.write("A .csv file is expected. A thorough analysis will be performed and business insights will be given.")

    uploaded_file = st.file_uploader(
        label="Upload .csv file",
        type=["csv"],
        help="Expected features: review_text, review_date, review_id, product_name, product_category, product_variant, product_price, product_url, product_id, rating, sold_count, shop_id, sentiment_label"
    )
    
    if uploaded_file is not None:
        st.info(f"File {uploaded_file.name} has been successfully uploaded")

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

            # mapping final_sentiment to original sentiment_label if not exist already
            if "sentiment_label" not in processed_df.columns:
                map_dict = {
                    "Positive": 2,
                    "Neutral": 1,
                    "Negative": 0,
                }
                processed_df["sentiment_label"] = processed_df["final_sentiment"].map(map_dict).fillna(value=-1).astype(int) # apparently pandas forces column type to be float if there are missing values, so we change it back to int just in case
        
            # getting file id and checking if it was uploaded before
            df_id = f"{uploaded_file.name}_{uploaded_file.size}" # using file name and size, we can create a unique id for each file uploaded
            if df_id in st.session_state["recent_uploaded_batch"]:
                st.warning(f"Note: A similar dataset ({df_id}) has already been appended previously, be mindful before appending this one to the database")
                alr_appended = True
            else:
                alr_appended = False

            # MIGHT REMOVE IF GONE UNUSED
            # using the same df_id as a determiner whether a new temp table will be created or not for aggregation
            if df_id == st.session_state["recent_temp_table_db"]: # for every rerun, it will check if the dataset uploaded if the same or not, if yes then no creating new table, if no then drop table and create one
                # if the user changes the threshold, it should reflect in temp_upload as well
                conn, c = data_ingestion.initialize_db()
                try:
                    db_df = pd.read_sql_query(f"""
                        SELECT final_sentiment
                        FROM temp_upload;
                    """, conn)
                    if not processed_df["final_sentiment"].equals(db_df["final_sentiment"]):
                        st.toast("Model Controls changes detected, updating final sentiment in the current uploaded dataset..", icon="ℹ️")
                        c.execute(f"""
                            DROP TABLE IF EXISTS temp_upload;
                        """)
                        conn.commit()
                        data_ingestion.ingest_data(conn, df=processed_df, table_name="temp_upload") # pandas to_sql() automatically creates table if not exist
                except Exception as e:
                    st.toast(f"An error occurred: {e}", icon="❌")
                finally:
                    conn.close()
            else:
                st.session_state["recent_temp_table_db"] = df_id
                st.toast("New/Different dataset from current session detected, refreshing database..", icon="ℹ️")
                conn, c = data_ingestion.initialize_db()
                try:
                    c.execute(f"""
                        DROP TABLE IF EXISTS temp_upload;
                    """)
                    conn.commit()
                    data_ingestion.ingest_data(conn, df=processed_df, table_name="temp_upload") # pandas to_sql() automatically creates table if not exist
                except Exception as e:
                    st.toast(f"An error occurred: {e}", icon="❌")
                finally:
                    conn.close()
            
            

            # drop duplicate rows from uploaded csv option
            if st.button("Drop Duplicated Rows"):
                if processed_df.duplicated().any():
                    total_dupes = processed_df.duplicated().sum()
                    processed_df = processed_df.drop_duplicates(keep="first")
                    st.toast(f"A total of {total_dupes}" + " rows have been dropped from the dataframe", icon="⚠️") # no toast_queue needed since we're not using st.rerun
                else:
                    st.toast("No duplicated rows found", icon="ℹ️")

            # database appending logic
            drop_dupe_append = st.checkbox("Append without Duplicates", value=True)
            if st.button("Append to Database", type="primary", disabled=alr_appended): # placeholder for now
                if drop_dupe_append:
                    processed_df = processed_df.drop_duplicates(keep="first")
                st.session_state["recent_uploaded_batch"].append(df_id)
                if alr_appended:
                    st.toast("Similar dataset already appended previously", icon="⚠️")
                else:
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
                        st.session_state["toast_queue"] = (f"{len(processed_df_list)} review(s) successfully appended to the database", "✅")
                    except Exception as e:
                        st.session_state["toast_queue"] = (f"Failed to append data to database: {e}", "❌")
                    finally:
                        conn.close()
                st.rerun() # for the button to be disabled right after clicking

# ---

            #  analysis on uploaded file
            st.divider()
            st.subheader("File Upload Analysis")
            
            top_col_1, top_col_2, top_col_3 = st.columns([1, 1, 5])
            with top_col_1:
                st.metric(
                    label="Total Uploaded Reviews",
                    value=len(processed_df)
                )
            with top_col_2:
                uncertain_prop = (len(processed_df[
                        (processed_df["final_sentiment"] != "Positive") &
                        (processed_df["final_sentiment"] != "Negative") &
                        (processed_df["final_sentiment"] != "Neutral")
                    ]) / len(processed_df)) if len(processed_df) > 0 else 0 # avoiding zerodivisionerror
                st.metric(
                    label="Uncertainty Proportion",
                    value=f"{uncertain_prop*100:.2f}%"
                )
            with top_col_3:
                st.metric(
                    label="Uncertain Amount",
                    value=len(processed_df[
                        (processed_df["final_sentiment"] != "Positive") &
                        (processed_df["final_sentiment"] != "Negative") &
                        (processed_df["final_sentiment"] != "Neutral")
                    ])
                )

            bot_col_1, bot_col_2 = st.columns(2)
            with bot_col_1: # trying out plotly
                st.subheader("Sentiment Distribution")
                labels_visual = ["Positive", "Neutral", "Negative", "Uncertain"]
                positive_val_visual = len(processed_df[processed_df["final_sentiment"] == "Positive"])
                neutral_val_visual = len(processed_df[processed_df["final_sentiment"] == "Neutral"])
                negative_val_visual = len(processed_df[processed_df["final_sentiment"] == "Negative"])
                uncertain_val_visual = len(processed_df[
                    (processed_df["final_sentiment"] != "Positive") &
                    (processed_df["final_sentiment"] != "Neutral") &
                    (processed_df["final_sentiment"] != "Negative")
                ])
                values_visual = [positive_val_visual, neutral_val_visual, negative_val_visual, uncertain_val_visual]
                df_visual = { # plotly expects a tidy data, which what dictionary is
                    "sentiment_category" : labels_visual,
                    "num_reviews" : values_visual
                }
                px_chart_batch_1 = px.bar(
                    data_frame=df_visual,
                    x="sentiment_category",
                    y="num_reviews"
                )
                px_chart_batch_1.update_xaxes(
                    title_text="<b>Sentiment Categories</b>",
                    showgrid=False,
                    tickfont=dict(family="Arial Black", size=12) # arial black is basically bold
                )
                px_chart_batch_1.update_yaxes(
                    title_text="<b>Number of Reviews</b>",
                    showgrid=True,
                    tickfont=dict(family="Arial Black", size=12)
                )
                px_chart_batch_1.update_traces(
                    texttemplate="<b>%{y}</b>", # without tidy data, this would not work
                    textposition="auto",
                    textfont=dict(size=24),
                    marker_color=[ # uses hex color codes
                        "#00ff00", # positive - green
                        "#ffff00", # neutral - yellow
                        "#ff0000", # negative - red
                        "#808080" # uncertain - gray
                    ]
                )
                st.plotly_chart(px_chart_batch_1, use_container_width=True)
            with bot_col_2:
                st.subheader("Top 10 Reviews with Highest Negative Sentiment Probability")
                st.dataframe(processed_df[["review_text", "predict_negative", "product_category", "product_price"]].sort_values(by="predict_negative", ascending=False).head(10), use_container_width=True)
                

            st.divider()

# ---

            # previewing processed dataframe
            with st.expander("Preview Processed Dataframe", expanded=True):
                if st.button("Sample 5 rows"):
                    st.session_state["sampled_5_rowids"] = processed_df[["review_text", "final_sentiment", "product_category", "product_price"]].sample(5).index
                if st.session_state["sampled_5_rowids"] is not None:
                    st.dataframe(processed_df.loc[st.session_state["sampled_5_rowids"]][["review_text", "final_sentiment", "product_category", "product_price"]], use_container_width=True)
                st.dataframe(processed_df[["review_text", "final_sentiment", "product_category", "product_price"]].head(10), use_container_width=True)
            
            
# -----------------------------

# tab - database business insights