# python packages
import streamlit as st
import numpy as np
import pandas as pd
import joblib
import os
import plotly.express as px

# software pipelines and dedicated tools
import pipeline
import data_ingestion
import database_utils

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
vectorizer, model = load_ml_pipeline() # this can stay up here since will be used for two tabs

# cache data loading (slang dictionary)
@st.cache_data(show_spinner="Loading slang dictionary..", ttl="1h", max_entries=5)
def load_slang_dict():
    slang_df, error_status_slang = database_utils.pull_db(columns=["slang", "formal"], table_name="lexicons")
    slang_dict = dict(zip(slang_df["slang"], slang_df["formal"]))

    return slang_dict, error_status_slang
slang_dict, error_status_slang = load_slang_dict()
# slang_dict no need to be a global variable cuz caching makes loading them faster
# changing from global variable with if guard to local variable is actually better, especially for deployment. every time a new different user opens the app, it will run the database connection. but if we use caching, streamlit only needs to run the database connection once for getting the slang_dict. it takes it from the server RAM instead rather than running the database connection every time

# cache data loading (reviews dataset (cleaned))
@st.cache_data(show_spinner="Loading dataset from database...", ttl="2h", max_entries=10)
def load_dataset_from_db():
    reviews_df, error_status_reviews = database_utils.pull_db(table_name="reviews")
    processed_reviews_df = reviews_df.dropna(subset=[
            "product_id", 
            "product_name", 
            "product_category", 
            "product_price", 
            "rating", 
            "sold_count", 
            "sentiment_label", 
            "review_text", 
            "review_date"
        ])
    processed_reviews_df = processed_reviews_df.drop_duplicates(keep="first") # subset is wider than in train_model.py, that's why no duplicates are detected. for ML training, this can be fatal. but for BI, they're fine.

    return reviews_df, processed_reviews_df, error_status_reviews

# cache data loading (database reviews preprocessing)
@st.cache_data(show_spinner="Processing batch file..", ttl="1h", max_entries=10) # ttl and max_entries prevents memory usage from getting too high yes
def process_batch_file(uploaded_df, _vectorizer, _model, _slang_dict): # not used for single review, as its just gonna run once when its submitted, and it will retrieve values from the session state anyway
    # underscore is just to tell streamlit to "trust me bro", streamlit wont look up hashes for cached data (complex models like these dont tend to be able to be hashed regardless)
    # it is to tell streamlit to not input the specific argument passed in its cache (hash table), so it just uses whatver it gives you without looking in its hash table (should increase performance as well)
    # uploaded_df may change depending on what the user uploads, so we will hash it. make sure streamlit knows whether uploaded_df changed or not
    # vectorizer, model, and slang_dict most likely don't change, so we don't need to hash them. tell streamlit to trust us that it wont change, it makes it faster since streamlit wont check whether theyre the same every time the app is reran

    uploaded_df = pd.read_csv(uploaded_df)

    required_columns = {
        "review_text", "review_date", "review_id", 
        "product_name", "product_category", "product_variant", 
        "product_price", "product_url", "product_id", 
        "rating", "sold_count", "shop_id", "sentiment_label"
    }
    missing_cols = [i for i in required_columns if i not in uploaded_df.columns]
    if missing_cols:
        return None, f"Missing required columns: {', '.join(missing_cols)}"
    
    uploaded_df["cleaned_review"] = uploaded_df["review_text"].apply(
        lambda x : pipeline.clean_review_text(text=x, data_dict=_slang_dict)
    )

    vectors = _vectorizer.transform(uploaded_df["cleaned_review"])
    proba_matrix = _model.predict_proba(vectors) # should return matrix size of (n, 3)

    uploaded_df["predict_negative"] = proba_matrix[:, 0]
    uploaded_df["predict_neutral"] = proba_matrix[:, 1]
    uploaded_df["predict_positive"] = proba_matrix[:, 2]

    return uploaded_df, "Success"

# cache data loading (avg vws across sold count calculation)
@st.cache_data(show_spinner="Calculating average volume-weighted sentiment across total sold count..", ttl="2h", max_entries=10)
def calc_avg_vws(reviews_df):
    # avg vws across total sold = sum of volume weighted sentiment across sold per product, something like sum of (average_sentiment_score_i * sold_i) / sum of sold_i

    # getting the product_id and sold_count column only and removing duplicates
    prod_sold_df = reviews_df[["product_id", "sold_count"]].copy() # subsetting is just a view, changing anything there will change the actual reviews_df as well

    sentiment_label_map = {
        "positive" : 1,
        "neutral" : 0,
        "negative" : -1
    }
    # getting product_id again, and the average net of sentiment_label
    # learned hard way: dont use sum lol i accidentally overinflated the final score
    sentiment_score = reviews_df[["product_id", "sentiment_label"]].copy()
    sentiment_score["sentiment_label"] = sentiment_score["sentiment_label"].map(sentiment_label_map).fillna(0) # fillna is for any uncertain and unknown value
    sentiment_score = sentiment_score.groupby("product_id")["sentiment_label"].mean().reset_index() # average the sentiment_score for each product_id, reset_index so that we can use product_id

    # merging them with inner join now
    prod_sold_score_df = pd.merge(prod_sold_df, sentiment_score, on="product_id", how="inner")

    # finally getting the avg vws across total sold
    avg_vws = (prod_sold_score_df["sentiment_label"] * prod_sold_score_df["sold_count"]).sum() / prod_sold_score_df["sold_count"].sum()

    return avg_vws
    


# cache data loading (time series cumsum sentiment label vs date)
@st.cache_data(show_spinner="Processing database reviews for time series visualization..", ttl="2h", max_entries=10)
def process_ts_cumsum_df(reviews_df): # note: we hash this bcs we expect reviews_df to change, after all the recaching will rerun every time refresh database pull button is clicked
    valid_sentiments = ["positive", "neutral", "negative"]

    time_series_df = reviews_df.copy()
    time_series_df = time_series_df[["review_date", "sentiment_label"]] # subset to only review_date and sentiment_label
    time_series_df.loc[~time_series_df["sentiment_label"].isin(valid_sentiments), "sentiment_label"] = "uncertain" # uncertains are appended with their percentages to the database, this locates all sentiment_label values that are not in valid_sentiments and change then to "uncertain" to normalize them
    time_series_df["review_date"] = pd.to_datetime(time_series_df["review_date"], format="mixed", errors="coerce") # convert to datetime data type to ensure compatibility (mixed will allow it to handle different formats)
    time_series_df = time_series_df.dropna(subset=["review_date"]) # remove rows that has NaT (not a time) as result from error="coerce" above
    ts_df_counts = time_series_df.groupby(["review_date", "sentiment_label"]).size().reset_index(name="review_count").sort_values(["review_date", "sentiment_label"]) # group by date and sentiment label and count the number of reviews (size is used over count because no third column, so kinda redundant here)
    ts_df_counts["cumulative_count"] = ts_df_counts.groupby("sentiment_label")["review_count"].cumsum() # compute cumulative sum of review_count for each sentiment_label

    return ts_df_counts

# cache data loading (vis 1: bar chart of true negative rate)
@st.cache_data(show_spinner="Processing database reviews for bar chart true negative rate visualization..", ttl="2h", max_entries=10)
def process_true_neg_df(reviews_df):
    # basically the following: (sum of all negative reviews in a cat / sum of all reviews in a cat)

    # getting product_category and sentiment_label together then subset to negative sentiment_label only
    cat_neg_reviews_df = reviews_df[["product_category", "sentiment_label"]].copy()
    cat_neg_reviews_df = cat_neg_reviews_df[cat_neg_reviews_df["sentiment_label"] == "negative"]

    # getting the total negative reviews per product category
    cat_neg_reviews_df = cat_neg_reviews_df.groupby("product_category").agg(
        total_negatives=("sentiment_label", "size")
    ).reset_index()

    # getting product_category but this time with review_text, then get their total number per category
    cat_tot_reviews_df = reviews_df[["product_category", "review_text"]].copy()
    cat_tot_reviews_df = cat_tot_reviews_df.groupby("product_category").agg(
        total_reviews=("review_text", "size")
    ).reset_index()
    
    # merging them and calculating the true negative rate for each category
    trueneg_df = pd.merge(cat_neg_reviews_df, cat_tot_reviews_df, on="product_category", how="inner")
    trueneg_df["true_negative"] = (trueneg_df["total_negatives"] / trueneg_df["total_reviews"])

    # converting to a "tidy" version (dictionary) for plotly in order for in-chart value labeling to work
    trueneg_df_tidy = trueneg_df[["product_category", "true_negative"]].to_dict(orient="list") # it needs list-format dict for the values to show up

    return trueneg_df_tidy

# cache data (vis 2: bubble chart of average rating vs total sale from each product)
@st.cache_data(show_spinner="Processing database reviews for bubble chart visualization..", ttl="2h", max_entries=10)
def process_bubble_df(reviews_df):
    # basically the following: getting the average rating and total sale count for each product

    # using groupby() immediately to get relevant columns, then returning them.
    bubble_df = reviews_df[["product_id", "product_name", "product_category", "product_price", "rating", "sold_count"]].copy()
    bubble_df = bubble_df.groupby(["product_id", "product_name", "product_category", "product_price"]).agg(
        average_rating=("rating", "mean"),
        total_sold=("sold_count", "sum")
    ).reset_index() # reset_index so that we can use product_id

    return bubble_df

# ----------------------------------------------------------------------

# title and caption
st.title("Genesis Reviewer")
st.caption("An Indonesian E-Commerce Review Sentiment Analyzer and Dashboard.")

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
    single_review.tab_content(vectorizer, model, prediction_threshold, slang_dict, error_status_slang)

# -----------------------------

# tab - batch reviews

with tabs[1]:
    st.subheader("Batch Review Analysis")
    st.write("A .csv file is expected. A thorough analysis will be performed and business insights will be given.")

    uploaded_file = st.file_uploader(
        label="Upload .csv file",
        type=["csv"],
        help="Expected features: review_text, review_date, review_id, product_name, product_category, product_variant, product_price, product_url, product_id, rating, sold_count, shop_id, sentiment_label"
    )
    
    if uploaded_file is not None:
        st.info(f"File {uploaded_file.name} has been successfully uploaded")

        slang_dict, error_status_slang = load_slang_dict()
        if error_status_slang:
            print(f"Failed to load slang dictionary from database - error_status: {error_status_slang}")
            st.toast(error_status_slang, icon="❌")
            result_df, status = process_batch_file(uploaded_file, vectorizer, model, None)
        else:
            result_df, status = process_batch_file(uploaded_file, vectorizer, model, slang_dict)
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
                        "positive" if idx == 2 else
                        "neutral" if idx == 1 else
                        "negative" if idx == 0 else
                        "error"
                    )
                else:
                    labeled_predictions.append(
                        f"uncertain ({prob*100:.2f}%  " + (
                            "positive)" if idx == 2 else
                            "neutral)" if idx == 1 else
                            "negative)" if idx == 0 else
                            "error)"
                        )
                    )
            label_df = pd.DataFrame(labeled_predictions, columns=["final_sentiment"])
            processed_df = processed_df.join(label_df)

            # mapping final_sentiment to original sentiment_label if not exist already
            if "sentiment_label" not in processed_df.columns:
                processed_df["sentiment_label"] = processed_df["final_sentiment"]
        
            # getting file id and checking if it was uploaded before
            df_id = f"{uploaded_file.name}_{uploaded_file.size}" # using file name and size, we can create a unique id for each file uploaded
            if df_id in st.session_state["recent_uploaded_batch"]:
                st.warning(f"Note: A similar dataset ({df_id}) has already been appended previously, be mindful before appending this one to the database")
                alr_appended = True
            else:
                alr_appended = False

            # this is supposed to create a temp_upload table if not exists and insert processed_df in the table, and update it by dropping and recreating the table every rerun the processed_df changes or any model control changes
            # gone unused since we just used processed_df directly for the real time uploaded csv visualization
            # using the same df_id as a determiner whether a new temp table will be created or not for aggregation
#            if df_id == st.session_state["recent_temp_table_db"]: # for every rerun, it will check if the dataset uploaded if the same or not, if yes then no creating new table, if no then drop table and create one
#                # if the user changes the threshold, it should reflect in temp_upload as well
#                conn, c = data_ingestion.initialize_db()
#                try:
#                    db_df = pd.read_sql_query(f"""
#                        SELECT final_sentiment
#                        FROM temp_upload;
#                    """, conn)
#                    if not processed_df["final_sentiment"].equals(db_df["final_sentiment"]):
#                        st.toast("Model Controls changes detected, updating final sentiment in the current uploaded dataset..", icon="ℹ️")
#                        c.execute(f"""
#                            DROP TABLE IF EXISTS temp_upload;
#                        """)
#                        conn.commit()
#                        data_ingestion.ingest_data(conn, df=processed_df, table_name="temp_upload") # pandas to_sql() automatically creates table if not exist
#                except Exception as e:
#                    st.toast(f"An error occurred: {e}", icon="❌")
#                finally:
#                    conn.close()
#            else:
#                st.session_state["recent_temp_table_db"] = df_id
#                st.toast("New/Different dataset from current session detected, refreshing database..", icon="ℹ️")
#                conn, c = data_ingestion.initialize_db()
#                try:
#                    c.execute(f"""
#                        DROP TABLE IF EXISTS temp_upload;
#                    """)
#                    conn.commit()
#                    data_ingestion.ingest_data(conn, df=processed_df, table_name="temp_upload") # pandas to_sql() automatically creates table if not exist
#                except Exception as e:
#                    st.toast(f"An error occurred: {e}", icon="❌")
#                finally:
#                    conn.close()
            
            

            # drop duplicate rows from uploaded csv option
            drop_dupe = st.checkbox("Drop Duplicated Rows", value=True)
            if drop_dupe:
                if processed_df.duplicated().any():
                    total_dupes = processed_df.duplicated().sum()
                    processed_df = processed_df.drop_duplicates(keep="first")
                    st.write(f"A total of {total_dupes}" + " row(s) have been dropped from the dataset")
                else:
                    st.write("No duplicated rows found")

            # database appending logic
            if st.button("Append to Database", type="primary", disabled=alr_appended): # placeholder for now
                st.session_state["recent_uploaded_batch"].append(df_id)
                if alr_appended:
                    st.toast("Similar dataset already appended previously", icon="⚠️")
                else:
                    # inserting into database
                    error_status = database_utils.push_db(df=processed_df, table_name="reviews")
                    if error_status:
                        st.session_state["toast_queue"] = (error_status, "❌")
                    else:
                        st.session_state["toast_queue"] = (f"{len(processed_df)} review(s) successfully appended to the database", "✅")
                
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
                        (processed_df["final_sentiment"] != "positive") &
                        (processed_df["final_sentiment"] != "negative") &
                        (processed_df["final_sentiment"] != "neutral")
                    ]) / len(processed_df)) if len(processed_df) > 0 else 0 # avoiding zerodivisionerror
                st.metric(
                    label="Uncertainty Proportion",
                    value=f"{uncertain_prop*100:.2f}%"
                )
            with top_col_3:
                st.metric(
                    label="Uncertain Amount",
                    value=len(processed_df[
                        (processed_df["final_sentiment"] != "positive") &
                        (processed_df["final_sentiment"] != "negative") &
                        (processed_df["final_sentiment"] != "neutral")
                    ])
                )

            bot_col_1, bot_col_2 = st.columns(2)
            with bot_col_1: # trying out plotly
                st.subheader("Sentiment Distribution")
                labels_visual = ["Positive", "Neutral", "Negative", "Uncertain"]
                valid_sentiments = ["positive", "neutral", "negative"]
                positive_val_visual = len(processed_df[processed_df["final_sentiment"] == "positive"])
                neutral_val_visual = len(processed_df[processed_df["final_sentiment"] == "neutral"])
                negative_val_visual = len(processed_df[processed_df["final_sentiment"] == "negative"])
                uncertain_val_visual = len(processed_df[~processed_df["final_sentiment"].isin(valid_sentiments)]) # using ~ to invert the isin boolean results, then counting. this works for uncertain since it has different percentages and stuff in its value
                values_visual = [positive_val_visual, neutral_val_visual, negative_val_visual, uncertain_val_visual]
                df_visual = { # plotly expects a tidy data, which what dictionary is
                    "sentiment_category" : labels_visual,
                    "num_reviews" : values_visual
                }
                px_chart_batch = px.bar(
                    data_frame=df_visual,
                    x="sentiment_category",
                    y="num_reviews"
                )
                px_chart_batch.update_xaxes(
                    title_text="<b>Sentiment Categories</b>",
                    showgrid=False,
                    tickfont=dict(family="Arial Black", size=12) # arial black is basically bold
                )
                px_chart_batch.update_yaxes(
                    title_text="<b>Number of Reviews</b>",
                    showgrid=True,
                    tickfont=dict(family="Arial Black", size=12)
                )
                px_chart_batch.update_traces(
                    texttemplate="<b>%{y}</b>", # without tidy data, this would not work
                    textposition="auto",
                    textfont=dict(size=24),
                    marker_color=[ # uses hex color codes, bcs also weirdly enough, using values of sentiment_category for x and color makes the bars very thin in visualization
                        "#00ff00", # positive - green
                        "#ffff00", # neutral - yellow
                        "#ff0000", # negative - red
                        "#808080" # uncertain - gray
                    ]
                )
                st.plotly_chart(px_chart_batch, use_container_width=True)
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
                    try:
                        st.dataframe(processed_df.loc[st.session_state["sampled_5_rowids"]][["review_text", "final_sentiment", "product_category", "product_price"]], use_container_width=True)
                    except KeyError: # got lucky finding this early lol
                        st.session_state["sampled_5_rowids"] = None
                        st.toast("Some sampled row(s) no longer exists, resampling another 5..")
                        st.session_state["sampled_5_rowids"] = processed_df[["review_text", "final_sentiment", "product_category", "product_price"]].sample(5).index
                        st.dataframe(processed_df.loc[st.session_state["sampled_5_rowids"]][["review_text", "final_sentiment", "product_category", "product_price"]], use_container_width=True)
                st.dataframe(processed_df[["review_text", "final_sentiment", "product_category", "product_price"]].head(10), use_container_width=True)
            
            
# -----------------------------

# tab - database business insights

with tabs[2]:
    st.subheader("Dataset Overview")
    st.write("Discover insights from the database within the visualizations present in this page.")
    st.write("Note: Duplicated entries from the reviews dataset pulled from the database is automatically omitted.")

    reviews_df, processed_reviews_df, error_status_reviews = load_dataset_from_db()
    if error_status_reviews:
        print(f"Failed to load dataset from database - error_status: {error_status_reviews}")
        st.toast(error_status_reviews, icon="❌")
    avg_vws = calc_avg_vws(processed_reviews_df)
    ts_df_counts = process_ts_cumsum_df(processed_reviews_df)
    trueneg_df_tidy = process_true_neg_df(processed_reviews_df)
    bubble_df = process_bubble_df(processed_reviews_df)
        
    if st.button("Refresh Database Pull"):
        load_dataset_from_db.clear()
        process_ts_cumsum_df.clear()
        process_true_neg_df.clear()
        process_bubble_df.clear()
        st.session_state["toast_queue"] = (f"Refreshing Database pull..", "🔄️")
        st.rerun()
    
    with st.expander("Preview Reviews Dataset (NaN and duplicates omitted)", expanded=False):
        if st.button("Sample 5 rows (from db)"):
            st.session_state["sampled_5_rowids_db"] = processed_reviews_df.sample(5).index
        if st.session_state["sampled_5_rowids_db"] is not None:
            st.dataframe(processed_reviews_df.loc[st.session_state["sampled_5_rowids_db"]], use_container_width=True)
        st.dataframe(processed_reviews_df.head(10), use_container_width=True)

    st.divider()
    st.subheader("Key Performance Indicators")

# ---

    # Key Performance Indicators
    KPI_1, KPI_2, KPI_3 = st.columns([2, 4, 2])
    with KPI_1: # total entries in the reviews table
        st.metric(
            label="Total Reviews (Raw)",
            value=len(reviews_df)
        )
    with KPI_2: # average volume-weighted sentiment across all reviews (check caching part above for formula)
        if (avg_vws - st.session_state["old_avg_vws"]) > 0.1: # dominant positive sentiment
            vws_color = "green"
        elif (avg_vws - st.session_state["old_avg_vws"]) < -0.1: # dominant negative sentiment
            vws_color = "red"
        else: # -0.10 <= avg_vws <= 0.10
            vws_color = "gray"
        st.metric(
            label="Average Volume-Weighted Sentiment Across Product Sold Count*",
            value=f"{avg_vws:.2f}",
            delta=0 if st.session_state["old_avg_vws"] is None else f"{avg_vws - st.session_state["old_avg_vws"]:.2f}",
            delta_color=vws_color
        )
        st.session_state["old_avg_vws"] = avg_vws
    with KPI_3: # total preprocessed reviews
        st.metric(
            label="Total (Preprocessed) Reviews",
            value=len(processed_reviews_df)
        )

    st.caption("*: Ranges from -1 to 1, negative number means negativity is more dominant and vice versa. Any uncertain and unknown sentiments were rated as neutral.")

# ---

    st.divider()
    st.subheader("Data Visualizations")

# ---

    # Time Series Analysis of cumulative sentiment labels over time
    px_ts = px.line(
        data_frame=ts_df_counts,
        x="review_date",
        y="cumulative_count",
        color="sentiment_label",
        color_discrete_map={
            "positive" : "#00ff00", # green
            "neutral" : "#ffff00", # yellow
            "negative" : "#ff0000", # red
            "uncertain" : "#808080" # gray
        },
        title="<b>Cumulative Sentiment Label Over Time</b>"
    )
    px_ts.update_xaxes(
        title_text="<b>Review Date</b>",
        showgrid=False,
        tickfont=dict(family="Arial Black", size=12)
    )
    px_ts.update_yaxes(
        title_text="<b>Cumulative Review Count</b>",
        showgrid=True,
        tickfont=dict(family="Arial Black", size=12)
    )
    px_ts.update_layout(
        legend=dict(
            title_text="Sentiment Label",
            orientation="h", # horizontal
            y=1.1, # top of the chart
            x=0.15, # left of the chart
            xanchor="center"
        ),
        title_font=dict(family="Arial Black", size=24),
        title_x=0.5,
        title_xanchor="center"
    )
    st.plotly_chart(px_ts, use_container_width=True)

# ---

    # More Visualizations
    vis_1, vis_2 = st.columns([3, 2])
    with vis_1: # Bar Chart of True Negativity rate of each category (check caching part above for formula)
        px_trueneg = px.bar(
            data_frame=trueneg_df_tidy,
            x="product_category",
            y="true_negative",
            title="<b>True Negativity Rate of Each Product Category</b>"
        )
        px_trueneg.update_xaxes(
            title_text="<b>Product Category</b>",
            tickfont=dict(family="Arial Black", size=12),
        )
        px_trueneg.update_yaxes(
            title_text="<b>True Negativity Rate</b>",
            tickfont=dict(family="Arial Black", size=12)
        )
        px_trueneg.update_layout(
            title_font=dict(family="Arial Black", size=16),
            title_x=0.5,
            title_xanchor="center",
        )
        px_trueneg.update_traces(
            texttemplate="<b>%{y:.2f}%</b>",
            textposition="auto",
            textfont=dict(size=12),
            marker_color=px.colors.qualitative.D3 # premade color sets brought to you by plotly themselves wow (great for auto assigning and also avoiding thin bar issue by not using the same value for x and color when creating the bar instance)
        )
        st.plotly_chart(px_trueneg, use_container_width=True)
    with vis_2: # Bubble Chart of product price over average rating per product (also sold count as bubble size)
        px_bubble = px.scatter(
            data_frame=bubble_df,
            x="product_price",
            y="average_rating",
            size="total_sold",
            color="product_category",
            title="<b>Avg Rating vs Product Price</b>",
        )
        px_bubble.update_xaxes(
            title_text="<b>Product Price</b>",
            tickfont=dict(family="Arial Black", size=12),
            showgrid=True,
        )
        px_bubble.update_yaxes(
            title_text="<b>Average Rating</b>",
            tickfont=dict(family="Arial Black", size=12),
            showgrid=True,
        )
        px_bubble.update_layout(
            legend=dict(
                title_text="Product Category",
                orientation="h",
                y=0.4,
                x=0.8,
                xanchor="center",
                bgcolor="rgba(0, 0, 0, 0.2)"
            ),
            title_font=dict(family="Arial Black", size=16),
            title_x=0.5,
            title_xanchor="center",
        )
        st.plotly_chart(px_bubble, use_container_width=True)
    st.divider()

# ----------------------------------------------------------------------
