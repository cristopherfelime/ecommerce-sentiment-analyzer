# python packages
import streamlit as st
import os
import joblib
import pandas as pd

# software pipelines and dedicated tools
import pipeline
import database_utils

# Cache resource loading
@st.cache_resource
def load_ml_pipeline():
    pipeline_path = os.path.join("models", "pipeline_package.joblib")

    if not os.path.exists(pipeline_path):
        st.error("Required pipeline package is missing.")
        return None
    
    pipeline = joblib.load(pipeline_path)
    return pipeline

# cache data loading (slang dictionary)
@st.cache_data(show_spinner="Loading slang dictionary..", ttl="1h", max_entries=5)
def load_slang_dict():
    slang_df, error_status_slang = database_utils.pull_db(columns=["slang", "formal"], table_name="lexicons")
    slang_dict = dict(zip(slang_df["slang"], slang_df["formal"]))

    return slang_dict, error_status_slang

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
def process_batch_file(uploaded_df, _pipeline_ml, _slang_dict): # not used for single review, as its just gonna run once when its submitted, and it will retrieve values from the session state anyway
    # underscore is just to tell streamlit to "trust me bro", streamlit wont look up hashes for cached data (complex models like these dont tend to be able to be hashed regardless)
    # it is to tell streamlit to not input the specific argument passed in its cache (hash table), so it just uses whatver it gives you without looking in its hash table (should increase performance as well)
    # uploaded_df may change depending on what the user uploads, so we will hash it. make sure streamlit knows whether uploaded_df changed or not
    # pipeline and slang_dict most likely don't change, so we don't need to hash them. tell streamlit to trust us that it wont change, it makes it faster since streamlit wont check whether theyre the same every time the app is reran

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

    # vectors = _vectorizer.transform(uploaded_df["cleaned_review"])
    # proba_matrix = _model.predict_proba(vectors) # should return matrix size of (n, 3)
    proba_matrix = _pipeline_ml.predict_proba(uploaded_df["cleaned_review"]) # since we moved to using pipeline, here we are just directly calling predict_proba method from the pipeline

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


if __name__ == "__main__":
    pass