# __init__.py will mark the "tabs" directory as importable package

import sys
import os

# only go up a folder if we are currently inside the tabs folder
if os.path.basename(os.getcwd()) == "tabs":
    os.chdir("..")

# basically makes the project root folder discoverable for this script, its for importing pipeline.py and other stuff
project_root = os.getcwd()
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# -----

import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px

import app_utils
import database_utils

def tab_content(pipeline_ml, prediction_threshold):
    st.subheader("Batch Review Analysis")
    st.write("A .csv file is expected. A thorough analysis will be performed and business insights will be given.")

    uploaded_file = st.file_uploader(
        label="Upload .csv file",
        type=["csv"],
        help="Expected features: review_text, review_date, review_id, product_name, product_category, product_variant, product_price, product_url, product_id, rating, sold_count, shop_id, sentiment_label"
        )
    st.session_state["uploaded_file_session"] = uploaded_file
    
    if uploaded_file is not None:
        st.info(f"File {uploaded_file.name} has been successfully uploaded")
        
        slang_dict, error_status_slang = app_utils.load_slang_dict()
        if error_status_slang:
            print(f"Failed to load slang dictionary from database - error_status: {error_status_slang}")
            st.toast(error_status_slang, icon="❌")
            result_df, status = app_utils.process_batch_file(uploaded_file, pipeline_ml, None)
        else:
            result_df, status = app_utils.process_batch_file(uploaded_file, pipeline_ml, slang_dict)
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
            


if __name__ == "__main__":
    pass