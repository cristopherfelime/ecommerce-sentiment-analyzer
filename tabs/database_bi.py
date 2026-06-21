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

def tab_content():
    st.subheader("Dataset Overview")
    st.write("Discover insights from the database within the visualizations present in this page.")
    st.write("Note: Duplicated entries from the reviews dataset pulled from the database is automatically omitted.")

    reviews_df, processed_reviews_df, error_status_reviews = app_utils.load_dataset_from_db()
    if error_status_reviews:
        print(f"Failed to load dataset from database - error_status: {error_status_reviews}")
        st.toast(error_status_reviews, icon="❌")
    avg_vws = app_utils.calc_avg_vws(processed_reviews_df)
    ts_df_counts = app_utils.process_ts_cumsum_df(processed_reviews_df)
    trueneg_df_tidy = app_utils.process_true_neg_df(processed_reviews_df)
    bubble_df = app_utils.process_bubble_df(processed_reviews_df)
        
    if st.button("Refresh Database Pull"):
        app_utils.load_dataset_from_db.clear()
        app_utils.calc_avg_vws.clear()
        app_utils.process_ts_cumsum_df.clear()
        app_utils.process_true_neg_df.clear()
        app_utils.process_bubble_df.clear()
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

if __name__ == "__main__":
    pass