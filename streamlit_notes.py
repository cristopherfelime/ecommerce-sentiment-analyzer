import streamlit as st
import pandas as pd
import pipeline
import joblib
import data_ingestion
import train_model

st.title("Indonesian Review Sentiment Analyzer")
st.set_page_config(layout="wide") # This will make us not waste any horizontal space it seems like

# caching demo
# @st.cache_resource is a decorator to cache data into system RAM instead of loading them from disk
# since this is significantly faster, it is great to use for loading models
@st.cache_resource
def load_production_files():
    lr_model = joblib.load("models/sentiment_lr_model.pkl")
    linearsvc_model = joblib.load("models/sentiment_linearsvc_model.pkl")
    vectorizer = joblib.load("models/sentiment_vectorizer.pkl")

    return lr_model, linearsvc_model, vectorizer
    
# @st.cache_data is the same thing

# Their main difference is the type of objects to cache
# @st.cache_resource: resource-heavy objects that are not easily serializable (ML models, database connections, etc.). If you modify the cached object, it may affect future calls
# @st.cache_data: data that can be serialized (dictionaries, lists, pandas dataframes, numpy arrays, JSON, etc.). It always returns a new, independent copy of the object, so modifying it won't affect future calls

# local variable (resets when the app reruns, which is basically whenever the user refresh or interact with any widget)
counter = 0

# session variables, a temporary storage that lasts for as long as the app is running
# they prevent reloading the same data/models multiple times
if "global_counter" not in st.session_state:
    st.session_state["global_counter"] = 0
if "slang_dict" not in st.session_state:
    conn, c = data_ingestion.initialize_db()
    slang_df = pd.read_sql_query("""
        SELECT slang, formal
        FROM lexicons;
    """, conn)
    st.session_state["slang_dict"] = dict(zip(slang_df["slang"], slang_df["formal"]))
if "cleaned_review" not in st.session_state:
    st.session_state["cleaned_review"] = None
if "review_vector" not in st.session_state:
    st.session_state["review_vector"] = None
if "prediction_lr" not in st.session_state:
    st.session_state["prediction_lr"] = None
if "prediction_svc" not in st.session_state:
    st.session_state["prediction_svc"] = None

lr_model, linearsvc_model, vectorizer = load_production_files()

layout_type = "A"

user_review = st.text_input("Input your user review: ")
if st.button("Submit Review"):
    counter += 1
    st.session_state["global_counter"] += 1
    st.write("Review was submitted.")
    layout_type = "B"

    if user_review:
        st.session_state["cleaned_review"] = pipeline.clean_review_text(
            text=user_review,
            data_dict=st.session_state["slang_dict"]
        )
        st.session_state["review_vector"] = vectorizer.transform([st.session_state["cleaned_review"]]) # transform() function from vectorizer expects a list of strings, not just a string
        st.session_state["prediction_lr"] = lr_model.predict(st.session_state["review_vector"])
        st.session_state["prediction_svc"] = linearsvc_model.predict(st.session_state["review_vector"])

container = st.container()
with container:
    if layout_type == "A":
        with st.sidebar:
            st.subheader("Information")
            st.write(f"Counter: {counter}, Global counter: {st.session_state['global_counter']}")
            st.write(f"Review: {"Waiting..."}")
            st.write(f"Cleaned Review: {"Waiting..."}")
            st.write(f"Prediction (Logistic Regression): {"Waiting..."}")
            st.write(f"Prediction (Linear SVC): {"Waiting..."}")
        
        st.subheader("Model predictions")
        col1, col2 = st.columns(2)
        with col1:
            st.metric(
                label="Logistic Regression Prediction",
                value="Waiting...",
            )
        with col2:
            st.metric(
                label="Linear SVC Prediction",
                value="Waiting...",
            )

    elif layout_type == "B":
        with st.sidebar:
            if user_review:
                st.subheader("Information")
                st.write(f"Counter: {counter}, Global counter: {st.session_state['global_counter']}")
                st.write(f"Review: {user_review}")
                st.write(f"Cleaned Review: {st.session_state['cleaned_review']}")
                st.write(f"Prediction (Logistic Regression): {st.session_state['prediction_lr'][0]}")
                st.write(f"Prediction (Linear SVC): {st.session_state['prediction_svc'][0]}")
            else:
                st.subheader("Information")
                st.write(f"Counter: {counter}, Global counter: {st.session_state['global_counter']}")
                st.write(f"Review: {"Error!"}")
                st.write(f"Cleaned Review: {"Error!"}")
                st.write(f"Prediction (Logistic Regression): {"Error!"}")
                st.write(f"Prediction (Linear SVC): {"Error!"}")

        st.subheader("Model predictions")
        col1, col2 = st.columns(2)
        with col1:
            if user_review:
                st.metric(
                    label="Logistic Regression Prediction",
                    value="Positive" if st.session_state["prediction_lr"][0] == 2 else "Neutral" if st.session_state["prediction_lr"][0] == 1 else "Negative" if st.session_state["prediction_lr"][0] == 0 else "Error occurred",
                )
            else:
                st.metric(
                    label="Logistic Regression Prediction",
                    value="Error!",
                )
        with col2:
            if user_review:
                st.metric(
                    label="Linear SVC Prediction",
                    value="Positive" if st.session_state["prediction_svc"][0] == 2 else "Neutral" if st.session_state["prediction_svc"][0] == 1 else "Negative" if st.session_state["prediction_svc"][0] == 0 else "Error occurred",
                )
            else:
                st.metric(
                    label="Linear SVC Prediction",
                    value="Error!"
                )




# import streamlit as st
# import pandas as pd
# import joblib

# import pipeline
# import data_ingestion
# import train_model

# st.title("Indonesian Review Sentiment Analyzer")
# st.set_page_config(layout="wide")

# with st.sidebar:
#     st.header("Control Panel")
#     st.info("Configure your NLP pipeline analysis below")

#     confidence_threshold = st.slider(
#         "Confidence Threshold",
#         min_value=0.5,
#         max_value=1.0,
#         value=0.75,
#         step=0.05,
#         help="Minimum confidence score required for a prediction to be positive"
#     )

#     st.divider()
#     st.caption("Model Ver 1.0 | Cristopher Jeremie Felime")

# tabs

