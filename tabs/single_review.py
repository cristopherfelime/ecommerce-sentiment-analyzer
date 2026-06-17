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

import pipeline

def tab_content(vectorizer, model, prediction_threshold, slang_dict, error_status_slang):
    leader_index = None # to prevent error in the detailed prediction result json part
    st.subheader("Single Review Classification")

    if "layout_type" not in st.session_state:
        st.session_state["layout_type"] = "A" # modularizing this tab into a function on a separate file causes some ghost layout_type issue, so i had to make this a global variable and stay at the last type (A or B) it was in instead of resetting back to A (not sure if it actually fixes it but it does somewhat hide it)
    if "submitted_review" not in st.session_state:
        st.session_state["submitted_review"] = None # because of the above, layout B (after user submit review) will stil be shown if they move the slider or do whatever. previously it would go back to type A so it was fine to hide missing values for layout B (especially the JSON). but now we gotta deal with it by making submitted user review a global variable

    user_review = st.text_input("Input your user review: ")

    if st.button("Submit Review"):
        if user_review is not None and user_review != "":
            st.write("Review was submitted.")
            st.session_state["layout_type"] = "B"
            st.session_state["submitted_review"] = user_review # only if the user clicked submit review button again the global variable will be replaced by the new submitted review
            
            # we only run the ML models when the user actually submits the review
            if error_status_slang:
                print(f"Failed to load slang dictionary from database - error_status: {error_status_slang}")
                st.toast(error_status_slang, icon="❌")
                st.session_state["cleaned_review"] = pipeline.clean_review_text(
                    text=user_review,
                    data_dict=None
                )
            else:
                st.session_state["cleaned_review"] = pipeline.clean_review_text(
                    text=user_review,
                    data_dict=slang_dict
                )
            st.session_state["review_vector"] = vectorizer.transform([st.session_state["cleaned_review"]]) # transform() function from vectorizer expects a list of strings, not just a string
            st.session_state["prediction_proba"] = model.predict_proba(st.session_state["review_vector"])
        else: # included the backend button logic inside the if statement above, just a tiny optimization
            st.write("Please input a valid review.")
            st.session_state["layout_type"] = "A"

    # since leader_index is reset to None on every func rerun, i excluded this from the submit review button so leader_index can still be calculated and persist across reruns.
    # ts also allows live updating prediction result since we now stay at layout_type B on every rerun (very cool)
    # not that important but: accidentally included them in the button logic above earlier and caused the predicted senitment in the json to be error every time i interacted with the prediction threshold. just for a future ref
    if "prediction_proba" in st.session_state and st.session_state["prediction_proba"] is not None:
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
        if st.session_state["layout_type"] == "A":
            st.divider()
            st.subheader("Result")
            st.metric(
                label="Sentiment Prediction:",
                value="Waiting for submission...",
            )
        
        elif st.session_state["layout_type"] == "B":
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
                        "Original Review" : st.session_state.get("submitted_review", user_review), # if submitted review is none then user_review acts as like its safe guard
                        "Cleaned Review" : st.session_state["cleaned_review"],
                        "Sentiment Probabilities": {
                            "Negative" : st.session_state["prediction_proba"][0][0],
                            "Neutral" : st.session_state["prediction_proba"][0][1],
                            "Positive" : st.session_state["prediction_proba"][0][2],
                        },
                        "Predicted Sentiment" : "Positive" if leader_index == 2 else "Neutral" if leader_index == 1 else "Negative" if leader_index == 0 else "Error",
                    })

if __name__ == "__main__":
    pass