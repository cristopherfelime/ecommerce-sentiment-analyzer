import pipeline
import data_ingestion

import numpy as np
import pandas as pd
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline
from scipy.stats import loguniform, uniform

import nltk
nltk.download("stopwords")
from nltk.corpus import stopwords

# ---------------------------------------------------------------------

# support functions

# function to remove missing values
def clean_missing_val(df):
    print("Checking for any missing value in columns..")
    missing_val_col = []
    for i, j in enumerate(df.isna().any()):
        if j != False:
            print(f"Column {i} has missing value!")
            missing_val_col.append(i)

        else:
            print(f"No missing value in column {i}")
    print(f"Following columns have missing value present: {[[df.columns[i] for i in missing_val_col] if missing_val_col else "None"]}")
    if missing_val_col:
        print("Removing rows with missing values..")
        df = df.dropna()
        print("Empty values have been removed from the dataset.")
    print("-----------------------")
    return df

# function to remove duplicate rows
def clean_dupe_rows(df):
    print("Checking for any duplicate rows..")
    if df.duplicated().any():
        print(f"A total of {df.duplicated().sum()} duplicate rows have been found in the dataset. Removing them..")
        df = df.drop_duplicates(keep="first")
    print(f"{df.duplicated().sum()} duplicate rows present in the database.\n{str(df.shape[0])} rows left in the database.")
    print("-----------------------")
    return df

# function to do both
def clean_df(df):
    df = clean_missing_val(df)
    df = clean_dupe_rows(df)
    return df.copy()

# ---------------------------------------------------------------------

# main training pipeline

# get raw data
def sql_retrieval_reviews(*reviews_columns):

    # handle case where no column names are given
    if not reviews_columns:
        reviews_columns = [
            "review_text",
            "sentiment_label"
        ]

    # SQL injection defense by whitelisting columns
    whitelisted_columns = { # hey set is great for checking if an item is inside a collection, lowk it's very fast i heard
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
    }
    for i in reviews_columns:
        if i not in whitelisted_columns:
            raise ValueError(f"Unauthorized column name \"{i}\" was found. Do NOT even try to SQL Inject lol.")

    # fetch data from database
    columns = ", ".join(reviews_columns)
    query_reviews = f"""
        SELECT {columns}
        FROM reviews;
    """
    conn, _ = data_ingestion.initialize_db()

    # try finally will ensure that the connection is closed even if an error occurs
    try:
        df = pd.read_sql_query(query_reviews, conn)
    finally:
        conn.close()

    # returning relevant objects to be used in the main guard below
    return df

def sql_retrieval_lexicons():
    
    query_lexicons = """
        SELECT slang, formal
        FROM lexicons;
    """
    conn, _ = data_ingestion.initialize_db()
    try:
        slang_df = pd.read_sql_query(query_lexicons, conn)
    finally:
        conn.close()
    
    # returning relevant objects to be used in the main guard below
    return slang_df

# function to preprocess data
def data_preprocessing(df=None, slang_df=None, slang_input=None):

    # Data Cleaning
    if df is not None:
        df = clean_df(df)

        # Ordinal Encoding
        print("Encoding sentiment labels...")
        df["sentiment_label"] = df["sentiment_label"].map({"negative" : 0, "neutral" : 1, "positive" : 2})
    else:
        print("No reviews dataframe provided, skipping data cleaning..")

    # Clean review text using slang dictionary
    if slang_df is not None:
        slang_dict = dict(zip(slang_df["slang"], slang_df["formal"]))
        print("Cleaning review text using slang dictionary...")
        if df is not None:
            df["review_text"] = df["review_text"].apply(
                lambda x : pipeline.clean_review_text(text=x, data_dict=slang_dict)
            )
        else:
            print("Cannot perform slang cleaning, no dataframe provided.")
    else:
        print("No slang dataframe provided, skipping slang cleaning..")
    
    if df is not None and slang_input is not None:
        df["review_text"] = df["review_text"].apply(
            lambda x : pipeline.clean_review_text(text=x, data_dict=slang_input)
        )

    # returning relevant objects to be used in the main guard below
    if df is not None and slang_df is not None:
        print("Data cleaning returned df and slang_dict.")
        return df, slang_dict
    elif df is not None:
        print("Data cleaning returned df only.")
        if slang_input is not None:
            print("Cleaning performed using input slang dictionary")
            return df
        else:
            print("Warning: Cleaning may be incomplete due to no slang dictionary provided.")
            return df
    elif slang_df is not None:
        print("Data cleaning returned slang_df only")
        return slang_dict
    else:
        print("No data provided")
        return None
    
# function to do train-test split
def perform_tts(df):

    # Train-Test split
    print("Splitting data into training and testing sets...")
    X = df["review_text"]
    y = df["sentiment_label"]
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        shuffle=True,
        stratify=y
    )

    # returning relevant objects to be used in the main guard below
    return X_train, X_test, y_train, y_test

# function to vectorize text
def vectorize_test(X_train, X_test):
    # Text Vectorization
    print("Vectorizing text data...")
    indo_stopwords = stopwords.words("indonesian")
    indo_negation_words = { # set cuz faster
        "tidak", "bukan", "jangan", "belum", "tanpa", "kurang", "tak", "tiada",
        "tidaklah", "bukanlah", "belumlah", "janganlah", 
        "tidakkah", "bukankah", "belumkah", "bukannya",
        "nggak", "gak", "ga", "ndak", "kagak", "enggak", "ngga", "engga"
    } # apparently nltk has these listed as stopwords lmao. this is to ensure that no negation words are present in the list of stopwords we're using (so that ngram may work properly)
    indo_stopwords = [i for i in indo_stopwords if i not in indo_negation_words]
    vectorizer = TfidfVectorizer(
        stop_words=indo_stopwords, 
        ngram_range=(1, 2), 
        min_df=3, 
        max_features=20000
    )
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    # returning relevant objects to be used in the main guard below
    return X_train_tfidf, X_test_tfidf, vectorizer

def initialize_models():
    # Model initialization
    print("Initializing models...")
    lr_model = LogisticRegression(
        max_iter=1000,
        random_state=42,
    )
    linearsvc_model = LinearSVC(
        multi_class="ovr",
        random_state=42,
    )

    # returning relevant objects to be used in the main guard below
    return lr_model, linearsvc_model

def tune_fit_models(lr_model, linearsvc_model, distribution_lr, distribution_svc, lr_rscv_params, svc_rscv_params, X_train_tfidf, y_train):
    # Hyperparameter tuning with RandomizedSearchCV
    lr_rscv = RandomizedSearchCV(lr_model, distribution_lr, **lr_rscv_params, refit=True)
    linearsvc_rscv = RandomizedSearchCV(linearsvc_model, distribution_svc, **svc_rscv_params, refit=True)

    # Model fitting
    print("Fitting models...")
    lr_rscv.fit(X_train_tfidf, y_train)
    linearsvc_rscv.fit(X_train_tfidf, y_train)

    # showing best params found in console
    print(f"Best Logistic Regression Parameters found: {lr_rscv.best_params_}")
    print(f"Best Linear SVC Parameters found: {linearsvc_rscv.best_params_}")

    return lr_rscv.best_estimator_, linearsvc_rscv.best_estimator_

def models_predict(lr_model, linearsvc_model, X_test_tfidf):
    # Model prediction
    print("Making predictions...")
    lr_preds = lr_model.predict(X_test_tfidf)
    linearsvc_preds = linearsvc_model.predict(X_test_tfidf)

    # returning relevant objects to be used in the main guard below
    return lr_preds, linearsvc_preds

# function to return classification report
def classification_report_func(y_test, lr_preds, linearsvc_preds):
    print("---Logistic Regression---")
    print(classification_report(y_test, lr_preds))
    print("---Linear SVC---")
    print(classification_report(y_test, linearsvc_preds))

# pickling/dumpin the best model
def dump_package(lr_model, lr_preds, linearsvc_model, linearsvc_preds, y_test, vectorizer):
    # finding the best model based off macro avg f1 score
    lr_score = classification_report(y_test, lr_preds, output_dict=True)["macro avg"]["f1-score"]
    svc_score = classification_report(y_test, linearsvc_preds, output_dict=True)["macro avg"]["f1-score"]
    best_model = lr_model if lr_score > svc_score else linearsvc_model
    if lr_score > svc_score:
        print("Logistic Regression is the best model (based on macro average f1-score).")
    else:
        print("LinearSVC is the best model (based on macro average f1-score).")

    # now dumping the vectorizer and the best model
    os.makedirs("models", exist_ok=True)
    pickle_path = os.path.join("models", "pipeline_package.joblib")

    pipeline_models = Pipeline([
        ("vectorizer", vectorizer), # so pipeline except last one to be transformers 
        ("best_model", best_model) # so when predict() or any other model methods are called, it will go through all of the steps in the pipeline
    ]) # so ts essentially cool since it basically bundles every necessary objects and steps for preprocessing up to prediction into one

    joblib.dump(pipeline_models, pickle_path)
    print("Models have been successfully been pickled.")

# ---------------------------------------------------------------------

# main guard

if __name__ == "__main__":
    df = sql_retrieval_reviews()
    slang_df = sql_retrieval_lexicons()
    df, slang_input = data_preprocessing(df, slang_df)
    X_train, X_test, y_train, y_test = perform_tts(df)
    X_train_tfidf, X_test_tfidf, vectorizer = vectorize_test(X_train, X_test)
    lr_model, linearsvc_model = initialize_models()

    # Setting up hyperparameters for optimization
    distribution_lr = [{ # hyperparams for logistic regression
        "C" : loguniform(0.01, 1.0), # inverse regularization parameter, log uniform numbers from 0.01 to 100 (0.01, 0.1, 1, 10, 100)
        "solver" : ["lbfgs"], # optimization/solver algorithm used when training
        "tol" : [0.0001, 0.001], # numerical tolerance, training stops when model performance improvement drops below the number
        "class_weight" : [{0: 10.0, 1: 15.0, 2: 1.0},
                          {0: 12.0, 1: 18.0, 2: 1.0},
                          {0: 14.0, 1: 20.0, 2: 1.0},
                          {0: 16.0, 1: 22.0, 2: 1.0},
                          {0: 18.0, 1: 25.0, 2: 1.0},
                          {0: 20.0, 1: 28.0, 2: 1.0}] # penalty rate for each label if the model got it wrong
    }]
    distribution_svc = [ # hyperparams for linear svc
        # for squared hinge loss
        {
            "C" : loguniform(0.1, 1.0),
            "penalty" : ["l2"],
            "loss" : ["squared_hinge"],
            "dual" : [False],
            "tol" : [0.0001, 0.001],
            "class_weight" : [{0: 12.0, 1: 18.0, 2: 1.0},
                            {0: 15.0, 1: 20.0, 2: 1.0},
                            {0: 18.0, 1: 25.0, 2: 1.0},
                            {0: 22.0, 1: 30.0, 2: 1.0}]
        },
        # for traditional hinge loss
        {
            "C" : loguniform(0.1, 1.0),
            "penalty" : ["l2"],
            "loss" : ["hinge"],
            "dual" : [True], # dual=True is required for traditional hinge loss, though performance may be degraded cuz our n_samples > n_features
            "tol" : [0.0001, 0.001],
            "class_weight": [{0: 12.0, 1: 18.0, 2: 1.0},
                            {0: 15.0, 1: 20.0, 2: 1.0},
                            {0: 18.0, 1: 25.0, 2: 1.0},
                            {0: 22.0, 1: 30.0, 2: 1.0}]
        }
    ]

    # Now ts for the RandomizedSearchCV initializations for each models
    lr_rscv_params = { # for logistic regression
        "n_iter": 15, # number of repetition, 15 * 3 = 45 tries, so larger will be slower but more param combinations will be tried
        "cv": 3, # cross validation folds, 3 folds will divide training data into 3 (approx 66% train, 33% val), and do this 3 times
        "scoring": "f1_macro", # f1 macro is used cuz both precision and recall matters (and our current dataset is HEAVILY imbalanced)
        "random_state": 42, # for reproducitbility
        "n_jobs": -1, # use all available processors
    }
    
    svc_rscv_params = { # for linear svc
        "n_iter": 15,
        "cv": 3,
        "scoring": "f1_macro",
        "random_state": 42,
        "n_jobs": -1, 
    }

    lr_model, linearsvc_model = tune_fit_models(lr_model, linearsvc_model, distribution_lr, distribution_svc, lr_rscv_params, svc_rscv_params, X_train_tfidf, y_train)
    lr_preds, linearsvc_preds = models_predict(lr_model, linearsvc_model, X_test_tfidf)
    best_model = classification_report_func(y_test, lr_preds, linearsvc_preds)
    dump_package(lr_model, lr_preds, linearsvc_model, linearsvc_preds, y_test, vectorizer)