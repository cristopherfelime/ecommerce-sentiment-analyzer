# Genesis Reviewer - An E-Commerce Review Sentiment Analyzer and Dashboard

![Python](https://img.shields.io/badge/Python-3.12.13%2B-blue?style=for-the-badge&logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.58.0-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![SQLite](https://img.shields.io/badge/SQLite-WAL_Optimized-003B57?style=for-the-badge&logo=sqlite&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/scikit--learn-1.5.1-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Pandas](https://img.shields.io/badge/pandas-2.2.2-150458?style=for-the-badge&logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/numpy-1.26.4-013243?style=for-the-badge&logo=numpy&logoColor=white)
![Plotly](https://img.shields.io/badge/Plotly-5.24.1-3F4F75?style=for-the-badge&logo=plotly&logoColor=white)
![SciPy](https://img.shields.io/badge/SciPy-1.13.1-8CAAE6?style=for-the-badge&logo=scipy&logoColor=white)

Genesis Reviewer is an executive sentiment reviewer and business analytics dashboard revolving around E-Commerce. Originally planned as a simple frontend for some Classification Machine Learning models I've trained, it has evolved into a comprehensive dashboard to be used for business intelligence and analytics by Data Analysts and Business Executives alike.

One of the inspirations behind this project was to build something that could support the E-Commerce market in my region. Since smaller Indonesian NLP projects are also relatively uncommon compared to their English counterpart, looking up references was very difficult, which made this quite a challenge to build.

### This project is **NOT** Vibe Coded. AI is only involved at the architectural level, for code reviewing and improvement suggestions, as well as drafting Git commit messages.

## Project Details and Features

- **Automated ML Training Pipeline**:
This project features a complete, automated Machine Learning training pipeline (`train_model.py`) that performs the following steps in order:
    - Pulling both reviews and slang dictionary from their respective tables in the database
    - Preprocess the data by removing any rows with missing values in relevant columns as well as dropping duplicated rows, encoding the sentiment labels and cleaning the review texts using the pulled slang dictionary
    - Perform train-test split, vectorize the reviews, perform hyperparameter tuning using `RandomizedSearchCV` and immediately refitting, showing its performance against the test set in the console
    - Automatically chooses the best model with the highest **Macro Average F1-Score**, package them into a `Pipeline` object alongside the vectorizer, and finally pickling them to be used in the Streamlit dashboard.
- **Indonesian NLP Preprocessing**: 
Thankfully, the `NLTK` library has Indonesian stopwords corpus, I don't need to create a large list of my own stopwords that could potentially take hours just to smack it into the vectorizer. I also managed to find an open source Indonesian slang-formal lexicon corpus to aid in my text cleaning pipeline. Anyway, I still had to manually remove specific negation words from the corpus since the model is already struggling enough with the massive 71:1 positive-to-negative reviews ratio from the reviews dataset we use, this actually boosted our scores.
- **Interactive Web Dashboard**:
Built with `streamlit` with proper caching and session variable management, the dashboard is expected to run relatively smoothly. `plotly.express` also integrates well with `streamlit`. The dashboard is primarily split into three tabs, each with its own purpose:
  - **Single Review Tab**: For analyzing individual reviews. Enter a single review, and receive a sentiment prediction instantly.
  - **Batch Analysis Tab**: For analyzing batch datasets of reviews. Insights on uploaded datasets are also provided, it also updates live with every model control change on the left sidebar.
  - **Database BI Tab**: For visualizing cumulative database statistics. While this one does not update live, it still provides the user with a refresh database pull button. This tab provides a comprehensive overview of data stores within the database through KPIs, TIme-Series Line chart, and other visualizations.
- **Local Database Integration**: 
`sqlite3` is used to create a simple local database. Write Ahead Logging (WAL) is enabled to accommodate for any concurrent transaction processing happening in the database should there be more than two users, and should also be an additional protection to any unexpected database corruption. Every instances of database interaction within the project, primarily during querying, is protected through parameterized queries and other specific conditional guards to prevent any potential SQL-injection attack or database corruption.

## Project Structure

Following is the folder structure of this project:

```text
ecommerce-sentiment-analyzer/
├── app.py                   # Main Streamlit application controller
├── app_utils.py             # Streamlit caching functions, session management, and ML Pipeline utilizations
├── database_utils.py        # Database pull and push operations (primarily used by Streamlit)
├── pipeline.py              # Where the clean_review_text() function comes from
├── train_model.py           # Automated model training, tuning, and Pipeline bundling with pickling in the end
├── data_ingestion.py        # Database initialization, raw data ingestion, as well as connection management
├── requirements.txt         # List of project dependencies
├── csv/                     # folder for the raw CSV files (READ RUNNING THE DASHBOARD SECTION BELOW)
│   ├── tokopedia_product_reviews_2025.csv   # raw e-commerce reviews csv file
│   └── colloquial-indonesian-lexicon.csv    # raw slang dictionary csv file
├── database/                # folder for the SQLite database file
│   └── ecom_nlp_production.db   # SQLite database file
├── models/                  # folder for the ML model pipeline package
│   └── pipeline_package.joblib  # Pickled Vectorizer + ML Model Pipeline object
└── tabs/                    # Folder containing Streamlit tabs
    ├── single_review.py     # UI for individual review predictions
    ├── batch_analysis.py    # UI for bulk CSV review analysis
    └── database_bi.py       # UI for database BI (including KPIs and visualizations)
```

## Running the Dashboard

### 1. Cloning the Repository

```bash
git clone https://github.com/cristopherfelime/ecommerce-sentiment-analyzer.git
cd ecommerce-sentiment-analyzer
```

### 2. Creating Virtual Environment and Installing Dependencies

```bash
python -m venv venv
source venv/bin/activate 
# On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Download Datasets
Download the dataset from Kaggle and the slang dictionary from the GitHub repository linked below in the **Citations and Resources** section. Place the files within the project folder, inside another folder named `csv`.

### 4. Database Initialization
Before running the dashboard, ingest the initial dataset and create the database tables:
```bash
python data_ingestion.py
```

### 5. Model Training
Run the automated ML model training script to perform hyperparameter tuning, model training and generate the ML pipeline package:
```bash
python train_model.py
```

### 6. Launch the Dashboard
Run the Streamlit application:
```bash
streamlit run app.py
```

## Noted Issues of the Project
As this is my own individual project I've made for studying and learning purposes, I've had time to reflect back on what I've built and some improvements I could make. So here are some issues that I am aware of that are still present in the project:
- Poor ML model performance: The dataset I've chosen as the base dataset for this project is heavily skewed towards positive reviews. I've mentioned earlier regarding the 71:1 positive-to-negative (and neutrals) reviews ratio. This results in only 0.52 Macro Average F1 Score for our best ML model (which is Logistic Regression). Looking at the bright side, the calculated the Baseline Macro Average F1-Score (where if a model only predicts positive) of the dataset is only 0.326:
(0.98 + 0 + 0) / 3 = 0.326
So while my ML score is not that impressive for industry use, it is still significantly better than just predicting the most frequent class all the time.
- Static Table Schemas: A static table schema is still present in the database, for both the "reviews" and "lexicons" tables. They were made specifically for the reviews dataset columns and slang dictionary that can be found on resource citations below this section.

I've come up with improvements and solutions for these issues, but didn't end up implementing them yet, as I've had enough scope creep while building this project. Some solutions includes practicing data scraping to find more entries to solve the data imbalance, as well as either implementing a more dynamic and flexible table schema for the database.

## Citations and Resources

- E-Commerce Dataset: Tokopedia Product Reviews (2025) curated by Salman Abdu. [Kaggle Source](https://www.kaggle.com/datasets/salmanabdu/tokopedia-product-reviews-2025)
- Colloquial Indonesian Lexicon: Kamus Alay colloquial mapping repository curated by Nasalsabila. [GitHub Source](https://github.com/nasalsabila/kamus-alay/blame/master/colloquial-indonesian-lexicon.csv)
