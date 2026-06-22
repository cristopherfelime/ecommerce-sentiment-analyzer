import pandas as pd
import sqlite3
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__)) # returns the directory where the current script is located

# ---------------------------------------------------------------------

# initialize db connection and cursor
def initialize_db(db_inner_path="ecom_nlp_production.db"):
    db_dir = os.path.join(BASE_DIR, "database") # create a directory string with "database" at the end
    os.makedirs(db_dir, exist_ok=True) # create the directory based on path above (if it doesn't exist, due to exist_ok=True)
    db_path = os.path.join(db_dir, db_inner_path) # combining the database folder file and the name of our soon-to-be database to form a path to the database file
    conn = sqlite3.connect(db_path) # connect to a db from path above (creates one if not exists)
    c = conn.cursor() # initiate cursor object based on the connection
        
    # some good practices for future reference
    c.execute("PRAGMA journal_mode = WAL") # makes sqlite faster in concurrent read/writes
    c.execute("PRAGMA cache_size = -64000") # uses 64mb of memory for caching
    c.execute("PRAGMA foreign_keys = ON") # enforces foreign key constraints
    conn.commit()
    
    return conn, c

# creating predefined schema for our main data (reviews table)
def create_schema_data(conn, c):
    c.execute("""
        CREATE TABLE IF NOT EXISTS reviews (
            review_text TEXT,
            review_date TEXT,
            review_id INTEGER,
            product_name TEXT,
            product_category TEXT,
            product_variant TEXT,
            product_price INTEGER,
            product_url TEXT,
            product_id INTEGER,
            rating INTEGER,
            sold_count INTEGER,
            shop_id INTEGER,
            sentiment_label TEXT
        )
    """)
    conn.commit()

# create a predefined schema for our lexicon data (lexicons table)
def create_schema_lexicon(conn, c):
    c.execute("""
        CREATE TABLE IF NOT EXISTS lexicons (
            slang TEXT,
            formal TEXT
        )
    """)
    conn.commit()

# ingest the raw csv into sqlite
def ingest_data(conn, df=None, csv_path=None, table_name="reviews"):
    if df is None:
        if csv_path is None:
            raise ValueError("csv_path not provided")
        df = pd.read_csv(csv_path)
    df.to_sql(table_name, conn, if_exists="append", index=False)
    conn.commit()
    print(f"Successfully ingested {len(df)} rows into table \"{table_name}\" in database.")

# initialize slang dictionary, to prevent loading the big slang csv multiple times
def ingest_lexicons(conn, data=None, dict_path=None):
    if data is None:
        if dict_path is None:
            raise ValueError("dict_path not provided")
        data = pd.read_csv(dict_path)
    data = data.loc[:, ["slang", "formal"]] # from the loaded csv, filter only columns slang and formal (automatic rearrangement as well)
    
    # a specific issue observed from test sample in notebook: resolving month name collisions (such as "jan" turning into "jangan")
    # we will prune those months
    months = ["jan", "feb", "mar", "apr", "mei", "jun", "jul", "agu", "sep", "okt", "nov", "des"]
    for month in months:
        if month in data["slang"].values:
            data = data[data["slang"] != month]
    
    data.to_sql("lexicons", conn, if_exists="append", index=False)
    conn.commit()

# ---------------------------------------------------------------------

# main guard

if __name__ == "__main__":
    conn, c = initialize_db("ecom_nlp_production.db")
    create_schema_data(conn, c)
    create_schema_lexicon(conn, c)
    ingest_data(conn, csv_path=os.path.join("csv", "tokopedia_product_reviews_2025.csv"))
    ingest_lexicons(conn, dict_path=os.path.join("csv", "colloquial-indonesian-lexicon.csv"))