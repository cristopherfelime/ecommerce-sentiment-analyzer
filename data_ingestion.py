import pandas as pd
import sqlite3

# initialize db connection and cursor
def initialize_db():
    conn = sqlite3.connect("ecom_nlp_production.db") # connect to a db (creates one if not exists)
    c = conn.cursor() # initiate cursor object based on the connection
        
    # some good practices for future reference
    c.execute("PRAGMA journal_mode = WAL") # makes sqlite faster in concurrent read/writes
    c.execute("PRAGMA cache_size = -64000") # uses 64mb of memory for caching
    c.execute("PRAGMA foreign_keys = ON") # enforces foreign key constraints
    conn.commit()
    
    return conn, c

# creating predefined schema for our main data
def create_schema_data(conn, c):
    c.execute("""
        CREATE TABLE IF NOT EXISTS reviews (
            review_text TEXT,
            sentiment_label TEXT
        )
    """)
    conn.commit()

# create a predefined schema for our lexicon data
def create_schema_lexicon(conn, c):
    c.execute("""
        CREATE TABLE IF NOT EXISTS lexicons (
            slang TEXT,
            formal TEXT
        )
    """)
    conn.commit()

# ingest the raw csv into sqlite
def ingest_data(csv_path, conn):
    df = pd.read_csv(csv_path)
    df = df[["review_text", "sentiment_label"]]
    df.to_sql("reviews", conn, if_exists="append", index=False)
    conn.commit()
    print(f"Successfully ingested {len(df)} rows into the database.")

# initialize slang dictionary, to prevent loading the big slang csv multiple times
def ingest_lexicons(dict_path, conn):
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

if __name__ == "__main__":
    conn, c = initialize_db()
    create_schema_data(conn, c)
    create_schema_lexicon(conn, c)
    ingest_data("tokopedia_product_reviews_2025.csv", conn)
    ingest_lexicons("colloquial-indonesian-lexicon.csv", conn)

# test
# c.execute("""
#     SELECT review_id, review_text
#     FROM reviews
#     LIMIT 5;
# """)

# first_row = c.fetchone()
# print(f"Row ID: {first_row[0]}")
# print(f"Review Text: {first_row[1]}")

# print("----")

# all_rows = c.fetchall()
# for row in all_rows:
#     print(f"Row ID: {row[0]}")
#     print(f"Review Text: {row[1]}")