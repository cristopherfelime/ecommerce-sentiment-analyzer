import numpy as np
import pandas as pd
import sqlite3

# initialize db connection and cursor
conn = sqlite3.connect("ecom_nlp_production.db") # connect to a db (creates one if not exists)
c = conn.cursor() # initiate cursor object based on the connection

# some good practices for future reference
c.execute("PRAGMA journal_mode = WAL") # makes sqlite faster in concurrent read/writes
c.execute("PRAGMA cache_size = -64000") # uses 64mb of memory for caching
c.execute("PRAGMA foreign_keys = ON") # enforces foreign key constraints
conn.commit()

# just for local development test
c.execute("DROP TABLE IF EXISTS reviews")
conn.commit()

# creating predefined schema
c.execute("""
    CREATE TABLE IF NOT EXISTS reviews (
        review_id INTEGER PRIMARY KEY AUTOINCREMENT,
        review_text TEXT NOT NULL,
        review_date DATE NOT NULL,
        product_name TEXT NOT NULL,
        product_category TEXT NOT NULL,
        product_variant TEXT NOT NULL,
        product_price INTEGER NOT NULL,
        product_url TEXT NOT NULL,
        product_id INTEGER NOT NULL,
        rating INTEGER NOT NULL,
        sold_count INTEGER NOT NULL,
        shop_id INTEGER NOT NULL,
        sentiment_label TEXT NOT NULL
    )
""")
conn.commit()

# ingest the raw csv into sqlite
df = pd.read_csv("tokopedia_product_reviews_2025.csv")
df.to_sql("reviews", conn, if_exists="append", index=False)
conn.commit()

# test
c.execute("""
    SELECT review_id, review_text
    FROM reviews
    LIMIT 5;
""")

first_row = c.fetchone()
print(f"Row ID: {first_row[0]}")
print(f"Review Text: {first_row[1]}")

print("----")

all_rows = c.fetchall()
for row in all_rows:
    print(f"Row ID: {row[0]}")
    print(f"Review Text: {row[1]}")