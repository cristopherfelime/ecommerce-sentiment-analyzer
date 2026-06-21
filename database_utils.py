import numpy as np
import pandas as pd
import sqlite3

import data_ingestion

# a generic function to pull specific columns from a specific table (for examples: used in app.py load_slang_dict and one other cache func)
def pull_db(columns=None, table_name="reviews"): # columns is expecting a collection of strings
    conn, _ = data_ingestion.initialize_db()
    try:
        df = None
        error_status = None

        if (columns is None) and (table_name == "reviews"): # column as none guard
            columns = [
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
                ]
        elif (columns is None) and (table_name == "lexicons"): # same like above but with the lexicons table
            columns = [
                "slang",
                "formal"
            ]

        # -- to prevent SQL injection
        tables_in_db = pd.read_sql_query("""
            SELECT name
            FROM sqlite_master
            WHERE type = 'table';
        """, conn)["name"].tolist() # found out abt how to get all table names in our db, take the name column from the database returned from read_sql_query() then convert them to list

        if table_name not in tables_in_db: # this one prevents it through table
            print(f"Table {table_name} was not found in the database")
            error_status = f"Table {table_name} was not found in the database"
            return df, error_status
        else:
            valid_columns = pd.read_sql_query(f"""
                PRAGMA table_info({table_name});
            """, conn)["name"].tolist()
            invalid_columns = []
            for col in columns:
                if col not in valid_columns:
                    invalid_columns.append(col)
            if invalid_columns:
                print(f"Columns not found in table {table_name}: {invalid_columns}")
                error_status = f"Columns not found in table {table_name}: {invalid_columns}"
                return df, error_status
        
        
        # --

        try:
            df = pd.read_sql_query(f"""
                SELECT {', '.join(col.lower() for col in columns)}
                FROM {table_name.lower()};
            """, conn) # in the select statement it will join each column string col from columns with ", ". itll look something like: col_a, col_b, col_c, etc
        except Exception as e:
            print(f"Failed to load from database: {e}")
            error_status = f"Failed to load from {table_name} in database: {e}"
    finally:
        conn.close()

    return df, error_status

# generic function to push data into specific columns of a specific table in the db
def push_db(df, columns=None, table_name=None):
    conn, c = data_ingestion.initialize_db()
    try:
        error_status = None

        if (table_name is None) and (columns is None): # none guard thing
            print("No column nor table was provided.")
            error_status = "No column nor table was provided."
            return error_status

        if table_name:
            tables_in_db = pd.read_sql_query("""
                SELECT name
                FROM sqlite_master
                WHERE type = 'table';
            """, conn)["name"].tolist()
            if table_name not in tables_in_db:
                print(f"Table {table_name} was not found in the database")
                error_status = f"Table {table_name} was not found in the database"
                return error_status
            if columns:
                invalid_columns = []
                for i in columns: # no column in dataframe guard
                    if i not in df.columns:
                        invalid_columns.append(i)
                if invalid_columns:
                    print(f"Column(s) {', '.join(invalid_columns)} not found in dataframe provided. Check the formatting and if the column is correct.")
                    error_status = f"Column(s) {', '.join(invalid_columns)} not found in dataframe provided. Check the formatting and if the column is correct."
                    return error_status
        else:
            print("No table name provided.")
            error_status = "No table name provided."
            return error_status

        try: # the if and elif are for existing column in our db already, but we made it so that it can accomodate other columns and tables in our database as well like we did above (should there be any)
            if (table_name == "reviews") and (columns is None): # if table name was given as "review" but no column provided, it defaults to the ones available in our database
                numpy_df = df[[ 
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
                ]].to_numpy()
                c.executemany("""
                        INSERT INTO reviews (
                            review_text,
                            review_date,
                            review_id,
                            product_name,
                            product_category,
                            product_variant,
                            product_price,
                            product_url,
                            product_id,
                            rating,
                            sold_count,
                            shop_id,
                            sentiment_label
                        )
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
                    """, numpy_df) # ts a standard parameterized insert statement, primarily prevents sql injection
            elif (table_name == "lexicons") and (columns is None): # same like above but with the lexicons table
                numpy_df = df[[
                    "slang",
                    "formal"
                ]].to_numpy()
                c.executemany("""
                    INSERT INTO lexicons (
                        slang,
                        formal
                    )
                    VALUES (?, ?);
                """, numpy_df)
            elif (table_name is not None) and (columns is not None): # if both table name and columns are provided, it will use the columns to insert data into the table
                numpy_df = df[[col for col in columns]].to_numpy() # convert the dataframe to numpy array
                question_marks = ", ".join("?" for i in columns) # for how many columns are provided, it will provide that amount of question marks to be put in VALUES in the SQL query
                c.executemany(f"""
                    INSERT INTO {table_name} (
                        {', '.join(col for col in columns)}
                    )
                    VALUES ({question_marks});
                """, numpy_df) # in the INSERT, join the column names (lowercased) with ", " similar to the one above
            elif (table_name is not None) and (columns is None): # if a table name was provided but not columns, it will assume that all columns from the dataframe will be pushed into the table in the database
                numpy_df = df.to_numpy()
                columns = list(df.columns)
                question_marks = ", ".join("?" for i in columns)
                c.executemany(f"""
                    INSERT INTO {table_name} (
                        {', '.join(col for col in columns)}
                    )
                    VALUES ({question_marks});
                """, numpy_df)
            conn.commit()
            print(f"Database push was successful, pushed {len(numpy_df)} rows into table {table_name}.")
        except Exception as e:
            print(f"Database push to {table_name} failed: {e}")
            error_status = f"Database push to {table_name} failed: {e}"
    finally:
        conn.close()

    return error_status

if __name__ == "__main__":
    pass