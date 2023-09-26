import pandas as pd
from sqlalchemy import create_engine, text as sql_text
import logging
from typing import List
import configparser

# Setting up logging
logging.basicConfig(
    filename="../logs/database_helpers.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)


def store_to_sql(df: pd.DataFrame, engine: create_engine, table_name: str) -> None:
    """
    Store the DataFrame to a SQL table.

    :param df: DataFrame to be stored.
    :param engine: SQLAlchemy engine.
    :param table_name: Name of the table to store data.
    """
    try:
        logging.info(f"Storing data to table {table_name}...")
        df.to_sql(table_name, engine, if_exists="append", index=False)
        logging.info("Data stored successfully!")
    except Exception as e:
        logging.error(f"Error while storing data to SQL: {e}")
        raise


def setup_database() -> create_engine:
    """
    Setup the database connection using environment variables.

    :return: SQLAlchemy engine.
    """
    # Load the configuration
    config = configparser.ConfigParser()
    config.read('../config/config.ini')  # You should provide the absolute path to your config file here.

    # Fetch values from the configuration
    username = config['database']['DB_USERNAME']
    password = config['database']['DB_PASSWORD']
    server_ip = config['database']['DB_SERVER_IP']
    port = config['database']['DB_PORT']
    database_access = f"postgresql://{username}:{password}@{server_ip}:{port}/cardb"

    engine = create_engine(database_access)
    return engine


def read_sql_query(engine: create_engine, query: str) -> pd.DataFrame:
    """
    Load data from a SQL database.

    :param engine: SQLAlchemy engine.
    :param query: SQL query to retrieve the data.
    :return: Loaded DataFrame.
    """
    try:
        logging.info("Loading data from the database...")
        df = pd.read_sql(sql_text(query), engine.connect())
        return df
    except Exception as e:
        logging.error(f"Error while loading data: {e}")
        raise


def get_columns_names(engine, table_name):
    # Load
    query = f"""
        SELECT *
        FROM {table_name}
        limit 1;
    """
    tmp = read_sql_query(engine, query)
    return list(tmp.columns)


def drop_columns(engine, table_name, columns_to_drop):
    """
    Drop columns from a table in a database.

    :param engine: SQLAlchemy engine object
    :param table_name: Name of the table
    :param columns_to_drop: List of column names to be dropped
    """
    with engine.connect() as connection:
        for column in columns_to_drop:
            query = f'ALTER TABLE {table_name} DROP COLUMN "{column}";'
            connection.execute(sql_text(query))
            connection.commit()
            print(f"Column '{column}' dropped from '{table_name}'.")


def delete_all_records(engine) -> None:
    """
    Deletes all records from the 'engineered_car_data' table.
    :param engine: SQLAlchemy engine object
    """
    with engine.connect() as connection:
        connection.execute(sql_text("DELETE FROM engineered_car_data"))
        connection.commit()


def add_missing_columns(df: pd.DataFrame, missing_columns: List[str]):
    """
    :param df: Dataframe to process
    :param missing_columns:
    :return:
    """
    df_processed = df.copy()
    for col in missing_columns:
        if col not in df_processed.columns:
            df_processed[col] = None

    return df_processed
