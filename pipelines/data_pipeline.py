import os
import json
from datetime import datetime
import pandas as pd
import numpy as np
import logging
import joblib

from sqlalchemy import create_engine, text as sql_text
from typing import List, Dict, Optional
from sklearn.feature_extraction.text import TfidfVectorizer
import simplemma

# Setting up logging
logging.basicConfig(
    filename="../logs/data_process.log",
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
    username = os.getenv("DB_USERNAME")
    password = os.getenv("DB_PASSWORD")
    DATABASE_URL = f"postgresql://{username}:{password}@localhost:5432/cardb"
    engine = create_engine(DATABASE_URL)
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


def handle_price(df: pd.DataFrame) -> pd.DataFrame:
    """
    Process the price columns in the dataframe.

    :param df: Original dataframe containing price columns.
    :return: Dataframe with processed price columns.
    """

    df_processed = df.copy()
    price_cols = ["vételár", "akciós ár", "extrákkal növelt ár", "vételár eur"]

    # Remove non-digit characters from the relevant columns.
    for col in price_cols:
        df_processed[col] = df_processed[col].str.replace(r"\D", "", regex=True)

    # Replace NaN values in 'vételár' with values from 'akciós ár', then 'extrákkal növelt ár'.
    replace_col = "extrákkal növelt ár"
    price_is_nan = df_processed["vételár"].isna()
    df_processed.loc[price_is_nan, "vételár"] = df_processed.loc[
        price_is_nan, replace_col
    ].values

    replace_col = "akciós ár"

    # Drop cars without a price.
    df_processed = df_processed[df_processed["vételár"] != ""]
    df_processed = df_processed[~df_processed["vételár"].isna()]

    # Convert 'vételár' to int.
    df_processed["vételár"] = df_processed["vételár"].astype(int)
    df_processed["vételár eur"] = df_processed["vételár eur"].astype(float)

    # Convert EUR prices to HUF.
    # Identify rows where the HUF and EUR price are the same.
    msk_eur_price = (
        (df_processed["vételár"] == df_processed["vételár eur"])
        & (df_processed["vételár"] < 200000)
    ).values
    euro_exchange_rate = 375
    count_of_invalid_price = np.sum(msk_eur_price)
    if np.sum(msk_eur_price) > 0:
        print(
            f"WARNING: {count_of_invalid_price} cars have inconsistent price value, EUR value is used!"
        )
        df_processed.loc[msk_eur_price, "vételár"] *= euro_exchange_rate

    return df_processed


def drop_high_nan_columns(
    df: pd.DataFrame, nan_ratio_threshold: float = 0.6
) -> pd.DataFrame:
    """
    Drop columns from the dataframe with fewer non-NaN values than the given threshold.

    :param df: Input dataframe containing potential columns with high NaN counts.
    :param nan_ratio_threshold: Over this NaN ratio drop the column.
                      Defaults to 0.6.
    :return: Dataframe with columns having high NaN counts removed.
    """

    # Calculate the dataset size.
    dataset_size = len(df)

    # Compute the number of NaN values in each column.
    nan_count = df.isna().sum()

    # Identify columns with fewer non-NaN values than the threshold.
    high_nan_cols = nan_count[(nan_count / dataset_size) > nan_ratio_threshold].keys()

    # Drop the identified columns.
    df_dropped = df.drop(columns=high_nan_cols)

    return df_dropped


def str_to_numeric(df: pd.DataFrame, columns: dict) -> pd.DataFrame:
    """
    Convert columns in a dataframe from string to specified numeric type.

    :param df: Input dataframe with columns to be converted.
    :param columns: Dictionary specifying columns as keys and target numeric type as values.
    :return: Dataframe with specified columns converted to numeric types.
    """

    # Make a copy of the dataframe to avoid modifying the original
    df_transformed = df.copy()

    for column, dtype in columns.items():
        # Remove non-digit characters and convert to the desired numeric type
        df_transformed[column] = (
            df_transformed[column].str.replace(r"\D", "", regex=True).replace('', np.nan).astype(dtype)
        )

    return df_transformed


def remove_non_alpha(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """
    Remove non-alphabetic characters from a specified column in the dataframe.

    :param df: Input dataframe with the column to be modified.
    :param column: The column from which non-alphabetic characters should be removed.
    :return: Dataframe with non-alphabetic characters removed from the specified column.
    """

    df_transformed = df.copy()
    df_transformed[column] = df_transformed[column].str.replace(
        r"[^a-zA-Z]", "", regex=True
    )
    return df_transformed


def transform_columns_to_numeric(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert multiple columns in the dataframe to numeric and perform specific transformations.

    :param df: Input dataframe to be transformed.
    :return: Transformed dataframe.
    """

    # Define columns and their target types
    columns_to_convert = {
        "km. óra állás": int,
        "szállítható szem. száma": float,
        "ajtók száma": float,
        "saját tömeg": float,
        "teljes tömeg": float,
        "csomagtartó": float,
        "hengerűrtartalom": float,
        "kezdőrészlet": float,
    }

    df_transformed = str_to_numeric(df, columns_to_convert)
    df_transformed = remove_non_alpha(df_transformed, "futamidő")

    return df_transformed


def convert_power_to_numeric(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert 'teljesítmény' column to numeric using only the kW unit.

    :param df: Input dataframe to be transformed.
    :return: Transformed dataframe.
    """

    df_transformed = df.copy()
    df_transformed["teljesítmény"] = (
        df_transformed["teljesítmény"]
        .str.replace(r"kW.*", "", regex=True)
        .str.replace(r"\D", "", regex=True)
        .astype(float)
    )

    return df_transformed


def convert_creation_date_to_age(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert 'évjárat' column to datetime format and compute the age in years.

    :param df: Input dataframe to be transformed.
    :return: Transformed dataframe.
    """
    current_month = str(datetime.now())[:7]
    df_transformed = df.copy()
    df_transformed["évjárat"] = pd.to_datetime(
        df_transformed["évjárat"].str.replace(r"\(.+\)", "", regex=True)
    )
    df_transformed["age (year)"] = (
        (pd.to_datetime(current_month) - df_transformed["évjárat"]).dt.days
        / 365
    ).astype(int)
    df_transformed.drop(
        columns=["age_days"], errors="ignore", inplace=True
    )  # remove the age_days column if it exists

    return df_transformed


def convert_technical_license_validity_date(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert 'műszaki vizsga érvényes' to datetime format and compute the difference in days.

    :param df: Input dataframe to be transformed.
    :return: Transformed dataframe.
    """
    current_month = str(datetime.now())[:7]
    df_transformed = df.copy()
    df_transformed["műszaki vizsga érvényes"] = np.where(
        df_transformed["műszaki vizsga érvényes"].str.contains(r"^/", regex=True),
        np.nan,
        df_transformed["műszaki vizsga érvényes"],
    )
    df_transformed["műszaki vizsga érvényes"] = (
        pd.to_datetime(df_transformed["műszaki vizsga érvényes"])
        - pd.to_datetime(current_month)
    ).dt.days

    return df_transformed


def extract_tire_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract various tire specifications using regex pattern and convert certain fields to numeric.

    :param df: Input dataframe to be transformed.
    :return: Transformed dataframe.
    """

    df_transformed = df.copy()
    pattern = "(?P<Width>\d+)/(?P<AspectRatio>\d+)\s(?P<ConstructionType>[A-Z])\s(?P<RimDiameter>\d+)"

    # Mapping columns to be extracted with their target column names
    column_mappings = {
        "nyári gumi méret": [
            "summer tire width",
            "summer tires aspect ratio",
            "summer tires construction type",
            "summer tires rim diameter",
        ],
        "téli gumi méret": [
            "winter tire width",
            "winter tires aspect ratio",
            "winter tires construction type",
            "winter tires rim diameter",
        ],
        "hátsó nyári gumi méret": [
            "back summer tire width",
            "back summer tires aspect ratio",
            "back summer tires construction type",
            "back summer tires rim diameter",
        ],
        "hátsó téli gumi méret": [
            "back winter tire width",
            "back winter tires aspect ratio",
            "back winter tires construction type",
            "back winter tires rim diameter",
        ],
    }

    for original_column, new_columns in column_mappings.items():
        df_transformed[new_columns] = df_transformed[original_column].str.extract(
            pattern
        )

    numeric_values_from_extractions = [
        "summer tire width",
        "summer tires aspect ratio",
        "summer tires rim diameter",
        "winter tire width",
        "winter tires aspect ratio",
        "winter tires rim diameter",
        "back summer tire width",
        "back summer tires aspect ratio",
        "back summer tires rim diameter",
        "back winter tire width",
        "back winter tires aspect ratio",
        "back winter tires rim diameter",
    ]
    df_transformed[numeric_values_from_extractions] = df_transformed[
        numeric_values_from_extractions
    ].astype(float)

    return df_transformed


def filter_invalid_power_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Filters out rows where the kW and LE values of 'teljesítmény' column do not match
    the expected mathematical relation.

    Specifically, it checks if the relationship between kW and LE (horsepower) values is valid.
    1 LE (horsepower) is approximately 0.746 kW. Rows where the difference between the actual kW value
    and the expected kW value (calculated from LE) exceeds 10 are removed.

    :param df: Input dataframe to be filtered.
    :return: Filtered dataframe.
    """

    # Extract kW and LE values from 'teljesítmény' column
    p_kw = (
        df["teljesítmény"]
        .str.replace(r"kW.*", "", regex=True)
        .str.replace(r"\D", "", regex=True)
        .astype(float)
    )
    p_le = (
        df["teljesítmény"].str.replace(r".*?(\d+).LE", r"\1", regex=True).astype(float)
    )

    # Filter rows where the difference between actual kW and expected kW (from LE) exceeds 10
    valid_indices = ~(abs(p_le * 0.746 - p_kw) > 10)
    return df[valid_indices].reset_index(drop=True)


def extract_feature_frequencies(df: pd.DataFrame) -> pd.Series:
    """
    Extracts and calculates the frequency of all special features in the dataset.

    :param df: The input dataframe.
    :return: A Series containing the frequency of each special feature.
    """
    without_description = df["feature_list"].values
    all_special_features = pd.concat([pd.Series(x) for x in without_description])
    return all_special_features.value_counts()


def determine_speaker_count(car_special_features: List[str]) -> float:
    """
    Determines the speaker count based on the provided special features of a car.

    :param car_special_features: List of special features of a car.
    :return: Speaker count or NaN if not found.
    """
    speaker_counts = [4, 6, 8, 10, 12]
    for sc in speaker_counts:
        if f"{sc} hangszóró" in car_special_features:
            return sc
    return np.nan


def filter_usable_features(feature_frequencies: pd.Series) -> List[str]:
    """
    Filters out unwanted special features from the feature frequencies.

    :param feature_frequencies: Frequencies of special features.
    :return: List of usable special features.
    """
    speaker_counts = [4, 6, 8, 10, 12]
    unwanted_features = list(pd.Series(speaker_counts).astype(str) + " hangszóró") + [
        "",
        "kültér",
        "műszaki",
        "beltér",
        "multimédia / navigáció",
        "egyéb információ",
    ]

    usable_features = feature_frequencies[feature_frequencies > 1000].keys()
    return [feat for feat in usable_features if feat not in unwanted_features]


def add_special_features(df: pd.DataFrame, usable_features: List[str]) -> pd.DataFrame:
    """
    Adds special features as binary columns to the dataframe.

    :param df: The input dataframe.
    :param usable_features: List of usable special features.
    :return: Dataframe augmented with special feature columns.
    """
    without_description = [np.array(x) for x in df["feature_list"].values]
    df_special = pd.DataFrame(
        [determine_speaker_count(x) for x in without_description],
        columns=["speaker count"],
    )

    for feature in usable_features:
        df_special[feature] = [feature in x for x in without_description]

    return pd.concat([df, df_special], axis=1)


def filter_price_outliers(df: pd.DataFrame, threshold: int = 100000) -> pd.DataFrame:
    """
    Filters out rows in the dataframe based on the 'vételár' column value.

    :param df: Input dataframe.
    :param threshold: The threshold value for the 'vételár' column. Rows with values below this threshold will be dropped.
    :return: Dataframe with rows filtered based on 'vételár'.
    """
    return df[df["vételár"] > threshold]


def drop_unwanted_columns(df: pd.DataFrame, col_list_file) -> pd.DataFrame:
    """
    Drops specified unwanted columns from the dataframe.

    :param df: Input dataframe.
    :param col_list_file: CSV file contains the name of columns to drop
    :return: Dataframe without the unwanted columns.
    """
    columns_to_drop = pd.read_csv(col_list_file)["col"].values

    return df.drop(columns=columns_to_drop, errors="ignore")


def load_json_mapping(filepath: str) -> Dict[str, str]:
    """
    Load a dictionary from a JSON file.

    :param filepath: Path to the JSON file.
    :return: Dictionary loaded from the JSON file.
    """
    with open(filepath, "r") as file:
        return json.load(file)


def rename_dataframe_columns(df: pd.DataFrame, mapping: Dict[str, str]) -> pd.DataFrame:
    """
    Rename columns of the input dataframe using the provided mapping dictionary.

    :param df: Input dataframe.
    :param mapping: Dictionary with old column names as keys and new names as values.
    :return: Dataframe with renamed columns.
    """
    unnamed_columns = set(df.columns) - set(mapping.keys())
    if unnamed_columns:
        print(f"Warning: The following columns were not renamed: {unnamed_columns}")

    return df.rename(columns=mapping)


def extract_brand_from_link(df: pd.DataFrame) -> pd.Series:
    """
    Extracts brand from the link column of the input dataframe.

    :param df: Input dataframe.
    :return: Series containing the extracted brand.
    """
    return df["link"].str.split("/").str[4]


def parse_color_information(df: pd.DataFrame) -> pd.DataFrame:
    """
    Processes the color column to determine metallic polish and normalize color name.

    :param df: Input dataframe.
    :return: Modified dataframe with metallic polish information and cleaned color.
    """
    df["metallic_polish"] = df["color"].str.contains("metál")
    df["color"] = df["color"].str.replace(" (metál)", "", regex=False).str.lower()
    return df


def load_hungarian_settlements(filepath: str) -> pd.DataFrame:
    """
    Loads and processes Hungarian settlements from a CSV file.

    :param filepath: Path to the settlements CSV.
    :return: DataFrame with Hungarian settlements.
    """
    df_hun = pd.read_csv(filepath)
    df_hun["cleaned_settlement"] = (
        df_hun["settlement"]
        .str.replace(r"Budapest.+", "Budapest", regex=True)
        .str.lower()
        .str.strip()
    )
    return df_hun


def get_city_from_zip(zip_code: int, settlements_df: pd.DataFrame) -> Optional[str]:
    """
    Gets the city corresponding to a given zip code.

    :param zip_code: Zip code to search.
    :param settlements_df: DataFrame of Hungarian settlements.
    :return: City name corresponding to the zip code or None.
    """
    city = settlements_df[settlements_df.zip == zip_code]["cleaned_settlement"]
    return city.values[0] if len(city) > 0 else None


def extract_address_information(
    df: pd.DataFrame, settlements_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Extracts and cleans address information to determine city for each entry in the dataframe.

    :param df: Input dataframe.
    :param settlements_df: DataFrame of Hungarian settlements.
    :return: Modified dataframe with city information.
    """
    mask = df["buy_from_shop"]
    address = pd.Series(
        np.where(mask, df.content_info_1, df.content_info_0)
    ).str.lower()
    address = address.str.replace("térkép megtekintése", "", regex=False)

    zip_code = (
        address.str.replace("[^\d ]", "", regex=True)
        .str.extract("( \d\d\d\d )")[0]
        .astype(float)
    )

    cities_based_on_zip = [get_city_from_zip(x, settlements_df) for x in zip_code]
    address_without_numbers = address.str.replace("[^a-záéíóöőúüű\s]", "", regex=True)
    df["city"] = address_without_numbers.str.extract(
        f"( {' | '.join(settlements_df['cleaned_settlement'].unique())} )"
    )
    df["city"] = df["city"].str.strip()

    df["city"] = np.where(df["city"].isna(), cities_based_on_zip, df["city"])
    return df


def address_pipeline(df: pd.DataFrame, settlements_filepath: str) -> pd.DataFrame:
    """
    Main pipeline function to process address, color, and brand information.

    :param df: Input dataframe.
    :param settlements_filepath: Path to the settlements CSV.
    :return: Modified dataframe with brand, metallic polish, color, and city information.
    """
    df["brand"] = extract_brand_from_link(df)
    df = parse_color_information(df)
    settlements_df = load_hungarian_settlements(settlements_filepath)
    df = extract_address_information(df, settlements_df)

    return df


def process_financing(df: pd.DataFrame) -> pd.DataFrame:
    """
    Process the 'financing' column of the dataframe by removing non-numeric characters.
    Handles missing values by replacing them with 100.

    :param df: Input dataframe with 'financing' column.
    :return: Modified dataframe with processed 'financing' column.
    """
    df["financing"] = (
        df["financing"]
        .fillna("100")
        .str.replace(r"\D", "", regex=True)
        .astype(int)
    )
    return df


def remove_taken_away_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Removes columns with names containing 'take' (case-insensitive).

    :param df: Input dataframe.
    :return: Modified dataframe without the taken away columns.
    """
    cols_to_drop = df.columns[df.columns.str.lower().str.contains("take")]
    df.drop(columns=cols_to_drop, inplace=True)
    return df


def clean_and_process_description(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans and processes the 'description' column of the dataframe.
    Strips extra spaces, removes introductory words, and calculates the description length and word count.

    :param df: Input dataframe with 'description' column.
    :return: Modified dataframe with cleaned 'description' and new columns for length and word count.
    """
    df["description"].fillna("", inplace=True)
    df["description"] = df["description"].str.strip()
    df["description"] = df["description"].str.replace(r"leírás\n", "", regex=True)
    df["word_count"] = df["description"].apply(lambda x: len(x.split()))
    return df


def lemmatize_text(text: str) -> str:
    """
    Lemmatizes the input text.

    :param text: Input string to be lemmatized.
    :return: Lemmatized string.
    """
    mytokens = text.split()
    lemmetized_tokens = [simplemma.lemmatize(x, lang="hu") for x in mytokens]
    return " ".join(lemmetized_tokens)


def clean_and_lemmatize_description(df: pd.DataFrame, word_map: dict) -> pd.DataFrame:
    """
    Cleans and lemmatizes the 'description' column of the dataframe.
    Applies additional manual lemmatization based on provided word map.

    :param df: Input dataframe with 'description' column.
    :param word_map: Dictionary for manual lemmatization.
    :return: Modified dataframe with 'description_lemmatized' column.
    """
    # Remove non-alphabetic characters
    df["description_lemmatized"] = df["description"].str.replace(
        "[^a-záéíóöőúüű\s]", "", regex=True
    )

    # Apply Lemmatization
    df["description_lemmatized"] = df["description_lemmatized"].apply(lemmatize_text)

    # Manual Lemmatization for Special Cases
    for k, v in word_map.items():
        df["description_lemmatized"] = df["description_lemmatized"].str.replace(k, v)

    return df


def apply_tfidf_vectorization(
    df: pd.DataFrame,
    stop_words: List[str],
    max_features: int = 50,
    vectorizer_file: str = "models/vectorizer.joblib",
    update: bool = False,
) -> pd.DataFrame:
    """
    Applies the TF-IDF vectorization on the 'description_lemmatized' column and adds the result to the dataframe.

    :param df: Input dataframe with 'description_lemmatized' column.
    :param stop_words: List of stop words.
    :param max_features: Maximum number of features for vectorization. Default is 50.
    :param vectorizer_file: File to save/load the vectorizer. Default is 'vectorizer.joblib'.
    :param update: If True, fit the vectorizer on new data and update it. Default is False.
    :return: Modified dataframe with new TF-IDF columns.
    """
    if os.path.exists(vectorizer_file) and not update:
        # Load existing vectorizer
        vectorizer = joblib.load(vectorizer_file)
    else:
        # Fit and save new vectorizer
        vectorizer = TfidfVectorizer(stop_words=stop_words, max_features=max_features)
        vectorizer.fit(df["description_lemmatized"])
        joblib.dump(vectorizer, vectorizer_file)

    tfidf_feature_names = vectorizer.get_feature_names_out()
    X = vectorizer.transform(df["description_lemmatized"])
    # TODO: Implement a better way to map the names of the features
    translator_map = {
        'csere': 'exchange',
        'elad': 'sell',
        'elektromos': 'electronic',
        'eset': 'case',
        'felszereltség': 'equipment',
        'garancia': 'guarantee',
        'gyári': 'factory',
        'gépkocsi': 'car',
        'használ': 'use',
        'hirdetés': 'ad',
        'kilométer': 'kilometer',
        'kér': 'please',
        'megtekintés': 'viewtechnical',
        'műszaki': 'system',
        'rendszer': 'service',
        'szerviz': 'owner',
        'tulajdonos': 'lead',
        'vezet': 'state',
        'állapot': 'seat'
    }

    tfidf_df = pd.DataFrame(
        X.toarray(), columns=["tfidf_" + translator_map[x].lower() for x in tfidf_feature_names]
    )
    tfidf_df.rename(
        columns={"tfidf_sports": "tfidf_sport"}, errors="ignore", inplace=True
    )

    return pd.concat([df.reset_index(drop=True), tfidf_df], axis=1)


def drop_low_cardinality_columns(
    df: pd.DataFrame, upper_threshold: float = 0.9, lower_threshold: float = 0.001
) -> pd.DataFrame:
    """
    Drops columns with extremely low or high cardinality from the dataframe.

    :param df: Input dataframe.
    :param upper_threshold: Upper threshold ratio for dropping a column. Default is 0.9.
    :param lower_threshold: Upper threshold ratio for dropping a column. Default is 0.001.
    :return: Modified dataframe without extreme cardinality columns.
    """
    categorical_features = df.select_dtypes(include=["object"]).columns
    extreme_cardinality = []

    for col in categorical_features:
        if col != "link":
            vals = df[col].dropna().value_counts()
            ratios = vals / vals.sum()
            if (ratios[0] > upper_threshold) or (ratios[0] < lower_threshold):
                extreme_cardinality.append(col)

    df.drop(columns=extreme_cardinality, inplace=True, errors="ignore")

    return df


def drop_empty_string_columns(df):
    """
    Drop columns from a DataFrame that contain any empty strings.

    :param df: Input DataFrame
    :return: DataFrame with columns containing empty strings dropped
    """
    # Find columns that contain empty strings
    cols_to_drop = [col for col in df.columns if (df[col] == '').any()]
    # Drop those columns
    df_clean = df.drop(columns=cols_to_drop)
    return df_clean


def data_procession(
    df: pd.DataFrame = None,
    output_table_name: str = "engineered_car_data",
    json_filepath: str = "../static/hun_eng_name_mapping.json",
    settlements_filepath: str = "../static/all_hun_settlement.csv",
    word_map_file: str = "../static/hun_word_map.json",
    stop_words_file: str = "../static/stopwords-hu.txt",
    vectorizer_file: str = "../models/vectorizer.joblib",
    initial_load: bool = False,
    for_prediction: bool = False,
):
    engine = setup_database()

    # For update keep the columns we choose in the initial load
    existing_columns = get_columns_names(engine, "engineered_car_data")

    if initial_load:
        query = """
            SELECT c.*
            FROM car_data c;
        """
        answer = input("Are you sure you want ot delete the current table? (yes/no)")
        if answer != "yes":
            print("Termine process")
            return

        delete_all_records(engine)
    else:
        query = """
            SELECT c.*
            FROM car_data c
            LEFT JOIN engineered_car_data e ON c.link = e.link
            WHERE e.link IS NULL;
        """
    if df is None:
        df = read_sql_query(engine, query)
    else:
        existing_columns_of_row_data = get_columns_names(engine, "car_data")
        df = drop_empty_string_columns(df)
        df = add_missing_columns(df, existing_columns_of_row_data)

    # Handle price
    df = handle_price(df)

    # String to numeric
    df = transform_columns_to_numeric(df)

    # Remove invalid performa values
    df = filter_invalid_power_values(df)

    # Comprehensive transformations
    df = convert_creation_date_to_age(df)
    df = convert_power_to_numeric(df)
    df = convert_technical_license_validity_date(df)
    df = extract_tire_data(df)

    # Feature engineering
    feature_frequencies = extract_feature_frequencies(df)
    usable_features = filter_usable_features(feature_frequencies)
    df = add_special_features(df, usable_features)

    # Drop extremely low prices
    df = filter_price_outliers(df)

    # Drop extracted and useless columns
    df = drop_unwanted_columns(df, "../static/unwanted_columns_1.csv")

    rename_dict = load_json_mapping(json_filepath)
    df = rename_dataframe_columns(df, rename_dict)

    # Add brand
    df["brand"] = extract_brand_from_link(df)

    # Add color and polish type
    df = parse_color_information(df)

    # Add city
    settlements_df = load_hungarian_settlements(settlements_filepath)
    df = extract_address_information(df, settlements_df)

    df = process_financing(df)
    df = remove_taken_away_columns(df)
    df = clean_and_process_description(df)

    # Add TF-IDF features
    word_map = load_json_mapping(word_map_file)
    df = clean_and_lemmatize_description(df, word_map)

    # list of stop words in Hungarian
    with open(stop_words_file, "r", encoding="utf-8") as f:
        stop_words = [line.strip() for line in f]

    df = apply_tfidf_vectorization(
        df, stop_words, max_features=20, vectorizer_file=vectorizer_file
    )

    # Replace empty string to nan
    df.replace("", np.nan, inplace=True)

    # Drop all the column we don't want
    if initial_load:
        # Drop useless columns
        df = drop_unwanted_columns(df, "../static/unwanted_columns_2.csv")

        # Handle missing values
        df = drop_high_nan_columns(df)

        # Drop low cardinality features
        df = drop_low_cardinality_columns(df)
        columns_to_drop_from_table = list(set(existing_columns) - set(df.columns))
        drop_columns(engine, "engineered_car_data", columns_to_drop_from_table)
    else:
        # For update keep the columns we choose in the initial load
        existing_columns = get_columns_names(engine, "engineered_car_data")
        columns_to_drop = list(set(df.columns) - set(existing_columns))
        df.drop(columns=columns_to_drop, errors="ignore")

    if for_prediction:
        return df
    else:
        # Store cleaned data
        store_to_sql(df, engine, output_table_name)


if __name__ == "__main__":
    data_procession(initial_load=False)
