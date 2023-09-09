import pandas as pd
import numpy as np
from catboost import CatBoostRegressor

from sklearn.model_selection import train_test_split
from sklearn import metrics
import seaborn as sns
import matplotlib.pyplot as plt

from data_pipeline import read_sql_query, setup_database


def main():
    engine = setup_database()

    # Load
    query = """
        SELECT *
        FROM engineered_car_data
    """

    df = read_sql_query(engine, query)
    df = df.replace("", np.nan)

    # Drop high nans
    new_nan_mean = df.isna().mean()
    high_nans = new_nan_mean[new_nan_mean > 0.5].keys()
    print("High nans", high_nans)
    df.drop(columns=high_nans, inplace=True, errors="ignore")

    # Mainly 1 value
    categorical_features = df.select_dtypes(include=["object"]).columns
    extramely_low_cardinlity = []
    print("Extramely low cardinality")
    for col in categorical_features:
        vals = df[col].dropna().value_counts()
        ratios = vals / vals.sum()
        if ratios[0] > 0.9:
            print(col, ratios[0])
            extramely_low_cardinlity.append(col)

    df.drop(columns=extramely_low_cardinlity, inplace=True, errors="ignore")

    cols = df.columns

    numeric_features = df.select_dtypes(include=["float", "int"]).columns
    tfid_features = list(cols[cols.str.contains("tfid")])
    numeric_features = [
        x for x in numeric_features if x not in tfid_features + ["word_count"]
    ]

    binary_features = df.select_dtypes(include=["bool"]).columns
    categorical_features = list(df.select_dtypes(include=["object"]).columns)

    non_categorical_features = list(numeric_features) + list(binary_features)
    df[non_categorical_features] = df[non_categorical_features].astype(float)

    df.head()


if __name__ == "__main__":
    main()
