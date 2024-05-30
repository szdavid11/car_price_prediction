import logging
import os
import sys
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import metrics
from catboost import CatBoostRegressor
from typing import List, Tuple, Optional

# Get the directory of the currently executing script
script_dir = os.path.dirname(os.path.realpath(__file__))

# Append it to sys.path
sys.path.append(script_dir)

from database_helpers import read_sql_query, setup_database

# Set up logging
logging.basicConfig(filename="../logs/model_training.log", level=logging.INFO)


def regression_train(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    categorical_features: List[str],
    X_test: pd.DataFrame = None,
    y_test: pd.Series = None,
    iterations: int = 10000,
    learning_rate: Optional[float] = None,
    l2_leaf_reg: int = 10,
    depth: int = 5,
) -> CatBoostRegressor:
    """
    Train a CatBoost regression model.
    :param X_train: Training features.
    :param X_test: Testing features.
    :param y_train: Training target.
    :param y_test: Testing target.
    :param categorical_features: List of categorical feature names.
    :param iterations: The maximum number of iterations. Default is 15000.
    :param learning_rate: The learning rate. If None, it will be set by the model. Default is None.
    :param l2_leaf_reg: L2 regularization term on weights. Default is 10.
    :param depth: Depth of the tree. Default is 5.

    :return: Trained model.
    """
    model = CatBoostRegressor(
        loss_function="MAE",
        depth=depth,
        iterations=iterations,
        l2_leaf_reg=l2_leaf_reg,
        learning_rate=learning_rate,
        cat_features=[x for x in categorical_features if x in X_train.columns],
        random_seed=42,
        subsample=0.5,
        bagging_temperature=10,
        colsample_bylevel=0.7,
        # boosting_type='Ordered',
        # max_bin=264,
        min_child_samples=100,
    )

    if X_test is None:
        model.fit(
            X_train,
            y_train,
            early_stopping_rounds=100,
            verbose=1000,
        )
    else:
        model.fit(
            X_train,
            y_train,
            eval_set=(X_test, y_test),
            early_stopping_rounds=100,
            verbose=100,
        )

    return model


def test_model(
    model: CatBoostRegressor,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
) -> None:
    """
    Test the trained CatBoost model and log the results.

    :param model: Trained CatBoost model.
    :param X_train: Training features.
    :param X_test: Testing features.
    :param y_train: Training target.
    :param y_test: Testing target.
    """
    test_predictions = model.predict(X_test)
    train_predictions = model.predict(X_train)

    print("Train R2: %s" % metrics.r2_score(y_train, train_predictions))
    print("Test R2: %s" % metrics.r2_score(y_test, test_predictions))
    print(
        "Train MAPE: %s",
        # metrics.mean_absolute_percentage_error(10**y_train, 10**train_predictions),
    )
    print(
        "Train MAPE original: %s",
        metrics.mean_absolute_percentage_error(y_train, train_predictions),
    )
    print(
        "Test MAPE: %s",
        # metrics.mean_absolute_percentage_error(10**y_test, 10**test_predictions),
    )
    print(
        "Test MAPE original: %s",
        metrics.mean_absolute_percentage_error(y_test, test_predictions),
    )


def feature_importance_catboost(model: CatBoostRegressor) -> pd.DataFrame:
    """
    Get feature importances from the CatBoost model.

    :param model: Trained CatBoost model.

    :return: DataFrame with feature importances.
    """
    # Get feature importances
    importance = model.get_feature_importance()
    # Combine feature names and importances into a dataframe
    feature_importance_df = pd.DataFrame(
        {"feature": model.feature_names_, "importance": importance}
    )
    # Sort the dataframe by importance in descending order
    feature_importance_df = feature_importance_df.sort_values(
        "importance", ascending=False
    )
    return feature_importance_df


def replace_less_frequent(
    ser: pd.Series, count_limit: int, max_feature_number: int
) -> pd.Series:
    """
    Replace less frequent values in a pandas Series with 'Other'.

    :param ser: Input pandas Series.
    :param count_limit: Minimum count to be considered as a separate category.
    :param max_feature_number: Maximum number of categories to keep.

    :return: pandas Series with less frequent values replaced.
    """
    # Calculate value counts
    counts = ser.value_counts()

    # Select top "max_feature_number" most frequent values
    top_values = counts.nlargest(max_feature_number).index

    # Select values which have count less than "count_limit"
    less_freq_values = counts[counts < count_limit].index

    # Values to replace: not in top values or less than "count_limit"
    to_replace = less_freq_values.union(counts.index.difference(top_values))

    if len(to_replace) == 0:
        return ser

    # Replace these values with "Other"
    ser_replaced = ser.replace(to_replace, "Other")

    # Recalculate value counts
    new_counts = ser_replaced.value_counts()

    # If "Another" count still doesn't reach "count_limit", replace the least frequent values
    while new_counts["Other"] < count_limit:
        least_frequent_value = (
            new_counts[new_counts.index != "Other"].nsmallest(1).index[0]
        )
        ser_replaced = ser_replaced.replace(least_frequent_value, "Other")
        new_counts = ser_replaced.value_counts()

    return ser_replaced


def get_train_test(
    df: pd.DataFrame,
    target_name: str,
    non_feature_cols: Optional[List[str]] = None,
    do_split: bool = True,
):
    """
    Split a dataframe into training and testing sets.

    :param df: Input dataframe.
    :param target_name: Name of the target column.
    :param non_feature_cols: List of non-feature columns to be dropped. Default is None.

    :return: X_train, X_test, y_train, y_test
    """
    if non_feature_cols is None:
        X = df.drop(columns=[target_name]).copy()
    else:
        X = df.drop(columns=non_feature_cols + [target_name]).copy()

    categorical_features = list(df.select_dtypes(include=["object"]).columns)
    if categorical_features:
        X[categorical_features] = X[categorical_features].fillna("missing")
    y = df[target_name]

    if not do_split:
        return X, y

    return train_test_split(X, y, test_size=0.2, random_state=0)


def training_process(
    output_file: str = "../models/car_price_predictor.cbm", max_feature_count: int = 50
):
    """
    Main workflow for training the model.
    """
    engine = setup_database()

    # Load data
    # ecd.link, ecd."price (HUF)",
    query = """
        SELECT 
            ecd."price (HUF)", -- target
            cof.has_model_issues, 
            cof.has_current_issues,
            cof.has_recent_fixes,
            cof.worth_price, 
            cof.price_adjustment,
            ecd."age (year)",
            ecd."power (kW)",
            ecd."clock position (km)",
            ecd."gearbox",
            ecd."brand",
            ecd."cylinder capacity (cm3)",
            ecd."design",
            ecd."own weight (kg)",
            ecd."drive",
            ecd."summer tire width",
            ecd."reversing camera",
            ecd."condition",
            ecd."trunk (l)",
            ecd."shippable persons number",
            ecd."front-rear parking radar",
            ecd."type of climate",
            ecd."MOT is valid (days)",
            ecd."fuel",
            ecd."electric rear window",
            ecd."keyless start",
            ecd."sliding door",
            ecd."esp (speed stabilizer)",
            ecd."led headlight",
            ecd."rain sensor",
            ecd."financing",
            ecd."city",
            ecd."color",
            ecd."number of doors",
            ecd."spare wheel",
            ecd."heatable window washer nozzles",
            ecd."freshly serviced",
            ecd."double-sided sliding door",
            ecd."tire pressure monitoring system",
            ecd."knee airbag",
            ecd."first placing on the market in Hungary",
            ecd."board computer",
            ecd."heated front seat"
        FROM engineered_car_data ecd
        INNER JOIN car_openai_features cof on ecd.link = cof.link 
        WHERE "price (HUF)" > 1000000 and "price (HUF)" < 100000000
    """
    print("Load data")

    df = read_sql_query(engine, query)
    print("Count of rows:", df.shape[0])

    # Drop columns
    # df = df.drop(columns=["link"]+tfidf_columns)

    categorical_features = list(df.select_dtypes(include=["object"]).columns)
    non_categorical_features = list(df.select_dtypes(exclude=["object"]).columns)
    df[non_categorical_features] = df[non_categorical_features].astype(float)

    target = "price (HUF)"
    # target_log = "price log"
    # df[target_log] = np.log10(df[target])

    print("Handle high cardinality features")
    # Handle categorical values that are less frequent
    for col in categorical_features:
        if col == "city":
            df[col] = replace_less_frequent(df[col], 500, 10)
        else:
            df[col] = replace_less_frequent(df[col], 1000, 10)

    print("Train the first model")
    X_train, X_test, y_train, y_test = get_train_test(df, target, [target])
    model = regression_train(
        X_train, y_train, categorical_features, X_test, y_test, iterations=10000
    )

    # Test first model
    print("All feature model")
    test_model(model, X_train, X_test, y_train, y_test)

    X, y = get_train_test(df, target, [target], do_split=False)

    # Top feature model
    print("Train the second model")
    model2 = regression_train(X, y, categorical_features, iterations=10000)

    # Select top features
    df_feature_importance = feature_importance_catboost(model2)
    top_features = list(df_feature_importance["feature"].values[:max_feature_count])
    print("Top features:", top_features)

    # Save model
    os.remove(output_file)
    model2.save_model(output_file)  # """


if __name__ == "__main__":
    training_process()
