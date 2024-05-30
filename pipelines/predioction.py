import logging
import pandas as pd
import numpy as np
from catboost import CatBoostRegressor
from database_helpers import read_sql_query, setup_database

# Setting up logging
logging.basicConfig(
    filename="../logs/scraping.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)


def write_predictions_into_database() -> None:
    """
    Write the predictions into the database.

    :param df: Dataframe containing the predictions.
    :return: None
    """

    # Get the database connection
    engine = setup_database()

    # Load the model
    model = CatBoostRegressor()
    model.load_model("../models/car_price_predictor.cbm")

    # Read new data from a database
    query = f"""
    SELECT ecd.link, "{'", "'.join(model.feature_names_)}"
    FROM engineered_car_data ecd
    LEFT  JOIN predicted_prices pp on pp.link = ecd.link
    INNER JOIN car_openai_features oaf on oaf.link = ecd.link
    WHERE  pp.predicted_price IS NULL;
    """

    df = read_sql_query(engine, query)

    # If there is no new data, return
    if len(df) == 0:
        return

    print("Number of new data", len(df))

    # Final changes on the dataset
    df.set_index("link", inplace=True)
    cat_features_indices = model.get_cat_feature_indices()
    categorical_features = (
        pd.Series(model.feature_names_).loc[cat_features_indices].values
    )
    non_categorical_features = [
        col for col in df.columns if col not in categorical_features
    ]
    df[non_categorical_features] = df[non_categorical_features].astype(float)
    # Handle missing values
    for col in model.feature_names_:
        if col in categorical_features:
            df[col] = df[col].fillna("missing")
        else:
            df[col] = df[col].fillna(np.nan)

    # Predict the prices
    predictions = model.predict(df[model.feature_names_])

    # If less than 0, set it to 0
    predictions = [max(0, prediction) for prediction in predictions]

    # Write the predictions into the database
    df_predictions = pd.DataFrame({"link": df.index, "predicted_price": predictions})
    df_predictions["predicted_price"] = (df_predictions["predicted_price"]).astype(int)
    df_predictions.to_sql(
        "predicted_prices", con=engine, if_exists="append", index=False
    )


if __name__ == "__main__":
    write_predictions_into_database()
