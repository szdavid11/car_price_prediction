import re
import sys
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from catboost import CatBoostRegressor
import numpy as np
from fastapi.responses import JSONResponse

sys.path.append("../")
from pipelines.scraper import scrape_car_data
from pipelines.preprocess_pipeline import data_processing
from pipelines.database_helpers import read_sql_query, setup_database
from pipelines.get_openai_features import OpenAIFeatures
from sqlalchemy import MetaData, Table

from fastapi.middleware.cors import CORSMiddleware
import shap
import matplotlib.pyplot as plt
from fastapi.responses import FileResponse

app = FastAPI()

# Add CORS middleware
origins = [
    "https://szalaidatasolutions.online",  # Allow your hostinger domain
    "http://localhost",  # Allow requests from localhost (for local testing)
    "https://builder.hostinger.com",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the model
model = CatBoostRegressor()
model.load_model("../models/car_price_predictor.cbm")

# Initialize explainer (do this only once after loading the model)
explainer = shap.Explainer(model)

# Use SQLAlchemy to reflect the table structure
metadata = MetaData()
engine = setup_database()


class CarLink(BaseModel):
    link: str


def add_openai_features(
    df_scraped: pd.DataFrame, df_engineered: pd.DataFrame
) -> pd.DataFrame:
    # Add openai features
    df_all_data = pd.merge(df_scraped, df_engineered, on="link")

    # TODO: Read from the database
    openai_features = OpenAIFeatures(df_all_data).process_first_row()
    for key, value in openai_features.items():
        df_engineered[key] = value

    return df_engineered


def save_shap_waterfall(df_processed, link, max_display=20):
    # Get SHAP values for the instance
    shap_values = explainer(df_processed[model.feature_names_])

    shap_one = shap_values[0]
    new_base = shap_one.base_values
    new_values = []
    current_value = shap_one.base_values

    for val in shap_one.values:
        diff = val
        new_values.append(diff)
        current_value = current_value + val

    # Round the numbers
    new_values = ((np.array(new_values) / 1000).round() * 1000).astype(int)
    new_base = int(round(new_base / 1000)) * 1000

    # If one of the feature is 'has_model_issues'
    # then change the name and value to `model_issues_detail` value of df_processed
    issue_col = "has_model_issues"
    if issue_col in model.feature_names_:
        model_issue_idx = model.feature_names_.index(issue_col)
        if df_processed[issue_col].values[0]:
            shap_one.data[model_issue_idx] = df_processed["model_issues_detail"].values[
                0
            ]
            model.feature_names_[model_issue_idx] = "model_issues_detail"

    new_shap_exp = shap._explanation.Explanation(
        values=new_values,
        base_values=new_base,
        data=shap_one.data,
        feature_names=model.feature_names_,
    )

    # Create a waterfall plot
    fig, ax = plt.subplots(figsize=(20, 5))  # Adjust the size for better visualization
    shap.plots.waterfall(
        new_shap_exp, max_display=max_display, show=False
    )  # Increase max_display to max_display
    y_labels = ax.get_yticklabels()

    # Adjust the below y-label extraction for max_display features
    values = np.array(
        pd.Series([x.get_text() for x in y_labels[1:max_display]])
        .str.split(" = ")
        .to_list()
    )
    revers_values = pd.Series(values[:, 1]) + " = " + pd.Series(values[:, 0])

    usd = [None] * max_display + [y_labels[0].get_text()] + list(revers_values)
    ax.set_yticklabels(usd, ha="left", x=-0.55)  # Adjust the 'x' value as needed

    # Save plot
    name_tag = re.sub("#sid.*", "", link.split("/")[-1])
    png_file_name = f"shap_waterfall_{name_tag}.png"
    plt.savefig("shap-images/" + png_file_name, bbox_inches="tight")

    return png_file_name


def prediction_process(link: str) -> tuple[int, int, str]:
    """
    Predicts the car price from the link.
    :param link: The link of the car on hasznaltauto.hu
    :return: The predicted price of the car
    """
    link = link.strip()
    # Remove sid from the link
    link = re.sub("#sid.*", "", link)

    table = Table("engineered_car_data", metadata, autoload_with=engine)
    bool_columns = [
        col.name for col in table.columns if str(col.type).lower() == "boolean"
    ]
    cat_features_indices = model.get_cat_feature_indices()
    cat_features = pd.Series(model.feature_names_).loc[cat_features_indices].values

    # Ensure the model has been loaded correctly
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")

    # Scrape data
    df_scraped = scrape_car_data(link)

    # Ensure the model has been loaded correctly
    if df_scraped is None:
        raise HTTPException(status_code=500, detail="Data not scraped")

    # Process data
    df_processed = data_processing(df_scraped, for_prediction=True)

    # Add openai features
    df_processed = add_openai_features(df_scraped, df_processed)

    # Convert bool columns to float
    existing_bools = df_processed.select_dtypes(include=["bool"]).columns
    existing_bools = [col for col in existing_bools if col in bool_columns]
    existing_bools = [col for col in existing_bools if col not in cat_features]
    df_processed[existing_bools] = df_processed[existing_bools].astype(float)

    # Check for missing columns and add them with appropriate default values
    for idx, col in enumerate(model.feature_names_):
        if (col not in df_processed.columns) or (df_processed[col].values[0] is None):
            # Fill missing categorical features with 'missing' and numerical features with np.nan
            if idx in cat_features_indices:
                df_processed[col] = "missing"
            elif col in bool_columns:
                df_processed[col] = 0.0
            else:
                df_processed[col] = np.nan

    # Make prediction
    prediction = model.predict(df_processed[model.feature_names_])[0]
    df_processed.to_parquet("processed.parquet")
    png_file_name = save_shap_waterfall(df_processed, link)

    return (
        int(round(prediction / 1000) * 1000),
        int(df_processed["price (HUF)"].values[0]),
        png_file_name,
    )


@app.get("/shap-image/{file_name}")
async def get_shap_image(file_name: str):
    return FileResponse("shap-images/" + file_name, media_type="image/png")


@app.get("/best-deals/{number_of_urls}", response_class=JSONResponse)
async def get_some_good_deals(number_of_urls: int):
    """
    Returns the best deals from the database.
    """

    query = f"""
        SELECT link as "URL", 
        predicted_price as "Predicted price", 
        original_price as "Original price"
        FROM (
            SELECT pp.link, predicted_price, ecd."price (HUF)" as original_price, 
            ((1.0 + cof.price_adjustment::float/100) * predicted_price) - ecd."price (HUF)" / ecd."price (HUF)" as price_difference_ratio
            FROM predicted_prices pp
            LEFT join car_links cl on pp.link = cl.link
            LEFT join engineered_car_data ecd on pp.link = ecd.link
            LEFT join car_openai_features cof on pp.link = cof.link
            WHERE pp.predicted_price > ecd."price (HUF)"
            AND cl.estimated_sold_date is NULL
            AND cl.collected_at > now() - interval '7 days'
            AND ecd."price (HUF)" < 10000000
            AND ecd."price (HUF)" > 1000000
            AND cof.price_adjustment::int > 1
        ) foo
        ORDER BY price_difference_ratio DESC 
        LIMIT {number_of_urls}
    """

    df = read_sql_query(engine, query)
    df["Predicted price"] = (1000 * (df["Predicted price"] / 1000).astype(int)).astype(
        str
    ) + " HUF"
    df["Original price"] = (1000 * (df["Original price"] / 1000).astype(int)).astype(
        str
    ) + " HUF"

    # Convert DataFrame to HTML
    return JSONResponse({"data": df.to_dict(orient="records")})


@app.post("/predict/")
async def predict_car_price(car_link: CarLink):
    """
    Predicts the car price from the link provided.
    """
    link = car_link.link
    prediction, original, saved_plot_path = prediction_process(link)
    return {
        "predicted_price": prediction,
        "original_price": original,
        "plot_path": saved_plot_path,
    }


if __name__ == "__main__":
    link = (
        "https://www.hasznaltauto.hu/szemelyauto/bmw/750/bmw_750_active_hibrid-20741687"
    )
    price = prediction_process(link)
    print(price)
    """
    df_asd = pd.read_parquet("processed.parquet")
    df_asd["has_model_issues"] = True
    df_asd["model_issues_detail"] = "Nem indul"
    save_shap_waterfall(df_asd, link)"""
