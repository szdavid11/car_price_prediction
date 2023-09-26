import re
import sys
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from catboost import CatBoostRegressor
import numpy as np
from fastapi.responses import HTMLResponse

sys.path.append('../')
from pipelines.scraper import scrape_car_data
from pipelines.preprocess_pipeline import data_processing
from pipelines.database_helpers import read_sql_query, setup_database

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
model.load_model('../models/car_price_predictor.cbm')

# Initialize explainer (do this only once after loading the model)
explainer = shap.Explainer(model)


class CarLink(BaseModel):
    link: str


def save_shap_waterfall(df_processed, link):
    # Get SHAP values for the instance
    shap_values = explainer(df_processed[model.feature_names_])

    shap_one = shap_values[0]
    new_base = 10 ** shap_one.base_values
    new_values = []
    current_value = shap_one.base_values

    for val in shap_one.values:
        diff = (10 ** (current_value + val)) - (10 ** current_value)
        new_values.append(diff)
        current_value = current_value + val

    # Round the numbers
    new_values = ((np.array(new_values) / 1000).round() * 1000).astype(int)
    new_base = int(round(new_base / 1000)) * 1000

    new_shap_exp = shap._explanation.Explanation(
        values=new_values,
        base_values=new_base,
        data=shap_one.data,
        feature_names=model.feature_names_
    )

    # Create a waterfall plot
    fig, ax = plt.subplots(figsize=(17, 5))
    shap.plots.waterfall(new_shap_exp, show=False)
    asd = ax.get_yticklabels()
    values = np.array(pd.Series([x.get_text() for x in asd[1:10]]).str.split(' = ').to_list())
    revers_values = pd.Series(values[:, 1]) + " = " + pd.Series(values[:, 0])

    usd = [None] * 10 + [asd[0].get_text()] + list(revers_values)
    ax.set_yticklabels(usd, ha='left', x=-0.55)  # Adjust the 'x' value as needed

    # Save plot
    name_tag = re.sub("#sid.*", "", link.split('/')[-1])
    png_file_name = f"shap_waterfall_{name_tag}.png"
    plt.savefig("shap-images/" + png_file_name)

    return png_file_name


def prediction_process(link: str) -> tuple[int, int, str]:
    """
    Predicts the car price from the link.
    :param link: The link of the car on hasznaltauto.hu
    :return: The predicted price of the car
    """

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

    # Check for missing columns and add them with appropriate default values
    cat_features_indices = model.get_cat_feature_indices()
    for idx, col in enumerate(model.feature_names_):
        if (col not in df_processed.columns) or (df_processed[col].values[0] is None):
            # Fill missing categorical features with 'missing' and numerical features with np.nan
            if idx in cat_features_indices:
                df_processed[col] = 'missing'
            else:
                df_processed[col] = np.nan

    # Make prediction
    prediction = model.predict(df_processed[model.feature_names_])[0]
    png_file_name = save_shap_waterfall(df_processed, link)

    return int(round(10 ** prediction / 1000) * 1000), int(df_processed['price (HUF)'].values[0]), png_file_name


@app.get("/shap-image/{file_name}")
async def get_shap_image(file_name: str):
    return FileResponse('shap-images/' + file_name, media_type="image/png")


@app.get("/best-deals/{number_of_urls}", response_class=HTMLResponse)
async def get_some_good_deals(number_of_urls: int):
    """
    Returns the best deals from the database.
    """
    engine = setup_database()

    query = f"""
        SELECT link as "URL", 
        predicted_price as "Predicted price", 
        original_price as "Original price"
        FROM (
            SELECT pp.link, predicted_price, ecd."price (HUF)" as original_price, 
            predicted_price - ecd."price (HUF)" as price_difference
            FROM predicted_prices pp
            LEFT join car_links cl on pp.link = cl.link
            LEFT join engineered_car_data ecd on pp.link = ecd.link
            WHERE pp.predicted_price > ecd."price (HUF)"
            AND cl.estimated_sold_date is NULL
        ) foo
        ORDER BY price_difference DESC 
        LIMIT {number_of_urls}
    """
    df = read_sql_query(engine, query)
    df["Predicted price"] = (1000*(df["Predicted price"]/1000).astype(int)).astype(str) + " HUF"
    df["Original price"] = (1000*(df["Original price"]/1000).astype(int)).astype(str) + " HUF"

    # Convert DataFrame to HTML
    return df.to_html()


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
        "plot_path": saved_plot_path
    }


if __name__ == '__main__':
    prediction_process(
        "https://www.hasznaltauto.hu/szemelyauto/mercedes-benz/eqs/mercedes-benz_eqs_580_4matic_afa-s_2000_km-19005067"
    )
