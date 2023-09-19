import re
import sys

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from catboost import CatBoostRegressor
import numpy as np

sys.path.append('../')
from pipelines.scraper import scrape_car_data
from pipelines.data_pipeline import data_procession

from fastapi.middleware.cors import CORSMiddleware
import shap
import matplotlib.pyplot as plt
from fastapi.responses import FileResponse

app = FastAPI()

# Add CORS middleware
origins = [
    "https://szalaidatasolutions.online",  # Allow your hostinger domain
    "http://localhost",  # Allow requests from localhost (for local testing)
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

    # Create a waterfall plot
    plt.figure()
    shap.plots.waterfall(shap_values[0])

    # Save plot
    name_tag = re.sub("#sid.*", "", link.split('/')[-1])
    png_file_name = f"shap_waterfall_{name_tag}.png"
    plt.savefig(png_file_name)

    return png_file_name


def prediction_process(link: str) -> tuple[int, int, str]:
    """
    Predicts the car price from the link.
    :param link: The link of the car on hasznaltautok.hu
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
    df_processed = data_procession(df_scraped, for_prediction=True)

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

    return int(10 ** prediction), int(df_processed['price (HUF)'].values[0]), png_file_name

@app.get("/shap-image/{file_name}")
async def get_shap_image(file_name: str):
    file_path = f"./{file_name}"  # Adjust the path if saved elsewhere
    return FileResponse(file_path, media_type="image/png")


@app.post("/predict/")
def predict_car_price(car_link: CarLink):
    """
    Predicts the car price from the link provided.
    """
    link = car_link.link
    prediction, original, saved_plot_path = prediction_process(link)
    return {
        "predicted_price": prediction,
        "original_price": original,
        "shap_image_url": f"/shap-image/{saved_plot_path}"
    }


if __name__ == '__main__':
    prediction_process(
        "https://www.hasznaltauto.hu/szemelyauto/mercedes-benz/eqs/mercedes-benz_eqs_580_4matic_afa-s_2000_km-19005067"
    )
