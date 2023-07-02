# Used Car Price Prediction

This project aims to predict the price of used cars listed on a hungarian website www.hasznaltauto.hu. The prediction model is built using various machine learning techniques including CatBoostRegressor for handling categorical features, and feature engineering for improving model performance.

## Project Overview

The project includes various stages:

1. **Data Collection:** Data is collected from www.hasznaltauto.hu. A Python script is used to scrape the car listing links and details, and the collected data is stored in a CSV file.

2. **Data Cleaning & Feature Engineering:** The collected data is then cleaned and processed for the machine learning model. This includes handling missing values, outliers, and bucketing of rare categories in categorical features.

3. **Model Training & Validation:** A CatBoostRegressor model is trained using the processed data. Hyperparameter tuning and validation strategies such as K-Fold Cross Validation are implemented for model selection.

4. **Model Evaluation:** The performance of the model is evaluated using suitable metrics such as Mean Absolute Error (MAE). The predictions of the model are compared with the actual values through various visualization techniques.

5. **Model Optimization:** The model is further optimized using automated machine learning techniques with Google Cloud AutoML.

## Requirements

This project requires the following Python libraries:

- pandas
- numpy
- seaborn
- requests
- beautifulsoup4
- catboost
- sklearn

## Usage

Ensure that you have the necessary Python packages installed. Use the Python scripts provided in the repository for various tasks like data collection, preprocessing, model training, etc.

## Contributing

Contributions, issues, and feature requests are welcome. Feel free to check issues page if you want to contribute.
