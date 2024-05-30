from scraper import CarDataScraper
from preprocess_pipeline import data_processing
from predioction import write_predictions_into_database
from get_openai_features import OpenAIFeatures


if __name__ == "__main__":
    scraper = CarDataScraper()
    scraper.collect_car_data()
    data_processing(initial_load=False)
    OpenAIFeatures().process_batch_files()

    write_predictions_into_database()
