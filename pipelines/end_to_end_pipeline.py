from scraper import CarDataScraper
from preprocess_pipeline import data_processing
from predioction import write_predictions_into_database


if __name__ == '__main__':
    scraper = CarDataScraper()
    scraper.collect_car_data()
    data_processing(initial_load=False)
    write_predictions_into_database()
