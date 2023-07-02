import os
import re
import pickle
import ast
import requests
from datetime import datetime
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np


def scrape_car_data(link):
    """
    Scrape car data from a link
    :param link: link of the car in hasznaltauto.hu
    :return: tuple of:
        - advertisement data which is a dictionary
        with well-defined keys as it contains necessary information about the vehicle.
        - special information about the given car it can be anything without a well-defined stricture.
    """
    try:
        # Get the html
        response = requests.get(link)
        soup = BeautifulSoup(response.content, "html.parser")

        header = soup.find_all("div", {"class": "data-relay-banner"})

        # If there is no header that means the car is not available anymore
        if len(header) == 0:
            return None, None

        # Get the header information
        # It is in a dictionary format but as a string, so we need to convert it
        header = ast.literal_eval(
            header[0]["data-vehicle-information-banner-parameters"]
        )

        # Get the data from the base advertisement information table
        advertisement_data = soup.find_all("table", {"class": "hirdetesadatok"})

        # Get the text from xml
        # Separator is # because it is not in the text and without it
        # the words would be merged without spaces.
        advertisement_data = advertisement_data[0].get_text(separator="#")

        # Clean the text
        advertisement_data = (
            advertisement_data.strip()
        )  # Remove leading and trailing spaces
        advertisement_data = re.sub(
            r"#\n", "", advertisement_data
        )  # Remove new lines without any text

        # Get lines by splitting on the defined separator #
        advertisement_data = advertisement_data.split("#")
        advertisement_data = pd.Series(advertisement_data)

        # Add header to advertisement data
        advertisement_data.update(header)

        keys_id = advertisement_data.str.contains(":$")
        values_id = np.where(keys_id)[0] + 1
        keys = advertisement_data[keys_id]
        values = advertisement_data[values_id]
        advertisement_data = dict(zip(keys, values))

        # Get equipment info
        equipments = soup.find_all("div", {"class": "row felszereltseg"})
        if equipments:
            equipments = equipments[0].text.split("\n")

        # Get other info
        other = soup.find_all("div", {"class": "egyebinformacio"})
        if other:
            other = other[0].text.split("\n")

        # Get description
        description = soup.find_all("div", {"class": "leiras"})
        if description:
            description = [description[0].text]

            # Get oll special info about the car and clean it
        special_car_info = equipments + other + description
        if special_car_info:
            special_car_info = pd.Series(special_car_info)
            special_car_info = special_car_info.str.strip()
            special_car_info = special_car_info.str.lower()
            special_car_info = special_car_info.dropna()
            special_car_info = special_car_info[~special_car_info.isin(["", "-"])]

        return advertisement_data, special_car_info
    except Exception as e:
        print(e)
        return None, None


class CarDataScraper:
    # Base URL is get from after I set the proper search parameters
    BASE_URL = "https://www.hasznaltauto.hu/talalatilista/PCOG2VG3R3RDADH5S56ADN4Z3EEY7O2CQO2EEI5NGSXZLINGQROIZEVUDJAPZ6Z2NVQZVCVHLLDY47L4OJJJB43XPEGXEOUTIJB4JCV4QIGTMYWJ3BKIU4CBRPTAY5NECZNEQAJGF7OPJ6CB7EJCHBQGJIEBPMO4ZJE57JDB3CAC7LDSVHEUICRRZ3APOMHTEIC5NPWIZZWZ4JOSWHDXZHXKYDCOECG62ZWR6KNG5X3U3CQKHS4J2AVDYKJYEWO2CQGLSJZZT24ULD7GQCI3F5FE7W7EXQP4FCUIQMIW64YWGWLIAXDQRRZ3Z2F646IDCZVF2UUPJUP5QYHBEF4F7FVX6EMGUE2DEIYGPHUTR3FK74JF7QHH575DDQF7IQIW76QW7VAOIOTNZZ7B5TEX3KISPM5DHHV4VNBSJ62XQX4OTSKSKTSNMZMXPZWU67NNG6MVJLEFBIV57HKRVZNCSE6RO4TZRHAFV6QVA7MLEQOJK2GMAPQZFKCOOU4KEGPZAQNRMHTCF3GAGHAIUNK4GF3B7IRL5VR5HM3UM3OH43FVFGP3TBL43LQHO6DOVUIXRSARGRNRAT4SFL5KWN3LXNUS2MXJ5JLY23C7GFGPXN4JDYSFC67YRZMNAO7LGA52BJ663Y4YLCPMXXYFO5QBXEJO3CWBM7UWVQGQNTCQL2BFKNDIF6SBG2NOXK3AFJ4XNQQ4FUGDLRGCCXBUXNOA74OXOSO3IV67THQRSRMDHITEXBACWYDLEQ3JQ2U5TNCORXQIR5UUMZT53WUXS4OHTUPEZ3GTBR2KMA4I3CBW4TUEBHVKEH7NN7KJ7DUQ2UNWXQ3MUZVLUZPQCT2DO7DFB3GJK5QOZAHHVUJZTLP5Q53LKRJXEB57WD3X72QGI/page"

    def __init__(
        self,
        link_collection_file="data/car_links.csv",
        scraped_data_file="data/car_data.pickle",
    ):
        """
        Scrapes the data from the hasznaltauto.hu website
        :param link_collection_file: Output file for the links of the cars.
        File extension must be .csv
        :param scraped_data_file: Output file for all the scraped cars data.
        File extension must be .pickle
        """
        self.link_collection_file = link_collection_file
        self.scraped_data_file = scraped_data_file

        self.existing_links = self.load_existing_links()

    def get_links(self, page_number):
        """
        Get the links of the cars from the search page
        :param page_number:
        :return:
        """
        url = self.BASE_URL + str(page_number)
        response = requests.get(url)
        soup = BeautifulSoup(response.content, "html.parser")
        matches = soup.find_all("a", {"class": ""})

        # Extract the hrefs from matches
        hrefs = pd.Series(
            [match["href"] for match in matches if match.has_attr("href")]
        )

        # Keep only those that a like for a car
        hrefs = hrefs[
            hrefs.str.contains("www.hasznaltauto.hu/szemelyauto", regex=False)
        ].to_list()

        return hrefs

    def load_existing_links(self):
        if not os.path.exists(self.link_collection_file):
            print("No existing links found")
            return []

        df = pd.read_csv(self.link_collection_file)

        print(f"Loaded {len(df)} existing links")
        return df["car_links"].values

    def get_all_links(self) -> pd.Series:
        """
        Get all links from the website
        :return:
        """
        all_links = []
        page_number = 1
        while True:
            links_on_page = self.get_links(page_number)
            if not links_on_page:
                break
            page_number += 1
            all_links.append(links_on_page)

        all_links = pd.Series(np.concatenate(all_links))
        all_links = all_links.str.replace("#sid.*", "")
        return all_links

    def get_new_links(self) -> pd.Series:
        """
        Get the new links that are not in the existing links
        :return: series of new links
        """
        all_hrefs = self.get_all_links()

        # Extract the hrefs from page results
        new_links = all_hrefs[~all_hrefs.isin(self.existing_links)]
        new_links = new_links.drop_duplicates()

        return new_links

    def update_links(self) -> pd.Series:
        """
        Update the links in the csv file
        :return: series of new links
        """
        # Get the new links
        new_links = self.get_new_links()

        # Create a dataframe from the new links
        df_hrefs = pd.DataFrame({"car_links": list(new_links)})
        df_hrefs["collected_at"] = str(datetime.today())[:19]

        # Make sure that the links are unique
        df_hrefs.drop_duplicates(inplace=True)

        # Append to the existing file
        df_hrefs.to_csv(self.link_collection_file, index=False, mode="a")

        return new_links

    def collect_car_data(self):
        """
        Collect the data of the cars from the new links and save it to a pickle file
        :return:
        """
        print("Update links of the cars. This may take a while. (~30 minutes)")
        new_links = self.update_links()

        print("Collect data of the cars. This may take a while. (~3 hours)")
        print("Number of new cars:", len(new_links))
        res_list = []
        for i, link in enumerate(new_links):
            advertisement_data, special_car_info = scrape_car_data(link)
            res_list.append([advertisement_data, special_car_info, link])
            if i % 100 == 0:
                print("Number of cars scraped:", i)
                with open(self.scraped_data_file, mode="ab") as file:
                    pickle.dump(res_list, file)


if __name__ == "__main__":
    scraper = CarDataScraper()
    scraper.collect_car_data()
