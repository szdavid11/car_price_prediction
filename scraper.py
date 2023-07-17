import os
import pickle
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
        # Get the html content of the link
        response = requests.get(link)

        # Parse the html content
        soup = BeautifulSoup(response.content, 'html.parser')

        # Get the data from the base advertisement information table
        advertisement_data = soup.find_all("table", {"class": "hirdetesadatok"})

        # If there is no advertisement_data that means the car is not available anymore
        if len(advertisement_data) == 0:
            return None, None

        # Get the table in html format
        table_html = str(soup.find_all('table', {'class': 'hirdetesadatok'})[0])

        # Read the table with pandas and
        advertisement_data = pd.read_html(table_html)[0]

        # Clean the data
        advertisement_data.columns = ['key', 'value']
        # Remove the rows where the key is not contains ':' at the end
        # This is because the table contains some other information which is not irrelevant,
        # not real key value pairs
        advertisement_data = advertisement_data[advertisement_data['key'].str.contains(':$', regex=True)]

        # Transform dataframe key to columns and values to rows
        # This will be easier to work with as we collect the car data inta rows of a dataframe
        advertisement_data = advertisement_data[advertisement_data['key'].str.contains(':$', regex=True)]
        advertisement_data = advertisement_data.T
        advertisement_data.reset_index(drop=True, inplace=True)
        advertisement_data.columns = advertisement_data.iloc[0]
        advertisement_data = advertisement_data[1:]

        # Seller information
        h4_headers = [x.text for x in soup.find_all("h4")]
        # Decide if the seller is a shop or private person
        advertisement_data["buy_from_shop"] = 'Kereskedés adatai' in h4_headers

        # Sale contact
        contacts = soup.find_all('span', {"class": "contact-button-text"})
        for i in range(len(contacts)):
            advertisement_data[f"content_info_{i}"] = contacts[i].text

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

        # Load data from the pickle file if it exists
        if os.path.exists(self.scraped_data_file):
            with open(self.scraped_data_file, mode="rb") as file:
                self.res_list = pickle.load(file)
            missing_values = pd.Series([x[0] is None for x in self.res_list])
            links = pd.Series([x[2] for x in self.res_list])[~missing_values]

            self.df_existing_links = pd.DataFrame({"car_link": links})

            if os.path.exists(self.link_collection_file):
                df_saved_links = pd.read_csv(self.link_collection_file)
                self.df_existing_links = pd.merge(
                    self.df_existing_links, df_saved_links, on="car_link", how='left'
                )
        else:
            self.res_list = []
            self.df_existing_links = None

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

        # Keep only those that a link for a car
        hrefs = hrefs[
            hrefs.str.contains("www.hasznaltauto.hu/szemelyauto", regex=False)
        ].to_list()

        return hrefs

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
        all_links = all_links.str.replace("#sid.*", "", regex=True)
        return all_links

    def get_new_links(self) -> pd.Series:
        """
        Get the new links that are not in the existing links
        :return: series of new links
        """
        all_hrefs = self.get_all_links()

        if self.df_existing_links is not None:
            # Add estimated sold date to the existing links if they are not available anymore
            is_still_exists = (self.df_existing_links["car_link"]).isin(all_hrefs)
            if 'estimated_sold_date' not in self.df_existing_links.columns:
                self.df_existing_links['estimated_sold_date'] = np.nan

            already_sold = ~self.df_existing_links["estimated_sold_date"].isna()
            self.df_existing_links['estimated_sold_date'] = np.where(
                (~is_still_exists) & (~already_sold),
                str(datetime.today())[:19],
                self.df_existing_links['estimated_sold_date']
            )

            # Extract the hrefs from page results
            new_links = all_hrefs[~all_hrefs.isin(self.df_existing_links["car_link"])]
        else:
            new_links = all_hrefs

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
        df_new_hrefs = pd.DataFrame({"car_link": list(new_links)})
        df_new_hrefs["collected_at"] = str(datetime.today())[:19]

        # Make sure that the links are unique
        df_new_hrefs.drop_duplicates(inplace=True)

        if self.df_existing_links is None:
            df_hrefs = df_new_hrefs
        else:
            df_hrefs = pd.concat([self.df_existing_links, df_new_hrefs])

        # Update the links in the csv file
        df_hrefs.to_csv(self.link_collection_file, index=False)

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

        success_count = 0
        for i, link in enumerate(new_links):
            advertisement_data, special_car_info = scrape_car_data(link)
            if advertisement_data is not None:
                self.res_list.append([advertisement_data, special_car_info, link])
                success_count += 1

            if i % 1000 == 0:
                # Backup safe the data
                print("Number of already scraped cars:", success_count)
                with open(self.scraped_data_file, mode="wb") as file:
                    pickle.dump(self.res_list, file)

        print("Number of total cars scraped:", success_count)
        with open(self.scraped_data_file, mode="wb") as file:
            pickle.dump(self.res_list, file)


if __name__ == "__main__":
    scraper = CarDataScraper()
    scraper.collect_car_data()
