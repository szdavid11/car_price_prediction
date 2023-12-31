import requests
import re
import sys
import os
import logging
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
from sqlalchemy import MetaData, text as sql_text
from io import StringIO

# Get the directory of the currently executing script
script_dir = os.path.dirname(os.path.realpath(__file__))

# Append it to sys.path
sys.path.append(script_dir)

from database_helpers import (
    execute_query,
    setup_database,
    read_sql_query
)

# Setting up logging
logging.basicConfig(
    filename="../logs/scraping.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

engine = setup_database()
metadata = MetaData()
metadata.reflect(bind=engine)

HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) '
                  'Chrome/91.0.4472.124 Safari/537.36'
}


# PROXY_NAME = f"{os.getenv('BRIGHT_USER')}:{os.getenv('BRIGHT_PASSWORD')}@brd.superproxy.io:22225"


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
        response = requests.get(link, headers=HEADERS)  # proxies={'http': PROXY_NAME, 'https': PROXY_NAME})

        # Parse the html content
        soup = BeautifulSoup(response.content, "html.parser")

        # Get the data from the base advertisement information table
        advertisement_data = soup.find_all("table", {"class": "hirdetesadatok"})

        # If there is no advertisement_data that means the car is not available anymore
        if len(advertisement_data) == 0:
            return None

        history = soup.find_all("div", {"class": "car-history-card__cta"})
        if history:
            try:
                chassis_number = re.findall(
                    r"elozetes-ellenorzes\?vin=(.+?)\&amp", str(history[0])
                )[0]
            except Exception as e:
                logging.info(e)
                chassis_number = None
        else:
            chassis_number = None

        # Get the table in html format
        table_html = str(soup.find_all("table", {"class": "hirdetesadatok"})[0])

        # Read the table with pandas and
        advertisement_data = pd.read_html(StringIO(table_html))[0]

        # Clean the data
        advertisement_data.columns = ["key", "value"]
        # Remove the rows where the key is not contains ':' at the end
        # This is because the table contains some other information which is not irrelevant,
        # not real key value pairs
        advertisement_data = advertisement_data[
            advertisement_data["key"].str.contains(":$", regex=True)
        ]

        # Transform dataframe key to columns and values to rows
        # This will be easier to work with as we collect the car data inta rows of a dataframe
        advertisement_data = advertisement_data[
            advertisement_data["key"].str.contains(":$", regex=True)
        ]
        advertisement_data = advertisement_data.T
        advertisement_data.reset_index(drop=True, inplace=True)
        advertisement_data.columns = advertisement_data.iloc[0]
        advertisement_data = advertisement_data[1:]

        # Seller information
        h4_headers = [x.text for x in soup.find_all("h4")]
        # Decide if the seller is a shop or private person
        advertisement_data["buy_from_shop"] = "Kereskedés adatai" in h4_headers

        # Sale contact
        contacts = soup.find_all("span", {"class": "contact-button-text"})
        for i in range(len(contacts)):
            advertisement_data[f"content_info_{i}"] = contacts[i].text

        # Use lower case column names
        advertisement_data.columns = advertisement_data.columns.str.lower().str.replace(
            ":", ""
        )

        # Remove '\x00' characters
        for col in advertisement_data.select_dtypes([object]):
            advertisement_data[col] = advertisement_data[col].str.replace("\x00", "")

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
        special_car_info = equipments + other
        special_car_info = pd.Series(special_car_info)
        if len(special_car_info) > 0:
            special_car_info = special_car_info.str.strip()
            special_car_info = special_car_info.str.lower()
            special_car_info = special_car_info.dropna()
            special_car_info = special_car_info[~special_car_info.isin(["", "-"])]
            special_car_info = special_car_info.str.replace("\x00", "")

        advertisement_data["feature_list"] = [special_car_info.to_list()]

        if not description:
            advertisement_data["description"] = ''
        else:
            advertisement_data["description"] = description

        advertisement_data["link"] = link
        advertisement_data["chassis_number"] = chassis_number

        return advertisement_data
    except Exception as e:
        print(e)
        logging.error(e)
        return None


class CarDataScraper:
    # Base URL is get from after I set the proper search parameters
    BASE_URL = "https://www.hasznaltauto.hu/talalatilista/PCOG2VG3R3RDADH5S56ADN4Z3EEY7O2CQO2EEI5NGSXZLINGQROIZEVUDJAPZ6Z2NVQZVCVHLLDY47L4OJJJB43XPEGXEOUTIJB4JCV4QIGTMYWJ3BKIU4CBRPTAY5NECZNEQAJGF7OPJ6CB7EJCHBQGJIEBPMO4ZJE57JDB3CAC7LDSVHEUICRRZ3APOMHTEIC5NPWIZZWZ4JOSWHDXZHXKYDCOECG62ZWR6KNG5X3U3CQKHS4J2AVDYKJYEWO2CQGLSJZZT24ULD7GQCI3F5FE7W7EXQP4FCUIQMIW64YWGWLIAXDQRRZ3Z2F646IDCZVF2UUPJUP5QYHBEF4F7FVX6EMGUE2DEIYGPHUTR3FK74JF7QHH575DDQF7IQIW76QW7VAOIOTNZZ7B5TEX3KISPM5DHHV4VNBSJ62XQX4OTSKSKTSNMZMXPZWU67NNG6MVJLEFBIV57HKRVZNCSE6RO4TZRHAFV6QVA7MLEQOJK2GMAPQZFKCOOU4KEGPZAQNRMHTCF3GAGHAIUNK4GF3B7IRL5VR5HM3UM3OH43FVFGP3TBL43LQHO6DOVUIXRSARGRNRAT4SFL5KWN3LXNUS2MXJ5JLY23C7GFGPXN4JDYSFC67YRZMNAO7LGA52BJ663Y4YLCPMXXYFO5QBXEJO3CWBM7UWVQGQNTCQL2BFKNDIF6SBG2NOXK3AFJ4XNQQ4FUGDLRGCCXBUXNOA74OXOSO3IV67THQRSRMDHITEXBACWYDLEQ3JQ2U5TNCORXQIR5UUMZT53WUXS4OHTUPEZ3GTBR2KMA4I3CBW4TUEBHVKEH7NN7KJ7DUQ2UNWXQ3MUZVLUZPQCT2DO7DFB3GJK5QOZAHHVUJZTLP5Q53LKRJXEB57WD3X72QGI/page"

    def __init__(
            self,
            link_collection_table="car_links",
            scraped_data_table="car_data",
    ):
        """
        Scrapes the data from the hasznaltauto.hu website
        :param link_collection_table: SQL source for the links of the cars.
        :param scraped_data_table: SQL source for all the scraped cars data.
        """
        self.link_collection_table = link_collection_table
        self.scraped_data_table = scraped_data_table

        # Load existing links
        if self.scraped_data_table in metadata.tables:
            sql = f"SELECT cl.link, array_length(feature_list, 1) IS NULL as empty_features " \
                  f"FROM {self.link_collection_table} cl " \
                  f"LEFT JOIN {self.scraped_data_table} cd on cd.link = cl.link"
            self.df_existing_links = read_sql_query(engine, sql)
        else:
            self.df_existing_links = None

    def get_links(self, page_number):
        """
        Get the links of the cars from the search page
        :param page_number:
        :return:
        """
        url = self.BASE_URL + str(page_number)
        response = requests.get(url, headers=HEADERS)  # proxies={'http': PROXY_NAME, 'https': PROXY_NAME})
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
        all_links = all_links.drop_duplicates()

        return all_links

    def get_new_links(self) -> pd.Series:
        """
        Get the new links that are not in the existing links
        :return: series of new links
        """
        all_hrefs = self.get_all_links()
        df_hrefs = pd.DataFrame({'link': all_hrefs.values})
        df_hrefs.to_sql('tmp_existing_links', engine, if_exists='replace', index=False)

        execute_query(engine, "CREATE UNIQUE INDEX idx_tmp ON tmp_existing_links(link);")

        if self.df_existing_links is not None:
            # Update the estimated_sold_date for rows where link is not in the exclude list
            set_sold_date_query = """
                UPDATE car_links
                SET estimated_sold_date = CURRENT_DATE
                WHERE car_links.link in (
                    select cl.link
                    from car_links cl
                    left join tmp_existing_links tel on tel.link = cl.link
                    where cl.estimated_sold_date is null
                    and tel.link is NULL
                );
            """

            # Execute the query using the engine
            execute_query(engine, set_sold_date_query)

            # Remove links that has empty feature list but still available among all_hrefs
            delete_links = all_hrefs[
                all_hrefs.isin(
                    self.df_existing_links[self.df_existing_links.empty_features]['link']
                )
            ]
            delete_query = f"""
                DELETE FROM car_links
                WHERE link IN ('{"','".join(delete_links.to_list())}');
            """

            # Execute the delete query using the engine
            execute_query(engine, delete_query)

            print('Number of deleted links: ', len(delete_links))

            # Deleted links not exists anymore we will scrape them
            self.df_existing_links = self.df_existing_links[~self.df_existing_links.link.isin(delete_links)]

            # Extract the hrefs from page results
            new_links = all_hrefs[~all_hrefs.isin(self.df_existing_links["link"])]
        else:
            new_links = all_hrefs

        return new_links

    def update_links(self) -> pd.Series:
        """
        Update the links
        :return: series of new links
        """
        # Get the new links
        new_links = self.get_new_links()

        # Create a dataframe from the new links
        df_new_hrefs = pd.DataFrame({"link": list(new_links)})

        # Make sure that the links are unique
        df_new_hrefs.drop_duplicates(inplace=True)

        # Update the links
        df_new_hrefs.to_sql(
            self.link_collection_table, engine, if_exists="append", index=False
        )
        print("Links are updated")

        return new_links

    def collect_car_data(self):
        """
        Collect the data of the cars from the new links
        :return:
        """
        print("Update links of the cars. This may take a while. (~30 minutes)")
        new_links = self.update_links()

        # Get a sample of car_data table
        car_data_sql = """
            select *
            from car_data
            limit 1;
        """
        sample = read_sql_query(engine, car_data_sql)

        '''
        sql = """
            select cl.*
            from car_links cl 
            left outer join car_data cd 
            on cl.link = cd.link 
            where cd.link is null;
        """
        new_links = pd.read_sql(sql_text(sql), engine.connect())["link"]
        '''
        print("Collect data of the cars. This may take a while. (~3 hours)")
        print("Number of new cars:", len(new_links))

        success_count = 0
        failed_links = []
        for i, link in enumerate(new_links):
            advertisement_data = scrape_car_data(link)
            # Keep only the existing columns
            advertisement_data = advertisement_data[sample.columns]

            if advertisement_data is not None:
                advertisement_data.to_sql(
                    "car_data", engine, if_exists="append", index=False
                )
                success_count += 1
                print(i)
            else:
                failed_links.append(link)

        print("Number of total cars scraped:", success_count)
        logging.info(f"Number of total cars scraped: {success_count}")

        # Delete unsuccessful links for links table
        if failed_links:
            with engine.connect() as connection:
                # SQL statement to delete rows from car_links where the link isn't in car_data
                delete_statement = sql_text(
                    f"""
                DELETE FROM {self.link_collection_table}
                WHERE link IN :failed_links
                """
                )

                # Execute the delete statement
                connection.execute(
                    delete_statement, {"failed_links": tuple(failed_links)}
                )


if __name__ == "__main__":
    scraper = CarDataScraper()
    scraper.collect_car_data()
