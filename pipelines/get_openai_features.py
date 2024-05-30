import json
import os
import time

import pandas as pd
from openai_helper import get_openai_response, schema_to_instruction, BatchProcessor
from database_helpers import read_sql_query, setup_database, store_to_sql
import concurrent.futures

engine = setup_database()


class OpenAIFeatures:
    def __init__(self, df=None, model_name: str = "gpt-3.5-turbo", limit: int = 10000):
        """
        Initialize the OpenAIFeatures class. It will get the unclassified data from the database and
        create OpenAI features for each row.
        :param model_name: The name of the OpenAI model to use.
        :param limit: The number of rows to get from the database.
        """
        used_features = [
            "price (HUF)",
            "condition",
            "design",
            "finanszírozás",
            "clock position (km)",
            "shippable persons number",
            "number of doors",
            "color",
            "own weight (kg)",
            "total weight (kg)",
            "trunk (l)",
            "type of climate",
            "fuel",
            "cylinder capacity (cm3)",
            "power (kW)",
            "drive",
            "gearbox",
            "MOT is valid (days)",
            "age (year)",
            "MOT is valid (days)",
            "age (year)",
            "description",
            "fizetendő magyarországi forgalomba helyezés esetén",
            "extrákkal növelt ár",
            "akció feltételei",
            "átvehető",
            "finanszírozás típusa casco-val",
            "finanszírozás típusa casco nélkül",
            "garancia",
            "szavatossági garancia",
            "alaptípus ára",
            "futásidő",
            "átrozsdásodási garancia",
            "kezdőrészlet casco nélkül",
            "havi részlet casco nélkül",
            "futamidő casco nélkül",
            "garancia megnevezése",
            "garancia lejárata",
            "bérlési lehetőség",
            "kezdőrészlet",
            "futamidő",
        ]
        # Get unclassified data
        query = f"""
        SELECT  ecd.link, {", ".join(used_features)}
        FROM
            engineered_car_data ecd
        LEFT JOIN
            car_links cl
            ON ecd.link = cl.link
        LEFT join
            car_data cd 
            ON ecd.link = cd.link
        LEFT JOIN car_openai_features cof
            ON ecd.link = cof.link
        where cof.link is null
        and cl.collected_at > now() - interval '20 days'
        order by cl.collected_at desc
        limit {limit};
        """
        if df is None:
            self.df = read_sql_query(engine, query)
        else:
            self.df = df
            # Select only used features or create empty if missing
            for feature in used_features:
                if feature not in self.df.columns:
                    self.df[feature] = None
            self.df = self.df[used_features + ["link"]]

        # Add car name
        self.df["car_name"] = (
            self.df["link"]
            .str.replace("https://www.hasznaltauto.hu/szemelyauto/", "", regex=False)
            .str.replace(r"-\d+", "", regex=True)
            .str.replace(r"_", " ", regex=True)
            .str.split("/")
            .apply(lambda x: x[-1])
        )

        self.model_name = model_name
        self.tmp_json = {
            "custom_id": "request-1",
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {
                "model": self.model_name,
                "response_format": {"type": "json_object"},
                "messages": [
                    {
                        "role": "system",
                        "content": "You are a car expert and you are evaluating a used car advertisement.",
                    },
                    {"role": "user", "content": "Hello world!"},
                ],
                "max_tokens": 1000,
            },
        }

    def get_openai_prompt(self, row: pd.Series):
        """
        Create an OpenAI prompt for a given row.
        :param row: A row from the DataFrame.
        """
        # Drop nan values
        row = row.dropna()

        if "description" in row.keys():
            description_questions = (
                f"- Based on the description is this car has an issue? (has_current_issues)"
                f"- Based on the description is this car has recent fixes, upgrades, "
                f"or accessory replacements? (has_recent_fixes)"
            )
        else:
            description_questions = ""

        prompt = (
            f"{row.to_dict()}\n"
            f"These are the attributes of a used car advertisement from a Hungarian website. "
            f"This is why it's half Hungarian half English. "
            f"I want you to help me decide if this is a good deal for that car or not.\n"
            f"- Is this type of car have any known model-specific issues? (has_model_issues)\n"
            f"- If it has know model-specific issue what is that? (model_issues_detail)\n"
            f"{description_questions}\n"
            f"- Based on all the information about this car do you think it is worth the price? (worth_price)\n"
            f"- Would you pay more or less for this car and how much? "
            f"Write a percentage between -50 to +50. If you think the price is right ot can be 0. (price_adjustment)"
        )

        # Create langchain schema
        if "description" in row.keys():
            schema = {
                "properties": {
                    "has_model_issues": {
                        "type": "bool",
                    },
                    "model_issues_detail": {"type": "string"},
                    "has_current_issues": {
                        "type": "bool",
                    },
                    "has_recent_fixes": {
                        "type": "bool",
                    },
                    "worth_price": {
                        "type": "bool",
                    },
                    "price_adjustment": {"type": "int"},
                },
                "required": [
                    "has_model_issues",
                    "worth_price",
                    "current_issues",
                    "recent_fixes",
                    "price_adjustment",
                ],
            }
        else:
            schema = {
                "properties": {
                    "has_model_issues": {
                        "type": "bool",
                    },
                    "model_issues_detail": {"type": "string"},
                    "worth_price": {
                        "type": "bool",
                    },
                    "price_adjustment": {
                        "type": "int",
                    },
                },
                "required": ["has_model_issues", "worth_price", "price_adjustment"],
            }

        return prompt, schema

    def process_first_row(self):
        row = self.df.iloc[0]
        features = self.process_row(row)
        # Drop model name and link from openai_features dict
        features.pop("model_name")
        features.pop("link")

        return features

    def process_row(self, row):
        link = row["link"]
        try:
            # Remove the link from the row
            row = row.drop("link")

            prompt, schema = self.get_openai_prompt(row)
            response = get_openai_response(
                prompt=prompt,
                system_message=f"You are a car expert and you are evaluating a used car advertisement. ",
                schema=schema,
                model_name=self.model_name,
            )
            print(f"Got OpenAI features {response} for {row['car_name']}")
        except Exception as e:
            print(e)
            print(f"Failed to get OpenAI features for {row['car_name']}: {e}")
            response = {"error": e}

        # Add metadata
        response["link"] = link
        response["model_name"] = self.model_name

        return response

    def get_openai_features(self) -> pd.DataFrame:
        """
        Get OpenAI features for each row in the DataFrame using parallel processing.

        Returns:
            DataFrame: A DataFrame containing OpenAI features for each row.
        """
        response_list = []

        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
            # Submit each row processing to the executor
            futures = [
                executor.submit(self.process_row, row) for _, row in self.df.iterrows()
            ]
            # Iterate over the completed futures
            for future in concurrent.futures.as_completed(futures):
                # Get the result from the completed future
                response_list.append(future.result())

        return pd.DataFrame(response_list)

    def store_openai_features(self):
        openai_features = self.get_openai_features()
        store_to_sql(openai_features, engine, "car_openai_features")
        print("Stored OpenAI features to database.")
        return openai_features

    def create_batch_files(self):
        """
        The content will be the prompt.
        """
        # To be sure index starts from 0 reset index
        self.df.reset_index(drop=True, inplace=True)

        list_of_jsons = []
        jsonl_files = []
        outfile_index = 0
        for index, row in self.df.iterrows():
            link = row["link"]
            row = row.drop("link")
            prompt, schema = self.get_openai_prompt(row)
            instruction = schema_to_instruction(schema)
            prompt = f"{prompt}\n{instruction}"

            new_json = self.tmp_json.copy()
            new_json["body"]["messages"][1]["content"] = prompt
            new_json["custom_id"] = link

            list_of_jsons.append(json.dumps(new_json))
            # Save every 100 jsons
            if index + 1 % 101 == 0:
                outfile = f"openai_batch_{outfile_index}.jsonl"
                with open(outfile, "w") as f:
                    for request in list_of_jsons:
                        f.write(request + "\n")

                # Reset the list
                list_of_jsons = []

                jsonl_files.append(outfile)
                outfile_index += 1

        # Save the remaining jsons
        if list_of_jsons:
            outfile = f"openai_batch_{outfile_index}.jsonl"
            with open(outfile, "w") as f:
                for request in list_of_jsons:
                    f.write(request + "\n")

            jsonl_files.append(outfile)

        return jsonl_files

    def process_batch_files(self):
        jsonl_files = self.create_batch_files()
        bp = BatchProcessor(jsonl_files)
        df_response = bp.process_files()
        df_response.rename(columns={"custom_id": "link"}, inplace=True)
        # Upload to database
        store_to_sql(df_response, engine, "car_openai_features")

        # Delete the jsonl files
        for file in jsonl_files:
            os.remove(file)


if __name__ == "__main__":
    for i in range(100):
        print(i)
        start_time = time.time()
        openai_features = OpenAIFeatures(limit=100)
        openai_features.process_batch_files()
        end_time = time.time()
        print(f"Time elapsed: {end_time - start_time}")
