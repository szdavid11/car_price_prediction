from transformers import pipeline
from database_helpers import read_sql_query, setup_database, store_to_sql

engine = setup_database()


def translate_hungarian_to_english(text: str) -> str:
    """
    Translate Hungarian text to English.

    :param text: Hungarian text.
    :return: English translation.
    """
    if text:
        return None

    pipe = pipeline("translation", model="Helsinki-NLP/opus-mt-tc-big-hu-en")
    return pipe(text)


def extract_descriptions_with_llm() -> None:
    """
    Extract descriptions from the database and translate them to English.

    :return: None
    """
    # Load the data
    query = """
    SELECT description
    FROM car_data cd
    LEFT JOIN translated_descriptions td on cd.link = td.link
    WHERE td.link IS NULL;
    """
    df = read_sql_query(engine, query)

    # Translate the descriptions
    df["translated_description"] = df["description"].apply(
        translate_hungarian_to_english
    )

    # Save the translated descriptions
    store_to_sql(
        df[["link", "translated_description"]],
        engine,
        table_name="translated_descriptions",
        is_exists="append",
    )
