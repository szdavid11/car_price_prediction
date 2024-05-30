import sys
import json
import logging
from typing import Optional
from dotenv import load_dotenv
from openai import OpenAI
import pandas as pd
import time

# Create .env to store the API key
load_dotenv("../.env", override=True)
logging.basicConfig(level=logging.WARNING, stream=sys.stdout)

client = OpenAI()


class BatchProcessor:
    def __init__(self, files_queue: list[str]):
        """
        Initialize the BatchProcessor with an API key.
        :param files_queue: List of file paths to process.
        """
        self.files_queue = files_queue

    def process_files(self) -> pd.DataFrame:
        """
        Processes all files in the queue using the Batch API and returns the results in a DataFrame.

        :return: DataFrame containing the results with columns `custom_id` and `content`.
        """
        results = []

        for file_path in self.files_queue:
            file_id = self.upload_file(file_path)
            batch_id = self.create_batch(file_id)
            self.wait_for_completion(batch_id)
            response = self.retrieve_results(batch_id)
            results.extend(self.parse_responses(response.content))

        return pd.DataFrame(results)

    def upload_file(self, file_path: str) -> str:
        """
        Uploads a file and returns its ID.

        :param file_path: Path to the JSONL file.
        :return: ID of the uploaded file.
        """
        with open(file_path, "rb") as file:
            file_response = client.files.create(file=file, purpose="batch")
        return file_response.id

    def create_batch(self, file_id: str) -> str:
        """
        Creates a batch job and returns its ID.

        :param file_id: ID of the uploaded file.
        :return: ID of the created batch.
        """
        batch_response = client.batches.create(
            input_file_id=file_id,
            endpoint="/v1/chat/completions",
            completion_window="24h",
        )
        return batch_response.id

    def wait_for_completion(self, batch_id: str) -> None:
        """
        Waits for a batch to complete. Polls the batch status until it is completed.

        :param batch_id: ID of the batch.
        """
        while True:
            status = client.batches.retrieve(batch_id).status
            if status in ["completed", "failed", "cancelled"]:
                break
            time.sleep(30)

    def retrieve_results(self, batch_id: str) -> str:
        """
        Retrieves the results of a completed batch.

        :param batch_id: ID of the batch.
        :return: Content of the output file.
        """
        output_file_id = client.batches.retrieve(batch_id).output_file_id
        content = client.files.content(output_file_id)
        return content

    def parse_responses(self, content: str) -> list[dict]:
        """
        Parses the responses from a batch into a list of dictionaries.

        :param content: Content of the batch output file.
        :return: List of dictionaries with `custom_id` and `content` keys.
        """
        results = []
        for line in content.splitlines():
            # Parse the JSON response
            response = json.loads(line)
            try:
                # Extract the custom ID and GPT answer
                gpt_answer = response["response"]["body"]["choices"][0]["message"][
                    "content"
                ]
                json_answer = json.loads(gpt_answer)
            except Exception as e:
                logging.warning(f"Failed to parse JSON response: {e}")
                json_answer = {"error": str(e)}

            json_answer["custom_id"] = response["custom_id"]

            # Append the result to the list
            results.append(json_answer)

        return results


def validate_array_items(
    array_items, item_schema, path, min_items, max_items
) -> tuple[bool, str]:
    """
    TODO: Create a class for open AI call processes and move this function there
    Validate the items of an array.
    :param array_items: List of items in the array
    :param item_schema: Schema for the items in the array
    :param path: Path to the array in the response
    :param min_items: Minimum number of items required
    :param max_items: Maximum number of items allowed
    :return: Tuple of (bool, str) indicating if the response is valid and an error message if not.
    """
    if len(array_items) < min_items:
        return (
            False,
            f"Array at '{path}' has fewer items than the minimum required {min_items} items.",
        )
    if len(array_items) > max_items:
        logging.warning(
            f"Array at '{path}' has more items than the maximum allowed {max_items} items. Extra items will be removed."
        )
        del array_items[max_items:]

    for i, item in enumerate(array_items):
        item_path = f"{path}[{i}]"
        is_valid, error_message = validate_response(item, item_schema, item_path)
        if not is_valid:
            return False, error_message
    return True, ""


def validate_response(response: dict, schema: dict, path="") -> tuple[bool, str]:
    """
    Recursively validate the response based on the provided JSON schema.

    :param response: The response object to validate.
    :param schema: The JSON schema to validate against.
    :param path: The path to the current element in the response, used for error messages.
    :return: Tuple of (bool, str) indicating if the response is valid and an error message if not.
    """
    json_to_python_types = {
        "string": str,
        "number": (int, float),
        "boolean": bool,
        "object": dict,
        "array": list,
    }

    if "properties" in schema:
        for schema_key, key_schema in schema["properties"].items():
            key_path = f"{path}.{schema_key}" if path else schema_key
            if schema_key in response:
                if (
                    "enum" in key_schema
                    and response[schema_key] not in key_schema["enum"]
                ):
                    return (
                        False,
                        f"Value for '{key_path}' is not one of the allowed enum values.",
                    )

                if key_schema.get("type") == "array" and isinstance(
                    response[schema_key], list
                ):
                    min_items = key_schema.get("minItems", 0)
                    max_items = key_schema.get("maxItems", float("inf"))
                    item_schema = key_schema.get("items", {})
                    is_valid, error_message = validate_array_items(
                        response[schema_key],
                        item_schema,
                        key_path,
                        min_items,
                        max_items,
                    )
                    if not is_valid:
                        return False, error_message

                elif key_schema.get("type") == "object" and isinstance(
                    response[schema_key], dict
                ):
                    is_valid, error_message = validate_response(
                        response[schema_key], key_schema, key_path
                    )
                    if not is_valid:
                        return False, error_message
            elif (
                "required" in schema
                and schema_key in schema["required"]
                and schema_key not in response
            ):
                return False, f"Required key '{key_path}' not found in response"

    elif "type" in schema and "array" not in schema:
        expected_python_type = json_to_python_types.get(schema["type"], None)
        if expected_python_type and not isinstance(response, expected_python_type):
            expected_type = schema["type"]
            actual_type = type(response).__name__
            return (
                False,
                f"Type mismatch for '{path}'. Expected: {expected_type}, Got: {actual_type}",
            )

    if isinstance(response, dict):
        valid_keys = set(schema.get("properties", {}).keys())
        extra_keys = set(response.keys()) - valid_keys
        if extra_keys:
            for schema_key in extra_keys:
                logging.warning(
                    f"Extra key '{schema_key}' found in response and removed."
                )
                response.pop(schema_key)

    return True, ""


def schema_to_instruction(schema):
    """
    Generate instructions for the language model on how the response should look like
    based on the provided JSON schema, including handling of deeply nested objects.
    :param schema: JSON schema dictionary
    :return: String instruction for a language model
    """

    def parse_object_properties(properties, required):
        object_info = []
        for key_, value in properties.items():
            key_type = value.get("type", "unknown type")
            required_status = "required" if key_ in required else "optional"
            if key_type == "object":
                nested_properties = value.get("properties", {})
                nested_required = value.get("required", [])
                nested_info = parse_object_properties(
                    nested_properties, nested_required
                )
                object_info.append(
                    f"'{key_}': object ({required_status}, contains {nested_info})"
                )
            else:
                object_info.append(f"'{key_}': {key_type} ({required_status})")
        return ", ".join(object_info)

    def parse_properties(properties, required):
        keys_info = []
        json_template = {}
        for key_, value in properties.items():
            key_type = value.get("type", "unknown type")
            enum_values = value.get("enum", [])
            required_status = "required" if key_ in required else "optional"
            if key_type == "array":
                array_item_schema = value.get("items", {})
                min_items = value.get("minItems", 1 if key_ in required else 0)
                max_items = value.get("maxItems", None)
                if array_item_schema.get("type") == "object":
                    object_properties = array_item_schema.get("properties", {})
                    object_required = array_item_schema.get("required", [])
                    object_info = parse_object_properties(
                        object_properties, object_required
                    )
                    array_example = (
                        f"a list of {min_items}-{max_items} objects (each with {object_info})"
                        if max_items
                        else f"a list of at least {min_items} objects (each with {object_info})"
                    )
                    keys_info.append(
                        f"'{key_}': array of objects ({required_status}, {array_example})"
                    )
                    json_template[key_] = f"[{{object with {object_info}}}]"
                else:
                    array_item_type = array_item_schema.get("type", "unknown type")
                    if "enum" in array_item_schema:
                        array_item_type = f"enum values: {array_item_schema['enum']}"
                    array_example = (
                        f"a list of {min_items}-{max_items} {array_item_type}"
                        if max_items
                        else f"a list of at least {min_items} {array_item_type}"
                    )
                    keys_info.append(
                        f"'{key_}': array of {array_item_type} ({required_status})"
                    )
                    json_template[key_] = f"[{array_example}]"
            elif key_type == "object":
                nested_properties = value.get("properties", {})
                nested_required = value.get("required", [])
                nested_info, nested_example = parse_properties(
                    nested_properties, nested_required
                )
                json_template[key_] = nested_example
                keys_info.append(
                    f"'{key_}': object ({required_status}, contains {nested_info})"
                )
            else:
                if enum_values:
                    json_template[key_] = "One of the allowed values comes here"
                    keys_info.append(
                        f"'{key_}': {key_type} ({required_status}, allowed values: {enum_values})"
                    )
                else:
                    json_template[key_] = "The {} value comes here".format(key_type)
                    keys_info.append(f"'{key_}': {key_type} ({required_status})")

        return ", ".join(keys_info), json_template

    instruction_parts = []
    properties_info, example_json = parse_properties(
        schema.get("properties", {}), schema.get("required", [])
    )
    instruction_parts.append(
        f"The JSON object should contain the following keys: {properties_info}."
    )
    instruction_parts.append(
        "Required fields need to be part of the response; others can be NULL or just not included."
    )

    # Concatenate the example JSON to the instructions
    example_json_string = json.dumps(example_json, indent=2)
    instruction_parts.append(
        f"Example JSON structure based on schema:\n{example_json_string}"
    )

    final_instruction = "\n".join(instruction_parts)
    return final_instruction


def get_openai_response(
    prompt,
    model_name: str = "gpt-3.5-turbo",
    system_message: str = "You are an expert in fMRI data analysis and help collect metadata for a dataset. "
    "Provide json response.",
    response_format: str = "json_object",
    schema: Optional[dict] = None,
    trial=0,
) -> dict:
    """
    Get response from OpenAI API
    :param prompt: Input text to the API
    :param model_name: OpenAI model name
    :param system_message: Instruction for the assistant behavior
    :param response_format: Response format
    :param schema: JSON schema for validating the response
    :param trial: Number of trials
    :return: Response in dictionary format
    """
    # Get parent function details
    try:
        # Log the question
        logging.info(f"PROMPT FOR OPENAI CALL: {prompt}")

        # Add JSON format instruction to the question
        if response_format == "json_object":
            if schema:
                instruction = schema_to_instruction(schema)
                prompt = f"{prompt}\n{instruction}"

        chat_completion = client.chat.completions.create(
            model=model_name,
            response_format={"type": response_format},
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": prompt},
            ],
        )
        response = json.loads(chat_completion.choices[0].message.content)
        # Log the response
        logging.info(f"RESPONSE FROM OPENAI CALL: {response}")
    except Exception as e:
        logging.exception(f"Error in OpenAI call: {e}")
        response = {"error": e}

    return response
