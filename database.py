import os
from astrapy import Database, Collection
from astrapy.constants import VectorMetric
from astrapy.info import CollectionVectorServiceOptions
import json
from astrapy import DataAPIClient, Database

def connect_to_database() -> Database:
    """
    Connects to a DataStax Astra database.
    This function retrieves the database endpoint and application token from the
    environment variables `ASTRA_DB_API_ENDPOINT` and `ASTRA_DB_APPLICATION_TOKEN`.

    Returns:
        Database: An instance of the connected database.

    Raises:
        RuntimeError: If the environment variables `ASTRA_DB_API_ENDPOINT` or
        `ASTRA_DB_APPLICATION_TOKEN` are not defined.
    """
    endpoint = os.environ.get("ASTRA_DB_API_ENDPOINT")
    token = os.environ.get("ASTRA_DB_APPLICATION_TOKEN")

    if not token or not endpoint:
        raise RuntimeError(
            "Environment variables ASTRA_DB_API_ENDPOINT and ASTRA_DB_APPLICATION_TOKEN must be defined"
        )

    # Create an instance of the `DataAPIClient` class with your token.
    client = DataAPIClient(token)

    # Get the database specified by your endpoint.
    database = client.get_database(endpoint)

    print(f"Connected to database {database.info().name}")

    return database




def create_collection(database: Database, collection_name: str) -> Collection:
    """
    Creates a collection in the specified database with vectorization enabled.
    The collection will use Nvidia's NV-Embed-QA embedding model
    to generate vector embeddings for data in the collection.

    Args:
        database (Database): The instantiated object that represents the database where the collection will be created.
        collection_name (str): The name of the collection to create.

    Returns:
        Collection: The created collection.
    """
    collection = database.create_collection(
        collection_name,
        metric=VectorMetric.COSINE,
        service=CollectionVectorServiceOptions(
            provider="nvidia",
            model_name="NV-Embed-QA",
        ),
    )

    print(f"Created collection {collection.full_name}")

    return collection


def upload_json_data(
    collection: Collection,
    data_file_path: str,
    embedding_string_creator: callable,
) -> None:
    """
     Uploads data from a file containing a JSON array to the specified collection.
     For each piece of data, a $vectorize field is added. The $vectorize value is
     a string from which vector embeddings will be generated.

    Args:
        collection (Collection): The instantiated object that represents the collection to upload data to.
        data_file_path (str): The path to a JSON file containing a JSON array.
        embedding_string_creator (callable): A function to create the string for which vector embeddings will be generated.
    """
    # Read the JSON file and parse it into a JSON array.
    with open(data_file_path, "r", encoding="utf8") as file:
        json_data = json.load(file)

    # Add a $vectorize field to each piece of data.
    documents = [
        {
            **data,
            "$vectorize": embedding_string_creator(data),
        }
        for data in json_data
    ]

    # Upload the data.
    inserted = collection.insert_many(documents)
    print(f"Inserted {len(inserted.inserted_ids)} items.")
