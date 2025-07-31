import json
import os
from tqdm import tqdm
import weaviate
from weaviate.classes.config import Property, DataType
from weaviate import WeaviateClient
from sentence_transformers import SentenceTransformer
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

# Initialize logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class WeaviateImporter:
    def __init__(self, client: WeaviateClient, data_path: str):
        self.encoder = SentenceTransformer("paraphrase-multilingual-mpnet-base-v2", device='cuda')
        self.batch_size = 50
        self.data_path = data_path
        self.client = client

    def setup_schema(self):
        collection = self.client.collections.get('TextChunk')

        if not collection:
            self.client.collections.create(
                name="TextChunk",
                properties=[
                    Property(name="content", data_type=DataType.TEXT),
                    Property(name="word_count", data_type=DataType.INT),
                    Property(name="source", data_type=DataType.TEXT)
                ]
            )
            print("Collection 'TextChunk' created.")
        else:
            print("Collection 'TextChunk' already exists.")

    def import_data(self):
        # Get all JSON files in the directory
        json_files = [f for f in os.listdir(self.data_path) if f.endswith('.json')]
        
        total_files = len(json_files)
        print(f"Found {total_files} JSON files in {self.data_path}. Starting import...")

        # Use ThreadPoolExecutor to process files in parallel
        with ThreadPoolExecutor(max_workers=1) as executor:
            futures = []
            for json_file in json_files:
                json_file_path = os.path.join(self.data_path, json_file)
                print(f"Scheduling file: {json_file_path}")
                futures.append(executor.submit(self.import_data_file, json_file_path))

            # Wait for all futures to complete
            for future in as_completed(futures):
                try:
                    future.result()  # If an exception was raised during processing, it will be re-raised here
                except Exception as e:
                    logger.error(f"Error during import: {str(e)}")

    def import_data_file(self, json_file: str):
        # Open and load the JSON file
        with open(json_file, 'r', encoding='utf-8') as f:
            chunks = json.load(f)
        
        total = len(chunks)
        print(f"Importing {total} chunks from {json_file}...")

        # Batch process the chunks
        with self.client.batch.fixed_size(batch_size=self.batch_size, concurrent_requests=4) as batch:
            for i, chunk in enumerate(chunks):
                try:
                    vector = self.encoder.encode(chunk)
                    batch.add_object(
                        collection="TextChunk",
                        properties={
                            "content": chunk,
                            "word_count": len(chunk.split()),
                            "source": json_file
                        },
                        vector=vector.tolist()
                    )

                    # Print progress every 1000 chunks
                    if i % 1000 == 0 and i > 0:
                        print(f"Processed {i}/{total} chunks from {json_file}")
                
                except Exception as e:
                    print(f"Error processing chunk {i} from {json_file}: {str(e)}")
                    continue

def main():
    client = None
    try:
        # Try to connect to Weaviate
        with weaviate.connect_to_custom(
            http_host=os.getenv("WEAVIATE_HTTP_HOST", "192.168.1.11"),
            http_port=os.getenv("WEAVIATE_HTTP_PORT", "8080"),
            http_secure=False,  # Use True for HTTPS
            grpc_host=os.getenv("WEAVIATE_GRPC_HOST", "192.168.1.11"),
            grpc_port=os.getenv("WEAVIATE_GRPC_PORT", "50051"),
            grpc_secure=False  # Use True for secure gRPC
        ) as client:

            # Check if connection is ready
            if client.is_ready():
                logger.info(f"Successfully connected to Weaviate {client.get_meta()})")
            else:
                raise Exception("Weaviate connection is not ready.")
            # Initialize importer and setup schema (will only run if connection is successful)
            importer = WeaviateImporter(client, os.getenv("DATA_PATH", "./processed_data"))
            importer.setup_schema()
            importer.import_data()

    except weaviate.exceptions.ConnectionError as conn_err:
        # Catch connection-related errors specifically
        logger.error(f"Weaviate connection failed: {str(conn_err)}")
        raise  # Re-raise the connection error to notify the caller

    except Exception as e:
        # Let all other exceptions propagate
        logger.error(f"An error occurred: {str(e)}")
        raise

    finally:
        # Always ensure the client is closed if it was successfully created
        if client:
            client.close()
            logger.info("Weaviate client connection closed.")

if __name__ == "__main__":
    main()
