import json
import os
from tqdm import tqdm
import weaviate
from weaviate import WeaviateClient
from sentence_transformers import SentenceTransformer
import logging

# Initialize logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class WeaviateImporter:
    def __init__(self, weaviate_url: str, grpc_port: str, data_path: str):
        self.weaviate_url = weaviate_url
        self.grpc_port = grpc_port
        self.encoder = SentenceTransformer("paraphrase-multilingual-mpnet-base-v2", device='cuda')
        self.batch_size = 50
        self.data_path = data_path
        self.client = None

    def connect(self):
        try:
            self.client = WeaviateClient(
                connection_params=weaviate.connect.ConnectionParams.from_url(
                    url=self.weaviate_url,
                    grpc_port=self.grpc_port  # Required in v4
                )
            )
            client = self.client .connect()
            logger.info(f"Successfully connected to Weaviate at {self.weaviate_url} (gRPC: {self.grpc_port})")
        except Exception as e:
            logger.error(f"Weaviate connection failed: {str(e)}")
            raise
        
    def setup_schema(self):
        schema = {
            "classes": [{
                "class": "TextChunk",
                "properties": [
                    {"name": "content", "dataType": ["text"]},
                    {"name": "word_count", "dataType": ["int"]},
                    {"name": "source", "dataType": ["string"]}
                ],
                "vectorizer": "none"
            }]
        }
        self.client.schema.create(schema)

    def import_data(self):
        # Get all JSON files in the directory
        json_files = [f for f in os.listdir(self.data_path) if f.endswith('.json')]
        
        total_files = len(json_files)
        print(f"Found {total_files} JSON files in {self.data_path}. Starting import...")

        for json_file in json_files:
            json_file_path = os.path.join(self.data_path, json_file)
            print(f"Processing file: {json_file_path}")
            self.import_data_file(json_file_path)

    def import_data_file(self, json_file: str):
        # Open and load the JSON file
        with open(json_file, 'r', encoding='utf-8') as f:
            chunks = json.load(f)
        
        total = len(chunks)
        print(f"Importing {total} chunks from {json_file}...")

        # Batch process the chunks
        with self.client.batch(batch_size=self.batch_size) as batch:
            for i, chunk in enumerate(chunks):
                try:
                    vector = self.encoder.encode(chunk)
                    batch.add_data_object(
                        data_object={"content": chunk, "word_count": len(chunk.split()), "source": json_file},
                        class_name="TextChunk",
                        vector=vector.tolist()
                    )

                    # Print progress every 1000 chunks
                    if i % 1000 == 0 and i > 0:
                        print(f"Processed {i}/{total} chunks from {json_file}")
                
                except Exception as e:
                    print(f"Error processing chunk {i} from {json_file}: {str(e)}")
                    continue

if __name__ == "__main__":
    # Get the Weaviate URL and the data path from environment variables
    weaviate_url = os.getenv("WEAVIATE_URL", "http://localhost:8080")  # Default if not provided
    grpc_port = os.getenv("WEAVIATE_GRPC_PORT", "50051")  # Default gRPC port
    data_path = os.getenv("DATA_PATH", "./processed_data")  # Default if not provided
    
    importer = WeaviateImporter(weaviate_url, grpc_port, data_path)
    importer.connect()
    importer.setup_schema()
    importer.import_data()
