import json
import os
from tqdm import tqdm
import weaviate
from sentence_transformers import SentenceTransformer

class WeaviateImporter:
    def __init__(self, weaviate_url: str, data_path: str):
        self.client = weaviate.Client(weaviate_url)
        self.encoder = SentenceTransformer("paraphrase-multilingual-mpnet-base-v2", device='cuda')
        self.batch_size = 50
        self.setup_schema()
        self.data_path = data_path

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
    data_path = os.getenv("DATA_PATH", "./processed_data")  # Default if not provided
    
    importer = WeaviateImporter(weaviate_url, data_path)
    importer.import_data()
