import json
import os
from tqdm import tqdm
import weaviate
from sentence_transformers import SentenceTransformer

class WeaviateImporter:
    def __init__(self, weaviate_url: str):
        self.client = weaviate.Client(weaviate_url)
        self.encoder = SentenceTransformer("paraphrase-multilingual-mpnet-base-v2", device='cuda')
        self.batch_size = 50
        self.setup_schema()

    def setup_schema(self):
        schema = {
            "classes": [{
                "class": "TextChunk",
                "properties": [
                    {"name": "content", "dataType": ["text"]},
                    {"name": "text", "dataType": ["string"]},
                    {"name": "chapter", "dataType": ["string"]},
                    {"name": "word_count", "dataType": ["int"]},
                    {"name": "source", "dataType": ["string"]}
                ],
                "vectorizer": "none"
            }]
        }
        self.client.schema.create(schema)

    def import_data(self, json_file: str):
        with open(json_file, 'r', encoding='utf-8') as f:
            chunks = json.load(f)
        
        total = len(chunks)
        print(f"Importing {total} chunks...")

        with self.client.batch(batch_size=self.batch_size) as batch:
            for i, chunk in enumerate(tqdm(chunks)):
                try:
                    vector = self.encoder.encode(chunk["content"])
                    batch.add_data_object(
                        data_object=chunk,
                        class_name="TextChunk",
                        vector=vector.tolist()
                    )
                    
                    # Print progress every 1000 chunks
                    if i % 1000 == 0 and i > 0:
                        print(f"Processed {i}/{total} chunks")
                
                except Exception as e:
                    print(f"Error processing chunk {i}: {str(e)}")
                    continue

if __name__ == "__main__":
    importer = WeaviateImporter("http://localhost:8080")  # Change to Weaviate server IP
    importer.import_data("./processed_data/text_chunks.json")