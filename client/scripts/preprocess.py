import os
import json
from pathlib import Path
from sentence_splitter import SentenceSplitter
from tqdm import tqdm
import re

class TextPreprocessor:
    def __init__(self):
        self.splitter = SentenceSplitter(language='zh')
        self.chunk_size = 500  # Target chunk size in characters
        self.min_chunk = 100   # Minimum chunk size

    def clean_text(self, text: str) -> str:
        """Remove unwanted characters and normalize text"""
        text = re.sub(r'\r\n', '\n', text)  # Standardize line breaks
        text = re.sub(r'[^\w\s，。！？、：；（）《》【】\n]', '', text)  # Remove special chars
        return text.strip()

    def chunk_text(self, text: str, title: str) -> list[dict]:
        """Split text into semantically coherent chunks"""
        chunks = []
        chapters = re.split(r'\n第[一二三四五六七八九十百]+章', text)[1:]  # Chinese chapter splitting
        
        for chap_num, chapter in enumerate(chapters, 1):
            sentences = self.splitter.split(chapter)
            current_chunk = ""
            
            for sent in sentences:
                if len(current_chunk) + len(sent) <= self.chunk_size:
                    current_chunk += sent
                else:
                    if len(current_chunk) >= self.min_chunk:
                        chunks.append(self._create_chunk(current_chunk, title, chap_num))
                    current_chunk = sent
            
            if current_chunk and len(current_chunk) >= self.min_chunk:
                chunks.append(self._create_chunk(current_chunk, title, chap_num))
        
        return chunks

    def _create_chunk(self, text: str, title: str, chap_num: int) -> dict:
        return {
            "content": text.strip(),
            "text": title,
            "chapter": f"第{chap_num}章",
            "word_count": len(text),
            "source": "txt"
        }

def process_all_texts(input_dir: str, output_file: str):
    processor = TextPreprocessor()
    all_chunks = []
    
    for text_file in tqdm(list(Path(input_dir).glob("*.txt"))):
        text = text_file.read_text(encoding='utf-8')
        cleaned = processor.clean_text(text)
        chunks = processor.chunk_text(cleaned, text_file.stem)
        all_chunks.extend(chunks)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(all_chunks, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    process_all_texts(
        input_dir="./texts_raw",
        output_file="./processed_data/text_chunks.json"
    )