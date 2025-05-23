import os
import json
from pathlib import Path
from tqdm import tqdm
import re

class TextPreprocessor:
    # Remve special characters
    # Return lines as a list
    def clean_text(self, lines: list[str]) -> list[str]:
        """Remove unwanted characters and normalize text"""
        cleaned_lines = []
        
        for line in lines:
            line = re.sub(r'\r\n', '\n', line)  # Standardize line breaks
            line = re.sub(r'[^\w\s，。！？、：；（）《》【】\n]', '', line)  # Remove special chars
            cleaned_lines.append(line)
        return cleaned_lines


    def sliding_window(self, lines: list[str], window_size=10, stride=5) -> list[str]:
        windows = []
        for i in range(0, len(lines) - window_size + 1, stride):
            window = lines[i:i + window_size]
            windows.append(window)
        return windows

def process_all_files(input_dir: str, output_dir: str):
    processor = TextPreprocessor()
    for novel_id in os.listdir(input_dir):
        chunks = []
        novel_path = os.path.join(input_dir, novel_id)
        if os.path.isdir(novel_path):
            for text_file in os.listdir(novel_path):
                text_file_path = os.path.join(novel_path, text_file)
                if os.path.isfile(text_file_path):
                    lines = []
                    with open(text_file_path, 'r', encoding='utf-8') as file:
                        lines = file.readlines()
                    lines = [line.strip() for line in lines if line.strip()]
                    lines = processor.clean_text(lines)
                    windows = processor.sliding_window(lines)
                    chunks.extend([" ".join(window) for window in windows])
    
            with open(f"{output_dir}/{novel_id}.json", 'w', encoding='utf-8') as f:
                json.dump(chunks, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    process_all_files(
        input_dir="./texts_raw",
        output_dir="./texts_processed"
    )
    