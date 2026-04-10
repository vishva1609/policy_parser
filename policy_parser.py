import os
import re
import json
import nltk
from collections import Counter
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
import argparse
import yaml
import importlib

# Download NLTK data if not already present
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)
nltk.download('stopwords', quiet=True)

# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# 1. PDF Parsing using dynamic selection
# ---------------------------------------------------------------------------
def get_pdf_extractor():
    """Loads the correct PDF parser based on config.yaml"""
    parser_type = "pdfplumber" # Default
    try:
        if os.path.exists("config.yaml"):
            with open("config.yaml", "r") as f:
                cfg = yaml.safe_load(f)
                if "pdf_parsing" in cfg and "library" in cfg["pdf_parsing"]:
                    parser_type = cfg["pdf_parsing"]["library"]
    except Exception as e:
        print(f"Warning: Could not read config.yaml to determine pdf parser, defaulting to pdfplumber: {e}")

    try:
        module = importlib.import_module(f"pdf_parsing.{parser_type}_parser")
        return module.extract_text
    except Exception as e:
        print(f"Error loading {parser_type} parser module: {e}. Falling back to empty text.")
        return lambda file_path: ""

# Legacy extraction for backward compatibility with the rest of the file
def extract_text(file_path):
    extractor = get_pdf_extractor()
    return extractor(file_path)


from compliance_engine.clause_segmentation import clean_text, segment_clauses, extract_keywords, generate_context

# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------
def main():
    parser_ = argparse.ArgumentParser(description="Policy PDF → structured JSON")
    parser_.add_argument('pdf_path', type=str, help='Path to the policy PDF file')
    parser_.add_argument('--output', type=str, default=None,
                         help='Output JSON file (default: <pdf_name>_output.json)')
    args = parser_.parse_args()

    pdf_path = args.pdf_path
    if not os.path.isfile(pdf_path):
        print(f"File not found: {pdf_path}")
        return

    file_name = os.path.basename(pdf_path)
    print(f"Extracting text from {file_name} …")
    raw_text = extract_text(pdf_path)
    cleaned = clean_text(raw_text)

    print("Segmenting clauses …")
    segments = segment_clauses(cleaned, file_name=file_name)

    output_path = args.output or os.path.splitext(pdf_path)[0] + '_output.json'
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(segments, f, indent=2, ensure_ascii=False)

    print(f"Done! {len(segments)} segments written to {output_path}")


if __name__ == "__main__":
    main()
