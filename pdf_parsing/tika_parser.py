def extract_text(file_path):
    """
    Extracts raw text from a PDF file using Tika.
    Requires Tika server to be running.
    """
    try:
        from tika import parser
        parsed = parser.from_file(file_path)
        return parsed.get("content", "") or ""
    except Exception as e:
        print(f"Error parsing PDF with tika: {e}")
        return ""
