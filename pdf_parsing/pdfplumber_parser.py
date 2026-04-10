import pdfplumber

def extract_text(file_path):
    """
    Extracts raw text from a PDF file using PDFPlumber.
    This replaces Tika to ensure full Windows compatibility and cleaner text.
    """
    text = ""
    try:
        with pdfplumber.open(file_path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
    except Exception as e:
        print(f"Error parsing PDF with pdfplumber: {e}")
    return text
