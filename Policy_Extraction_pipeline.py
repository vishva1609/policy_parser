import fitz  # PyMuPDF for PDF extraction
import re
import requests
import json
import pandas as pd

# ================================
# Configuration
# ================================
API_KEY = "Your_Api"  # Replace with your DeepSeek API key
API_URL = "https://openrouter.ai/api/v1/chat/completions"  # DeepSeek API endpoint

# ================================
# Extract Headings from Bank Policy
# ================================
def extract_headings(text):
    """
    Extracts headings from text using a regex pattern.
    Looks for lines that start with a number (e.g., "1.", "3.2", "3.25").
    Filters to include only headings starting with "1", "2", or "3".
    """
    pattern = r'^\s*(\d+(?:\.\d+)*)(?:\s+|-)(.+)$'
    headings = []
    for match in re.finditer(pattern, text, re.MULTILINE):
        number = match.group(1).strip()
        title = match.group(2).strip()
        if number.startswith(("1", "2", "3")):
            headings.append({
                "number": number,
                "title": title,
                "full_heading": f"{number} {title}"
            })
    return headings

# ================================
# PDF and Text File Extraction Functions
# ================================
def extract_text_from_pdf(file_path):
    """Extracts and returns all text from a PDF file using PyMuPDF."""
    doc = fitz.open(file_path)
    text = "\n".join([page.get_text() for page in doc])
    return text

def read_text_file(file_path):
    """Reads and returns text from a plain text file."""
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()

# ================================
# Extract Context for a Given Heading
# ================================
def get_paragraphs_for_heading(text, heading):
    """
    Extracts paragraphs from the text belonging to a given heading.
    Starts capturing from the heading and stops at the next numbered heading.
    """
    lines = text.splitlines()
    context_lines = []
    capture = False
    heading_pattern = r'^\s*\d+(?:\.\d+)*'  # Pattern for new numbered headings
    for line in lines:
        if not capture:
            if heading.lower() in line.lower():
                capture = True
                context_lines.append(line.strip())
        else:
            if re.match(heading_pattern, line) and heading.lower() not in line.lower():
                break
            context_lines.append(line.strip())
    return "\n".join(context_lines)

def find_vendor_context(text, heading):
    """
    Attempts to extract relevant context from the vendor policy by matching significant words from the heading.
    Returns up to the first three paragraphs that contain any word from the heading.
    """
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    heading_clean = re.sub(r"[^\w\s]", "", heading.lower())
    words = [w for w in heading_clean.split() if len(w) > 2]
    matched = [p for p in paragraphs if any(word in p.lower() for word in words)]
    return "\n\n".join(matched[:3])

# ================================
# DeepSeek API Call with Controlled Output
# ================================
def call_deepseek_api(prompt, max_tokens=800):
    """
    Sends a prompt to the DeepSeek API and returns the generated content.
    Uses controlled settings (low temperature and max tokens) for deterministic output.
    """
    headers = {
        'Authorization': f'Bearer {API_KEY}',
        'Content-Type': 'application/json'
    }
    data = {
        "model": "deepseek/deepseek-chat:free",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.3  # Lower temperature to limit randomness
    }
    response = requests.post(API_URL, headers=headers, json=data)
    try:
        res_json = response.json()
    except Exception as e:
        print("Failed to decode JSON response:", e)
        print("Response text:", response.text)
        return "Error: Unable to decode API response."
    if 'choices' in res_json:
        return res_json['choices'][0]['message']['content']
    else:
        print("API response missing 'choices' key. Full response:")
        print(json.dumps(res_json, indent=4))
        return "Error: API response did not contain choices."

# ================================
# Generate Controlled Q&A Pairs in JSON Format for a Heading
# ================================
def generate_controlled_qa(heading, bank_context, vendor_context):
    """
    Constructs a prompt that instructs DeepSeek to generate 5 Q&A pairs (in JSON format).
    The output must have the same question for both sources with keys:
    'question', 'bank_answer', and 'vendor_answer'.
    """
    prompt = (
        f"Using the following policy section heading and its contexts, generate exactly 5 Q&A pairs in valid JSON format (as a list of objects). "
        f"Each object must include the keys 'question', 'bank_answer', and 'vendor_answer'. "
        f"Ensure that the same question is used for both answers, and the bank_answer is derived from the Bank Policy Context, while the vendor_answer is derived from the Vendor Policy Context.\n\n"
        f"Heading: {heading}\n\n"
        f"Bank Policy Context (first 500 chars): {bank_context[:500]}...\n\n"
        f"Vendor Policy Context (first 500 chars): {vendor_context[:500]}...\n\n"
        f"Return the output as a JSON array, for example:\n"
        f"[{{\"question\": \"<question>\", \"bank_answer\": \"<bank answer>\", \"vendor_answer\": \"<vendor answer>\"}}, ...]\n\n"
        f"Do not include any extra text."
    )
    qa_response = call_deepseek_api(prompt, max_tokens=800)
    try:
        qa_json = json.loads(qa_response)
    except Exception as e:
        print("JSON parsing failed for heading:", heading)
        print("Raw output:", qa_response)
        qa_json = qa_response  # fallback to raw text if JSON parsing fails
    return qa_json

# ================================
# Process Headings for Controlled Q&A Comparison
# ================================
def process_headings_for_comparison(bank_file, vendor_file):
    """
    Extracts headings from the bank policy, retrieves corresponding context from both bank and vendor policies,
    then generates controlled Q&A pairs (same question, separate answers) for each heading.
    Returns a list of result dictionaries.
    """
    bank_text = extract_text_from_pdf(bank_file)
    vendor_text = read_text_file(vendor_file)
    
    headings = extract_headings(bank_text)
    print(f"Found {len(headings)} headings in bank policy.")
    results = []
    
    for heading in headings:
        heading_text = heading["full_heading"]
        print(f"\nProcessing heading: {heading_text}")
        
        bank_context = get_paragraphs_for_heading(bank_text, heading_text)
        vendor_context = find_vendor_context(vendor_text, heading_text)
        
        if not bank_context:
            print("Skipping heading (no bank context found):", heading_text)
            continue
        
        qa_pairs = generate_controlled_qa(heading_text, bank_context, vendor_context)
        
        result = {
            "heading": heading_text,
            "bank_context": bank_context[:300] + "..." if len(bank_context) > 300 else bank_context,
            "vendor_context": vendor_context[:300] + "..." if len(vendor_context) > 300 else vendor_context,
            "qa_pairs": qa_pairs
        }
        results.append(result)
    return results

# ================================
# Save Results to JSON and Excel
# ================================
def save_results(results, json_filename="controlled_heading_qna.json", excel_filename="controlled_heading_qna.xlsx"):
    with open(json_filename, "w", encoding="utf-8") as f_json:
        json.dump(results, f_json, indent=4, ensure_ascii=False)
    df = pd.DataFrame(results)
    df.to_excel(excel_filename, index=False)
    print(f"Results saved to {json_filename} and {excel_filename}")

# ================================
# Main Execution
# ================================
def main():
    bank_file = "bank_policy.pdf"      # Path to bank policy PDF
    vendor_file = "vendor_policy.txt"  # Path to vendor policy text file
    
    results = process_headings_for_comparison(bank_file, vendor_file)
    save_results(results)
    
    if results:
        sample = results[0]
        print("\nSample Result:")
        print("Heading:", sample["heading"])
        print("Bank Q&A (first 200 chars):", str(sample["qa_pairs"])[:200] + "...")
    else:
        print("No results generated.")

if __name__ == "__main__":
    main()
