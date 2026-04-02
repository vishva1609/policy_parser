# 🔍 Policy Parser

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.8+-3776AB?style=for-the-badge&logo=python&logoColor=white"/>
  <img src="https://img.shields.io/badge/Type-Privacy%20Policy%20Parser-FF6B6B?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/License-MIT-22C55E?style=for-the-badge"/>
  <img src="https://img.shields.io/github/stars/vishva1609/policy_parser?style=for-the-badge&logo=github"/>
</p>

<p align="center">
  <b>A Python tool that automatically parses website Privacy Policy documents and converts them into clean, structured, human-readable data.</b><br/>
  <i>Stop reading 10-page privacy policies manually — let the parser do it for you! 😄</i>
</p>

---

## 📌 About the Project

**Policy Parser** is a Python-based tool that takes any privacy policy document as input (from a URL or a local file) and intelligently extracts key information such as data collection practices, third-party sharing, user rights, cookie usage, and data retention policies.

---

## ✨ Features

- 🕵️ **Automatic Extraction** — Identifies and extracts important clauses from privacy policies
- 📑 **Section Detection** — Detects document sections like Data Collection, Cookies, User Rights, etc.
- 📤 **Structured Output** — Returns results in clean JSON format
- 🌐 **URL Support** — Fetch and parse a policy directly from any website URL
- 📁 **File Support** — Also supports local `.txt` and `.html` files
- ⚡ **Fast & Lightweight** — No heavy ML models, pure Python-based parsing

---

## 📂 Project Structure

```
policy_parser/
│
├── parser/
│   ├── __init__.py
│   ├── core.py            # Core parsing logic
│   ├── extractor.py       # Key clause and section extractor
│   └── utils.py           # Utility / helper functions
│
├── data/
│   └── sample_policy.txt  # Sample privacy policy for testing
│
├── output/
│   └── parsed_result.json # Parsed output is saved here
│
├── main.py                # CLI entry point
├── requirements.txt       # Project dependencies
└── README.md
```

---

## ⚙️ Installation

### Prerequisites

- Python **3.8** or higher
- pip package manager

### Steps

```bash
# 1. Clone the repository
git clone https://github.com/vishva1609/policy_parser.git

# 2. Navigate to the project directory
cd policy_parser

# 3. Install dependencies
pip install -r requirements.txt
```

---

## 🚀 Usage

### Using in a Python Script

```python
from parser.core import PolicyParser

# Initialize the parser
parser = PolicyParser()

# Parse from a local file
result = parser.parse_file("data/sample_policy.txt")

# OR parse directly from a URL
result = parser.parse_url("https://example.com/privacy-policy")

print(result)
```

### Using the Command Line Interface (CLI)

```bash
# Parse a local file
python main.py --input data/sample_policy.txt

# Parse directly from a URL
python main.py --url https://example.com/privacy-policy

# Specify a custom output file
python main.py --input data/sample_policy.txt --output output/result.json

# Enable verbose logging
python main.py --input data/sample_policy.txt --verbose
```

### CLI Arguments

| Argument | Short | Description | Example |
|----------|-------|-------------|---------|
| `--input` | `-i` | Path to a local policy file | `data/policy.txt` |
| `--url` | `-u` | Website URL to fetch policy from | `https://example.com/privacy` |
| `--output` | `-o` | Path for the output JSON file | `output/result.json` |
| `--verbose` | `-v` | Show detailed processing logs | — |

---

## 📤 Sample Output

```json
{
  "source": "https://example.com/privacy-policy",
  "parsed_at": "2026-04-02T10:00:00",
  "summary": {
    "data_collected": ["name", "email", "IP address", "cookies"],
    "data_sharing": "Data may be shared with third parties for advertising purposes",
    "user_rights": ["Access", "Deletion", "Opt-out"],
    "data_retention": "Data is retained for up to 2 years",
    "cookies_used": true,
    "contact_email": "privacy@example.com"
  },
  "sections": [
    {
      "title": "Information We Collect",
      "content": "We collect information you provide directly to us...",
      "keywords": ["collect", "personal data", "email"]
    },
    {
      "title": "Third Party Sharing",
      "content": "We may share your information with trusted partners...",
      "keywords": ["share", "third party", "partners"]
    }
  ]
}
```

---

## 🛠️ Tech Stack

| Library | Purpose |
|---------|---------|
| `requests` | Fetching policy pages from URLs |
| `BeautifulSoup4` | Parsing and extracting text from HTML |
| `re` | Regex-based pattern matching |
| `json` | Generating structured output |
| `argparse` | Building the CLI interface |

---

## 🧪 Testing

```bash
# Run with the sample policy file
python main.py --input data/sample_policy.txt --verbose
```

---

## 🤝 Contributing

Contributions are always welcome! Here's how you can help:

1. 🍴 Fork the repository
2. 🌿 Create a new branch — `git checkout -b feature/your-feature-name`
3. ✏️ Make your changes and commit — `git commit -m "feat: add your feature"`
4. 📤 Push to your branch — `git push origin feature/your-feature-name`
5. 🔁 Open a Pull Request

---

## 🐛 Reporting Issues

Found a bug or have a suggestion? [Open an issue here](https://github.com/vishva1609/policy_parser/issues) and describe the problem.

---

## 👤 Author

**Vishva**
🔗 GitHub: [@vishva1609](https://github.com/vishva1609)

---

## 📃 License

This project is licensed under the [MIT License](LICENSE) — free to use, modify, and distribute.

---

<p align="center">
  If you found this project helpful, please consider giving it a ⭐ — it means a lot! 🙏
</p>
