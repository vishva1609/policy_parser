import json
import pandas as pd

def generate_eval_dataset():
    """Generates a sample evaluation dataset for training compliance classifiers."""
    dataset = [
        {
            "clause_a": "Data must be encrypted using AES-256 at rest.",
            "clause_b": "All storage disks are encrypted with AES-256 standards.",
            "label": "compliant",
            "category": "Security",
            "explanation": "Exact match for encryption standard."
        },
        {
            "clause_a": "All access logs must be retained for 1 year.",
            "clause_b": "Activity logs are kept for 6 months.",
            "label": "partial",
            "category": "Logging",
            "explanation": "Retention period mismatch (6m < 1y)."
        },
        {
            "clause_a": "Multi-factor authentication is mandatory for all users.",
            "clause_b": "Password-based login is supported.",
            "label": "non-compliant",
            "category": "Access Control",
            "explanation": "MFA requirement missing in target policy."
        },
        {
            "clause_a": "PII data must not be transferred outside the EU.",
            "clause_b": "N/A",
            "label": "missing",
            "category": "Data Privacy",
            "explanation": "No mention of data localization in target policy."
        }
    ]
    
    with open("eval_dataset.json", "w") as f:
        json.dump(dataset, f, indent=2)
    
    df = pd.DataFrame(dataset)
    df.to_csv("eval_dataset.csv", index=False)
    print("Generated eval_dataset.json and eval_dataset.csv for classifier training.")

if __name__ == "__main__":
    generate_eval_dataset()
