
import os
import json
from policy_parser import clean_text, segment_clauses
from compliance_engine.pipeline import run_comparison_sync

# --- SETUP POLICY A (The Standard) ---
text_a = """
1 Data Security
All company data must be protected against unauthorized access.

1.1 Encryption
All sensitive data must be encrypted using AES-256.
- Encryption keys must be rotated every 90 days.
- Access to encryption keys is restricted.

1.2 Data Retention
Data must be retained for a minimum of 7 years.
- Data older than 7 years must be deleted securely.

2.1 User Authentication
All users must authenticate using multi-factor authentication (MFA).
- Passwords must be at least 12 characters.
- Passwords must be changed every 180 days.
"""

# --- SETUP POLICY B (The Implementation to Check) ---
text_b = """
1 Data Security
We protect our corporate assets from unauthorized eyes.

1.1 Encryption
We use standard AES-128 encryption.
- Keys are rotated every six months (180 days).

1.2 Data Retention
Customer data is kept for 3 years to save storage space.

2.1 User Authentication
We require MFA for all logins.
- Password minimum length is 8 characters.
"""

def demo():
    print("Step 1: Parsing Policies A and B...")
    clauses_a = segment_clauses(clean_text(text_a), file_name="Standard_A.txt")
    clauses_b = segment_clauses(clean_text(text_b), file_name="Implementation_B.txt")

    print(f"Parsed {len(clauses_a)} clauses from A and {len(clauses_b)} clauses from B.")

    print("\nStep 2: Building Vector Index and Comparing...")
    # Using 'rules' backend so it works without API keys
    report = run_comparison_sync(
        clauses_a,
        clauses_b,
        policy_a_name="Standard (A)",
        policy_b_name="Implementation (B)",
        llm_backend="rules"
    )

    print("\nStep 3: Saving Report...")
    with open("partial_compliance_report.json", "w", encoding='utf-8') as f:
        json.dump(report, f, indent=4)
    
    print("Done! Report saved to partial_compliance_report.json")

if __name__ == "__main__":
    demo()
