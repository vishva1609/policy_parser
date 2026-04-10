"""
test_compliance.py
==================
Smoke-test for the full compliance backend.

Run AFTER starting the API:
    uvicorn compliance_api:app --port 8000

Usage:
    python test_compliance.py
"""

import json
import time
import sys
import requests

BASE = "http://localhost:8000"

# ─────────────────────────────────────────────────────────────────────────────
#  Helpers
# ─────────────────────────────────────────────────────────────────────────────

def ok(label):
    print(f"  ✅  {label}")

def fail(label, detail=""):
    print(f"  ❌  {label}  →  {detail}")
    sys.exit(1)

def section(title):
    print(f"\n{'─'*60}")
    print(f"  {title}")
    print(f"{'─'*60}")

# ─────────────────────────────────────────────────────────────────────────────
#  Sample clause data (used for the /compare JSON endpoint)
# ─────────────────────────────────────────────────────────────────────────────

CLAUSES_A = [
    {
        "file_name": "policy_a.txt",
        "clause_id": "1.1",
        "level": "sub-clause",
        "title": "Data Encryption",
        "text": "All data must be encrypted at rest using AES-256. Encryption keys must be rotated every 90 days.",
        "keywords": ["encrypt", "aes", "keys"],
        "context": "All data must be encrypted at rest using AES-256."
    },
    {
        "file_name": "policy_a.txt",
        "clause_id": "1.2",
        "level": "sub-clause",
        "title": "Access Control",
        "text": "Access to systems must be granted on a least-privilege basis. Multi-factor authentication (MFA) is mandatory for all privileged accounts.",
        "keywords": ["access", "mfa", "privilege"],
        "context": "Access to systems must be granted on a least-privilege basis."
    },
    {
        "file_name": "policy_a.txt",
        "clause_id": "1.3",
        "level": "sub-clause",
        "title": "Audit Logging",
        "text": "All system access events must be logged. Logs must be retained for a minimum of 12 months and reviewed quarterly.",
        "keywords": ["audit", "log", "retain"],
        "context": "All system access events must be logged."
    },
    {
        "file_name": "policy_a.txt",
        "clause_id": "1.4",
        "level": "sub-clause",
        "title": "Data Breach Notification",
        "text": "In the event of a data breach, the relevant authorities must be notified within 72 hours.",
        "keywords": ["breach", "notify", "72 hours"],
        "context": "In the event of a data breach, the relevant authorities must be notified within 72 hours."
    },
    {
        "file_name": "policy_a.txt",
        "clause_id": "1.5",
        "level": "sub-clause",
        "title": "Penetration Testing",
        "text": "Annual penetration testing must be conducted by an independent third party. Results must be reviewed by the CISO.",
        "keywords": ["pentest", "penetration", "ciso"],
        "context": "Annual penetration testing must be conducted by an independent third party."
    },
]

CLAUSES_B = [
    {
        "file_name": "policy_b.txt",
        "clause_id": "A-1",
        "level": "clause",
        "title": "Encryption Standards",
        "text": "The organisation shall encrypt sensitive data at rest using AES-256 encryption. Key management procedures should follow NIST guidelines.",
        "keywords": ["encrypt", "aes", "nist"],
        "context": "The organisation shall encrypt sensitive data at rest using AES-256 encryption."
    },
    {
        "file_name": "policy_b.txt",
        "clause_id": "A-2",
        "level": "clause",
        "title": "User Access Management",
        "text": "User access rights are reviewed semi-annually. Administrator accounts require two-factor authentication.",
        "keywords": ["access", "2fa", "administrator"],
        "context": "User access rights are reviewed semi-annually."
    },
    {
        "file_name": "policy_b.txt",
        "clause_id": "A-3",
        "level": "clause",
        "title": "Log Management",
        "text": "Security events shall be logged and retained for 6 months. Log integrity must be maintained.",
        "keywords": ["log", "security", "retain"],
        "context": "Security events shall be logged and retained for 6 months."
    },
    {
        "file_name": "policy_b.txt",
        "clause_id": "A-4",
        "level": "clause",
        "title": "Privacy and Data Handling",
        "text": "Customer personal data must be handled in accordance with GDPR. Data subjects have the right to erasure.",
        "keywords": ["privacy", "gdpr", "personal"],
        "context": "Customer personal data must be handled in accordance with GDPR."
    },
]

# ─────────────────────────────────────────────────────────────────────────────
#  Tests
# ─────────────────────────────────────────────────────────────────────────────

def test_health():
    section("TEST 1 — GET /health")
    r = requests.get(f"{BASE}/health", timeout=10)
    assert r.status_code == 200, fail("Health check", r.text)
    body = r.json()
    assert body.get("status") == "ok"
    ok(f"Health OK  →  {body}")


def test_parse_text():
    section("TEST 2 — POST /parse/text")
    payload = {
        "text": (
            "1. Security Policy\n"
            "All employees must follow data security guidelines.\n\n"
            "1.1 Encryption\n"
            "Data must be encrypted using AES-256.\n\n"
            "1.2 Access Control\n"
            "Access shall be granted on least-privilege basis.\n"
        ),
        "file_name": "test_input.txt"
    }
    r = requests.post(f"{BASE}/parse/text", json=payload, timeout=30)
    assert r.status_code == 200, fail("Parse text", r.text)
    body = r.json()
    segments = body.get("segments", [])
    assert len(segments) > 0, fail("Parse text", "No segments returned")
    ok(f"Parsed {body['total_segments']} segments from raw text")
    for s in segments[:3]:
        print(f"       clause_id={s['clause_id']}  title={s['title']!r}  level={s['level']}")


def test_parse_pdf():
    section("TEST 3 — POST /parse  (PDF upload)")
    import os
    pdf = "app/data/bank_policy.pdf"
    if not os.path.exists(pdf):
        print("  ⚠️  bank_policy.pdf not found — skipping PDF parse test")
        return
    with open(pdf, "rb") as fh:
        r = requests.post(f"{BASE}/parse", files={"file": (pdf, fh, "application/pdf")}, timeout=60)
    assert r.status_code == 200, fail("Parse PDF", r.text)
    body = r.json()
    ok(f"PDF parsed  →  {body['total_segments']} segments from {body['file_name']}")


def test_compare_json():
    section("TEST 4 — POST /compare  (JSON clause lists)")
    payload = {
        "clauses_a": CLAUSES_A,
        "clauses_b": CLAUSES_B,
        "policy_a_name": "Company A Policy",
        "policy_b_name": "Company B Policy",
        "top_k": 3,
        "llm_backend": "rules"
    }
    r = requests.post(f"{BASE}/compare", json=payload, timeout=120)
    assert r.status_code == 200, fail("Compare JSON", r.text)
    report = r.json()

    pct      = report.get("overall_compliance_pct", 0)
    total    = report.get("total_clauses", 0)
    compliant = report.get("compliant", 0)
    partial  = report.get("partial_matches", 0)
    gaps     = report.get("gaps", 0)
    missing  = report.get("missing", 0)

    ok(f"Comparison complete")
    print(f"       Overall compliance : {pct:.1f}%")
    print(f"       Total clauses      : {total}")
    print(f"       Compliant (Yes)    : {compliant}")
    print(f"       Partial            : {partial}")
    print(f"       Gaps               : {gaps}")
    print(f"       Missing            : {missing}")

    # Category scores
    print("\n       Category scores:")
    for cat, v in report.get("category_scores", {}).items():
        print(f"         {cat:20s} → {v['compliance_pct']:.1f}%")

    # Clause details
    print("\n       Clause-by-clause:")
    for c in report.get("clause_details", []):
        status = c["status"]
        icon   = "✅" if status == "Yes" else "⚠️ " if status == "Partial" else "❌"
        print(f"         {icon}  {c['clause_a_id']:6s} {c['clause_a_title']:25s}"
              f"  →  {c['best_match_id']:5s}  sim={c['best_similarity']:.2f}"
              f"  [{c['strength']}]")

    # Critical gaps
    cg = report.get("critical_gaps", [])
    if cg:
        print(f"\n       ⚠️  Critical Gaps ({len(cg)}):")
        for g in cg:
            print(f"         • {g['clause_a_id']} {g['clause_a_title']}")
            print(f"           Gap: {g['gap']}")

    # Save full report
    with open("test_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    ok("Full report saved → test_report.json")

    return report


def test_compare_pdf_async():
    section("TEST 5 — POST /compare/pdf  (async PDF job)")
    import os
    pdfs = [p for p in ["app/data/bank_policy.pdf", "app/data/policy_3.pdf"] if os.path.exists(p)]
    if len(pdfs) < 2:
        print("  ⚠️  Need 2 PDF files in project root — skipping async PDF test")
        print(f"       Found: {pdfs}")
        return None

    with open(pdfs[0], "rb") as fa, open(pdfs[1], "rb") as fb:
        r = requests.post(
            f"{BASE}/compare/pdf?top_k=3&llm_backend=rules",
            files={
                "file_a": (pdfs[0], fa, "application/pdf"),
                "file_b": (pdfs[1], fb, "application/pdf"),
            },
            timeout=30,
        )
    assert r.status_code == 202, fail("Compare PDF submit", r.text)
    job_id = r.json()["job_id"]
    ok(f"Job submitted  →  job_id={job_id}")

    # Poll
    print("       Polling ", end="", flush=True)
    for _ in range(60):          # up to 5 minutes
        time.sleep(5)
        poll = requests.get(f"{BASE}/compare/{job_id}", timeout=15).json()
        status = poll["status"]
        print(".", end="", flush=True)
        if status == "done":
            print()
            ok(f"Job done  →  compliance={poll['summary']['overall_compliance_pct']}%")
            return job_id
        elif status == "error":
            print()
            fail("async job error", poll.get("error"))

    print()
    fail("async job timed out")


def test_export(job_id: str | None):
    section("TEST 6 — GET /compare/{job_id}/export")
    if not job_id:
        print("  ⚠️  No job_id — skipping export test")
        return

    for fmt in ["excel", "pdf"]:
        r = requests.get(f"{BASE}/compare/{job_id}/export?format={fmt}", timeout=30)
        assert r.status_code == 200, fail(f"Export {fmt}", r.text[:300])
        fname = f"test_export.{'xlsx' if fmt == 'excel' else 'pdf'}"
        with open(fname, "wb") as f:
            f.write(r.content)
        ok(f"Export {fmt}  →  {fname}  ({len(r.content):,} bytes)")


# ─────────────────────────────────────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\n" + "="*60)
    print("   Policy Compliance Backend — Smoke Test")
    print("="*60)

    test_health()
    test_parse_text()
    test_parse_pdf()
    report = test_compare_json()
    job_id = test_compare_pdf_async()
    test_export(job_id)

    print("\n" + "="*60)
    print("   ALL TESTS PASSED ✅")
    print("="*60 + "\n")
