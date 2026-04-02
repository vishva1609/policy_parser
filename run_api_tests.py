"""
Policy Parser API - Full Test Runner
Runs all 3 endpoints and shows pass/fail with response details.
"""
import json
import sys
import time
import io
import requests

BASE = "http://localhost:8000"
PDF_PATH = r"C:\policy_parser\bank_policy.pdf"
PASS = "[PASS]"
FAIL = "[FAIL]"

results = []

def separator(char="-", width=65):
    print(char * width)

def run_tests():
    separator("═")
    print("  POLICY PARSER API  —  Full Test Suite")
    separator("═")
    print()

    # ── TEST 1: GET /health ─────────────────────────────────────────
    separator()
    print("TEST 1 │ GET /health")
    separator()
    try:
        t0 = time.time()
        r = requests.get(f"{BASE}/health", timeout=10)
        ms = int((time.time() - t0) * 1000)
        body = r.json()
        ok = r.status_code == 200 and body.get("status") == "ok" and body.get("service") == "policy-parser"
        status = PASS if ok else FAIL
        print(f"  Status Code : {r.status_code}")
        print(f"  Response    : {json.dumps(body, indent=14)}")
        print(f"  Time        : {ms} ms")
        print(f"  Result      : {status}")
        results.append(("GET /health", ok))
    except Exception as e:
        print(f"  ERROR: {e}")
        results.append(("GET /health", False))

    print()

    # ── TEST 2: POST /parse/text ────────────────────────────────────
    separator()
    print("TEST 2 │ POST /parse/text")
    separator()
    try:
        payload = {
            "text": (
                "3.1 Introduction\n"
                "This policy applies to all staff.\n"
                "- All employees must comply.\n"
                "- Data must be protected at all times.\n"
                "3.2 Data Retention\n"
                "Data must be kept for a minimum of 7 years.\n"
                "3.3 Security\n"
                "All systems must use AES-256 encryption."
            ),
            "file_name": "test_policy.txt"
        }
        t0 = time.time()
        r = requests.post(f"{BASE}/parse/text", json=payload, timeout=30)
        ms = int((time.time() - t0) * 1000)
        body = r.json()
        segs = body.get("segments", [])
        ok = r.status_code == 200 and body.get("file_name") == "test_policy.txt" and len(segs) > 0
        status = PASS if ok else FAIL
        print(f"  Status Code    : {r.status_code}")
        print(f"  file_name      : {body.get('file_name')}")
        print(f"  total_segments : {body.get('total_segments')}")
        print(f"  Segments found :")
        for s in segs:
            print(f"    [{s['level']:12}] {s['clause_id']:8} │ {s['title'][:45]}")
        print(f"  Time           : {ms} ms")
        print(f"  Result         : {status}")
        results.append(("POST /parse/text", ok))
    except Exception as e:
        print(f"  ERROR: {e}")
        results.append(("POST /parse/text", False))

    print()

    # ── TEST 3: POST /parse (PDF upload) ────────────────────────────
    separator()
    print("TEST 3 │ POST /parse  (PDF upload — bank_policy.pdf)")
    separator()
    try:
        t0 = time.time()
        with open(PDF_PATH, "rb") as f:
            r = requests.post(
                f"{BASE}/parse",
                files={"file": ("bank_policy.pdf", f, "application/pdf")},
                timeout=120
            )
        ms = int((time.time() - t0) * 1000)
        body = r.json()
        segs = body.get("segments", [])
        ok = r.status_code == 200 and body.get("file_name") == "bank_policy.pdf" and len(segs) > 0
        status = PASS if ok else FAIL
        print(f"  Status Code    : {r.status_code}")
        print(f"  file_name      : {body.get('file_name')}")
        print(f"  total_segments : {body.get('total_segments')}")
        print(f"  First 5 segments:")
        for s in segs[:5]:
            print(f"    [{s['level']:12}] {s['clause_id']:8} │ {s['title'][:45]}")
        print(f"  Time           : {ms} ms")
        print(f"  Result         : {status}")
        results.append(("POST /parse (PDF)", ok))
    except Exception as e:
        print(f"  ERROR: {e}")
        results.append(("POST /parse (PDF)", False))

    # ── SUMMARY ─────────────────────────────────────────────────────
    print()
    separator("═")
    print("  FINAL SUMMARY")
    separator("═")
    passed = sum(1 for _, ok in results if ok)
    total = len(results)
    for name, ok in results:
        mark = PASS if ok else FAIL
        print(f"  {mark}  {name}")
    separator()
    print(f"  {passed}/{total} tests passed")
    separator("═")
    return passed == total

if __name__ == "__main__":
    log = open("api_test_results.txt", "w", encoding="utf-8")
    sys.stdout = log
    ok = run_tests()
    log.close()
    sys.exit(0 if ok else 1)
