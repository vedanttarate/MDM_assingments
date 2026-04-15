"""
LSTM API — Complete Test Suite
================================
Lab Assignment 5: LSTM-Based Sequence Prediction System

Tests all 3 endpoints:
    GET  /          → Health check
    POST /predict   → Next word prediction
    POST /generate  → Text generation

Usage:
    1. Start the server in Terminal 1:
           python -m uvicorn main:app --reload

    2. Run this file in Terminal 2:
           python test_api.py
"""

import requests
import json
import sys

BASE_URL = "http://127.0.0.1:8000"
PASS = 0
FAIL = 0


# ── Helpers ───────────────────────────────────────────────────────────────────

def section(title):
    print(f"\n{'='*65}")
    print(f"  {title}")
    print('='*65)


def result(passed: bool, message: str):
    global PASS, FAIL
    if passed:
        PASS += 1
        print(f"  ✅ PASS — {message}")
    else:
        FAIL += 1
        print(f"  ❌ FAIL — {message}")


# ── Test 1: Health Check ──────────────────────────────────────────────────────

def test_health():
    section("TEST 1: Health Check  →  GET /")
    try:
        r = requests.get(f"{BASE_URL}/", timeout=10)
        print(f"  Status Code  : {r.status_code}")

        result(r.status_code == 200, f"HTTP 200 OK")

        data = r.json()
        print(f"  API Status   : {data.get('status')}")
        print(f"  Model Loaded : {data.get('model_loaded')}")
        print(f"  Tok Loaded   : {data.get('tokenizer_loaded')}")
        print(f"  Vocab Size   : {data.get('vocab_size')}")
        print(f"  Seq Length   : {data.get('seq_length')}")
        print(f"  Parameters   : {data.get('total_parameters')}")
        print(f"  Message      : {data.get('message')}")

        result(data.get('model_loaded') == True,     "model_loaded is True")
        result(data.get('tokenizer_loaded') == True, "tokenizer_loaded is True")
        result(data.get('vocab_size', 0) > 0,        f"vocab_size > 0 (got {data.get('vocab_size')})")

    except Exception as e:
        result(False, f"Request failed: {e}")


# ── Test 2: Basic Prediction ──────────────────────────────────────────────────

def test_predict_basic():
    section("TEST 2: Basic Next Word Prediction  →  POST /predict")

    payload = {
        "text"       : "alice was beginning to get very tired of sitting by",
        "top_k"      : 5,
        "temperature": 1.0
    }
    print(f"  Input: \"{payload['text']}\"")

    try:
        r = requests.post(f"{BASE_URL}/predict", json=payload, timeout=30)
        print(f"  Status Code    : {r.status_code}")
        result(r.status_code == 200, "HTTP 200 OK")

        if r.status_code == 200:
            data = r.json()
            print(f"  Predicted word : \"{data['predicted_word']}\"")
            print(f"  Cleaned input  : \"{data['cleaned_input']}\"")
            print(f"\n  Top {len(data['top_candidates'])} Candidates:")
            for cand in data['top_candidates']:
                bar = '█' * int(cand['probability'] * 40)
                print(f"    {cand['word']:20s} {bar:<40} {cand['percentage']}")
            print(f"\n  Model Info:")
            for k, v in data['model_info'].items():
                print(f"    {k}: {v}")

            result(len(data['predicted_word']) > 0,        "predicted_word is not empty")
            result(len(data['top_candidates']) == 5,       "returned 5 candidates")
            result('model_info' in data,                   "model_info present in response")
        else:
            print(f"  Error: {r.json()}")

    except Exception as e:
        result(False, f"Request failed: {e}")


# ── Test 3: Multiple Seed Texts ───────────────────────────────────────────────

def test_predict_multiple_seeds():
    section("TEST 3: Multiple Seed Texts  →  POST /predict")

    seeds = [
        "the queen shouted off with",
        "curiouser and curiouser cried",
        "the white rabbit looked at its watch",
        "down the rabbit hole alice fell",
        "the mad hatter poured the tea"
    ]

    all_passed = True
    for seed in seeds:
        payload = {"text": seed, "top_k": 3, "temperature": 1.0}
        try:
            r = requests.post(f"{BASE_URL}/predict", json=payload, timeout=30)
            if r.status_code == 200:
                data   = r.json()
                top3   = [c['word'] for c in data['top_candidates'][:3]]
                best   = data['predicted_word']
                print(f"  Input : \"{seed}\"")
                print(f"  → Best: \"{best}\"  |  Top 3: {top3}\n")
            else:
                print(f"  ❌ Error for: \"{seed}\" — {r.json()}")
                all_passed = False
        except Exception as e:
            print(f"  ❌ Exception for: \"{seed}\" — {e}")
            all_passed = False

    result(all_passed, "All 5 seed texts returned valid predictions")


# ── Test 4: Temperature Effect ────────────────────────────────────────────────

def test_temperature_effect():
    section("TEST 4: Temperature Effect on Predictions  →  POST /predict")

    seed         = "alice looked up and saw"
    temperatures = [0.5, 1.0, 1.5, 2.0]

    print(f"  Seed: \"{seed}\"\n")
    all_passed = True

    for temp in temperatures:
        payload = {"text": seed, "top_k": 5, "temperature": temp}
        try:
            r = requests.post(f"{BASE_URL}/predict", json=payload, timeout=30)
            if r.status_code == 200:
                data = r.json()
                top5 = [c['word'] for c in data['top_candidates'][:5]]
                best = data['predicted_word']
                print(f"  Temp {temp:.1f}  → Best: \"{best:15s}\"  Top5: {top5}")
            else:
                print(f"  ❌ Error at temp={temp}")
                all_passed = False
        except Exception as e:
            print(f"  ❌ Exception at temp={temp}: {e}")
            all_passed = False

    result(all_passed, "All temperature values returned valid responses")


# ── Test 5: top_k Values ──────────────────────────────────────────────────────

def test_topk_values():
    section("TEST 5: Different top_k Values  →  POST /predict")

    seed    = "alice was curious"
    topk_vals = [1, 3, 5, 10, 20]
    all_passed = True

    for k in topk_vals:
        payload = {"text": seed, "top_k": k, "temperature": 1.0}
        try:
            r = requests.post(f"{BASE_URL}/predict", json=payload, timeout=30)
            if r.status_code == 200:
                data    = r.json()
                returned = len(data['top_candidates'])
                match   = returned == k
                print(f"  top_k={k:2d} → returned {returned} candidates  {'✅' if match else '❌'}")
                if not match:
                    all_passed = False
            else:
                print(f"  ❌ Error for top_k={k}")
                all_passed = False
        except Exception as e:
            print(f"  ❌ Exception for top_k={k}: {e}")
            all_passed = False

    result(all_passed, "All top_k values returned correct candidate counts")


# ── Test 6: Text Generation ───────────────────────────────────────────────────

def test_generate():
    section("TEST 6: Text Generation  →  POST /generate")

    test_cases = [
        {"text": "alice fell down the rabbit hole", "n_words": 15, "temperature": 1.0},
        {"text": "the mad hatter said",             "n_words": 10, "temperature": 0.5},
        {"text": "the queen of hearts",             "n_words": 20, "temperature": 1.5},
    ]

    all_passed = True
    for tc in test_cases:
        params = {"text": tc["text"], "n_words": tc["n_words"], "temperature": tc["temperature"]}
        try:
            r = requests.post(f"{BASE_URL}/generate", params=params, timeout=60)
            if r.status_code == 200:
                data  = r.json()
                added = data.get('words_added', 0)
                gen   = data.get('generated_text', '')
                print(f"  Seed  : \"{tc['text']}\"")
                print(f"  Temp  : {tc['temperature']}  |  Words added: {added}")
                print(f"  Output: \"{gen}\"")
                print()
                result(added == tc["n_words"], f"Generated exactly {tc['n_words']} words (got {added})")
            else:
                print(f"  ❌ Error: {r.json()}")
                all_passed = False
        except Exception as e:
            print(f"  ❌ Exception: {e}")
            all_passed = False


# ── Test 7: Error Handling ────────────────────────────────────────────────────

def test_error_handling():
    section("TEST 7: Error Handling — Edge Cases")

    # 7a: Empty input
    print("  7a — Empty text input:")
    try:
        r = requests.post(f"{BASE_URL}/predict",
                          json={"text": "", "top_k": 5}, timeout=10)
        print(f"       Status (expect 400 or 422): {r.status_code}")
        result(r.status_code in [400, 422], f"Empty input rejected with {r.status_code}")
    except Exception as e:
        result(False, f"Exception: {e}")

    # 7b: top_k too large (>20)
    print("\n  7b — top_k=100 (exceeds max of 20):")
    try:
        r = requests.post(f"{BASE_URL}/predict",
                          json={"text": "alice", "top_k": 100}, timeout=10)
        print(f"       Status (expect 422): {r.status_code}")
        result(r.status_code == 422, f"Over-limit top_k rejected with 422")
    except Exception as e:
        result(False, f"Exception: {e}")

    # 7c: temperature out of range
    print("\n  7c — temperature=0.0 (below min of 0.1):")
    try:
        r = requests.post(f"{BASE_URL}/predict",
                          json={"text": "alice", "top_k": 5, "temperature": 0.0}, timeout=10)
        print(f"       Status (expect 422): {r.status_code}")
        result(r.status_code == 422, f"Under-limit temperature rejected with 422")
    except Exception as e:
        result(False, f"Exception: {e}")

    # 7d: missing required field
    print("\n  7d — Missing 'text' field:")
    try:
        r = requests.post(f"{BASE_URL}/predict",
                          json={"top_k": 5}, timeout=10)
        print(f"       Status (expect 422): {r.status_code}")
        result(r.status_code == 422, f"Missing field rejected with 422")
    except Exception as e:
        result(False, f"Exception: {e}")


# ── Test 8: Response Schema Validation ────────────────────────────────────────

def test_response_schema():
    section("TEST 8: Response Schema Validation")

    payload = {"text": "the rabbit hole was very deep", "top_k": 3}
    try:
        r = requests.post(f"{BASE_URL}/predict", json=payload, timeout=30)
        result(r.status_code == 200, "HTTP 200 OK")

        data = r.json()

        # Check all required fields
        required_fields = ['input_text', 'cleaned_input', 'predicted_word',
                           'top_candidates', 'model_info']
        for field in required_fields:
            result(field in data, f"Response contains field '{field}'")

        # Check candidate structure
        if data.get('top_candidates'):
            cand = data['top_candidates'][0]
            for cand_field in ['word', 'probability', 'percentage']:
                result(cand_field in cand, f"Candidate contains field '{cand_field}'")

        # Check probability sums to <= 1
        total_prob = sum(c['probability'] for c in data['top_candidates'])
        result(total_prob <= 1.001, f"Probabilities valid (sum={total_prob:.4f} ≤ 1)")

    except Exception as e:
        result(False, f"Request failed: {e}")


# ── Final Report ──────────────────────────────────────────────────────────────

def print_final_report():
    total = PASS + FAIL
    print(f"\n{'='*65}")
    print(f"  FINAL TEST REPORT")
    print(f"{'='*65}")
    print(f"  Total Tests : {total}")
    print(f"  ✅ Passed   : {PASS}")
    print(f"  ❌ Failed   : {FAIL}")
    print(f"  Score       : {PASS}/{total} ({PASS/total*100:.1f}%)" if total > 0 else "")
    if FAIL == 0:
        print(f"\n  🎉 ALL TESTS PASSED — API is working correctly!")
    else:
        print(f"\n  ⚠️  Some tests failed. Check server logs for details.")
    print('='*65)


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\n🚀 LSTM TEXT PREDICTION — API TEST SUITE")
    print(f"   Server: {BASE_URL}")
    print("   Make sure server is running: python -m uvicorn main:app --reload")

    # Check server is reachable before running tests
    try:
        r = requests.get(f"{BASE_URL}/", timeout=5)
    except requests.exceptions.ConnectionError:
        print("\n❌ Cannot connect to server.")
        print(f"   Start it first: python -m uvicorn main:app --reload")
        sys.exit(1)

    # Run all tests
    test_health()
    test_predict_basic()
    test_predict_multiple_seeds()
    test_temperature_effect()
    test_topk_values()
    test_generate()
    test_error_handling()
    test_response_schema()

    print_final_report()