"""
run_api_test.py - FastAPI Endpoint Load & Routing Verification Script
----------------------------------------------------------------------
Starts the FastAPI server in a background thread, then fires simulated
traffic spikes (multiple concurrent requests) at /process-frame using the
local test image, and prints a formatted summary of routing decisions.

Usage:
    .venv\Scripts\python.exe run_api_test.py
"""
import os
import sys
import time
import json
import threading
import statistics
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests
import uvicorn

# Ensure project root is importable
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

# ── Config ─────────────────────────────────────────────────────────────────
SERVER_HOST  = "127.0.0.1"
SERVER_PORT  = 8765
BASE_URL     = f"http://{SERVER_HOST}:{SERVER_PORT}"
TEST_IMAGE   = "sample_media/test_traffic_pedestrians.jpg"
SPIKE_ROUNDS = [
    {"label": "Warm-up    (  1 req)", "concurrency": 1},
    {"label": "Normal     (  5 req)", "concurrency": 5},
    {"label": "Spike      ( 15 req)", "concurrency": 15},
]
STARTUP_WAIT_S = 6   # seconds to wait for uvicorn to be ready


# ── Server bootstrap ────────────────────────────────────────────────────────

def _start_server():
    """Run uvicorn in a daemon thread."""
    uvicorn.run(
        "src.server:app",
        host=SERVER_HOST,
        port=SERVER_PORT,
        log_level="warning",
    )


def _wait_for_server(timeout: float = STARTUP_WAIT_S) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            r = requests.get(f"{BASE_URL}/health", timeout=1)
            if r.status_code == 200:
                return True
        except Exception:
            pass
        time.sleep(0.3)
    return False


# ── Single request helper ───────────────────────────────────────────────────

def _send_one(image_path: str) -> dict:
    t0 = time.perf_counter()
    with open(image_path, "rb") as f:
        resp = requests.post(
            f"{BASE_URL}/process-frame",
            files={"file": ("frame.jpg", f, "image/jpeg")},
            timeout=60,
        )
    elapsed_ms = (time.perf_counter() - t0) * 1000

    if resp.status_code != 200:
        return {"error": resp.text, "http_latency_ms": elapsed_ms}

    data = resp.json()
    data["http_latency_ms"] = round(elapsed_ms, 1)
    return data


# ── Pretty print ────────────────────────────────────────────────────────────

def _print_result(idx: int, r: dict):
    if "error" in r:
        print(f"  [{idx:02d}] ✗ ERROR: {r['error']}")
        return
    dest  = r["routing"]["destination"].upper()
    node  = r["routing"]["node_id"] or "cloud"
    cpu   = r["routing"]["node_cpu_load"]
    score = r["routing"]["suitability_score"]
    sw    = r["semantic_weight"]
    lat   = r["latency_ms"]
    hlat  = r["http_latency_ms"]
    cpu_s = f"{cpu:.1f}%" if cpu is not None else "N/A"
    sc_s  = f"{score:.4f}" if score is not None else "N/A"
    print(
        f"  [{idx:02d}] dest={dest:<5}  node={node:<15}  cpu={cpu_s:<7}  "
        f"score={sc_s:<8}  sw={sw:<3}  gk_lat={lat:.1f}ms  http_lat={hlat:.1f}ms"
    )


def _print_round_stats(results: list, label: str):
    valid   = [r for r in results if "error" not in r]
    errors  = len(results) - len(valid)
    edge_n  = sum(1 for r in valid if r["routing"]["destination"] == "edge")
    cloud_n = sum(1 for r in valid if r["routing"]["destination"] == "cloud")
    lats    = [r["http_latency_ms"] for r in valid]
    print(f"\n  ── {label} summary ──────────────────────────────────────────")
    print(f"     Requests : {len(results)}  |  Errors: {errors}")
    print(f"     Edge     : {edge_n}  |  Cloud : {cloud_n}")
    if lats:
        print(f"     HTTP p50 : {statistics.median(lats):.1f} ms")
        print(f"     HTTP max : {max(lats):.1f} ms")


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    if not Path(TEST_IMAGE).exists():
        print(f"[ERROR] Test image not found: {TEST_IMAGE}")
        print("        Run run_test.py first to download the sample media.")
        sys.exit(1)

    print("=" * 70)
    print("  Semantic-Aware Edge Video Analytics – API Traffic Spike Test")
    print("=" * 70)

    # Start server
    print(f"\n[1] Launching FastAPI server on {BASE_URL} …")
    t = threading.Thread(target=_start_server, daemon=True)
    t.start()

    if not _wait_for_server():
        print("[ERROR] Server did not start in time.")
        sys.exit(1)
    print(f"[1] Server ready ✓")

    # Print node cluster status
    nodes = requests.get(f"{BASE_URL}/nodes").json()["nodes"]
    print(f"\n[2] Edge Node Cluster (initial state):")
    for n in nodes:
        print(f"    {n['node_id']:15s}  region={n['region']:<15}  cpu={n['cpu_load']}%  available={n['available']}")

    # Traffic spike rounds
    print(f"\n[3] Sending traffic spikes …\n")
    for round_cfg in SPIKE_ROUNDS:
        label       = round_cfg["label"]
        concurrency = round_cfg["concurrency"]
        print(f"  ▶ {label}")
        results = []
        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            futures = [pool.submit(_send_one, TEST_IMAGE) for _ in range(concurrency)]
            for idx, fut in enumerate(as_completed(futures), start=1):
                res = fut.result()
                results.append(res)
                _print_result(idx, res)
        _print_round_stats(results, label)
        print()

    # Final stats
    stats = requests.get(f"{BASE_URL}/stats").json()
    print("=" * 70)
    print("  SESSION ROUTING STATISTICS:")
    for k, v in stats.items():
        print(f"    {k:<25} : {v}")
    print("=" * 70)
    print("\n[SUCCESS] API traffic spike test completed.")


if __name__ == "__main__":
    main()
