# System Architecture – Semantic-Aware Edge Video Analytics (v0.3.0)

## Overview

A privacy-first, intelligence-at-the-edge video analytics pipeline that detects
sensitive entities in real-time, obfuscates them before transmission, and routes
frames dynamically to the optimal compute node using a suitability-score algorithm.

---

## High-Level Architecture Diagram

`
┌─────────────────────────────────────────────────────────────────────────┐
│                          VIDEO SOURCE                                   │
│              (IP Camera / RTSP Stream / Uploaded Frame)                 │
└─────────────────────────────┬───────────────────────────────────────────┘
                              │  Raw frame (multipart JPEG upload)
                              ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                 FastAPI Ingestion Server  v0.3.0                        │
│                 src/server.py · Port 8000                               │
│                                                                         │
│  Endpoints:                                                             │
│    POST /process-frame         (primary pipeline)                       │
│    GET  /health  /nodes  /stats                                         │
│    POST /api/v1/process-frame  (legacy alias)                           │
│    POST /api/v1/obfuscate-image                                         │
│                                                                         │
│  ┌─ Event Loop (async) ──────────────────────────────────────────────┐  │
│  │  await file.read()  →  async I/O, non-blocking                    │  │
│  │  await asyncio.to_thread(_run_pipeline, contents)                 │  │
│  │         │                                                         │  │
│  │         └──▶  AnyIO Worker Thread Pool (sync CPU work)            │  │
│  │                   YOLO inference + OpenCV blur + JPEG encode      │  │
│  └───────────────────────────────────────────────────────────────────┘  │
└──────────────────────────────────┬──────────────────────────────────────┘
                                   │  frame bytes
                                   ▼
┌─────────────────────────────────────────────────────────────────────────┐
│              PrivacyGatekeeper  (src/gatekeeper.py)                     │
│                                                                         │
│  YOLOv8n ──▶ BBox Filter (person/car/bus/truck) ──▶ Gaussian Blur      │
│                                                                         │
│  Output: GatekeeperResult                                               │
│    obfuscated_frame   (numpy array)                                     │
│    semantic_weight    (int – sensitive object count)                    │
│    class_breakdown    (dict)                                            │
│    detections[]       (bbox + confidence per entity)                   │
│    latency_ms         (float)                                           │
│                                                                         │
│  Bandwidth Tracking (in server):                                        │
│    raw_frame_bytes       = len(upload payload)                          │
│    obfuscated_bytes      = len(JPEG-encoded obfuscated frame)           │
│    bandwidth_saved_bytes = raw - obfuscated                             │
│    bandwidth_reduction_pct = saved / raw * 100                         │
└──────────────────────────────────┬──────────────────────────────────────┘
                                   │  semantic_weight + latency_ms
                                   ▼
┌─────────────────────────────────────────────────────────────────────────┐
│              DynamicScheduler  (src/scheduler.py)                       │
│                                                                         │
│  3 EdgeNodes with per-tick ±15% CPU drift simulation                    │
│                                                                         │
│  Suitability Score Algorithm:                                           │
│    score(W, C) = (100 - C) / (W + 1)                                   │
│    INELIGIBLE if: (W > 5 AND C > 75%) OR queue_full                    │
│    CLOUD FALLBACK if: no node scores above zero                         │
│                                                                         │
│  Output: RoutingDecision                                                │
│    destination  (edge | cloud | drop)                                   │
│    node_id, node_cpu_load, suitability_score, priority, reason          │
└─────────────┬──────────────────────────────────────┬───────────────────┘
              │ destination=edge                       │ destination=cloud
              ▼                                        ▼
┌─────────────────────────┐               ┌───────────────────────────┐
│    Edge Node Pool       │               │   Cloud Analytics Backend │
│  edge-node-01 (us-west) │               │  (high-capacity offload)  │
│  edge-node-02 (eu-cen)  │               └───────────────────────────┘
│  edge-node-03 (ap-sou)  │
└─────────────────────────┘
`

---

## Async Event Loop Fix (v0.3.0)

**Problem (v0.2):** YOLO inference and OpenCV obfuscation were called directly
inside sync def process_frame, which blocked the event loop for 100-300 ms per
frame. Concurrent requests would queue behind each other.

**Fix (v0.3):** The entire CPU-bound pipeline (_run_pipeline) is a plain
synchronous function executed via syncio.to_thread(). This delegates execution
to the AnyIO thread pool, keeping the event loop free for I/O (file.read, network).

`python
# v0.2 - BLOCKING (bad)
async def process_frame(file: UploadFile = File(...)):
    result = gatekeeper.process_frame(frame)   # blocks event loop!

# v0.3 - NON-BLOCKING (correct)
async def process_frame(file: UploadFile = File(...)):
    result, decision, obf_bytes, snapshot = await asyncio.to_thread(
        _run_pipeline, contents          # runs in thread pool, loop stays free
    )
`

---

## Container Topology (docker-compose.yml)

Three heterogeneous edge service instances simulate real hardware diversity:

`
┌────────────────────────────────────────────────────────────────────────┐
│                       Docker Bridge Network (edge-net)                  │
│                         Subnet: 172.28.0.0/16                          │
│                                                                         │
│  ┌──────────────────────┐  ┌──────────────────────┐  ┌──────────────┐  │
│  │   edge-node-alpha    │  │   edge-node-beta     │  │edge-node-gam │  │
│  │   Port: 8001         │  │   Port: 8002         │  │Port: 8003    │  │
│  │   Tier: High         │  │   Tier: Medium       │  │Tier:Constrai │  │
│  │   Region: us-west    │  │   Region: eu-central │  │Region:ap-sou │  │
│  │                      │  │                      │  │              │  │
│  │   CPU limit: 4.0     │  │   CPU limit: 2.0     │  │CPU limit: 1.0│  │
│  │   Mem limit: 1536M   │  │   Mem limit: 1024M   │  │Mem limit:512M│  │
│  │   CPU reserve: 2.0   │  │   CPU reserve: 1.0   │  │CPU res: 0.5  │  │
│  │   Mem reserve: 768M  │  │   Mem reserve: 512M  │  │Mem res: 256M │  │
│  └──────────────────────┘  └──────────────────────┘  └──────────────┘  │
│                                                                         │
│  Shared Volume: yolo-cache (YOLOv8 weights, pre-cached at build time)  │
└────────────────────────────────────────────────────────────────────────┘
`

### Starting the Cluster
`powershell
docker-compose up --build -d

# Verify nodes
curl http://localhost:8001/health
curl http://localhost:8002/health
curl http://localhost:8003/health

# Process a frame on the constrained node
curl -X POST http://localhost:8003/process-frame -F "file=@sample_media/test_traffic_pedestrians.jpg"
`

---

## File Structure

`
Multimedia/
├── .venv/                           # Python 3.11 virtual environment
├── requirements.txt                 # All Python dependencies
├── Dockerfile                       # Multi-stage production build
├── docker-compose.yml               # 3-node heterogeneous cluster definition
├── architecture.md                  # This document
│
├── run_test.py                      # Standalone visual gatekeeper test
├── run_api_test.py                  # API traffic spike simulation
├── benchmark_results.csv            # Empirical benchmark data (45 rows)
│
├── src/
│   ├── __init__.py
│   ├── gatekeeper.py                # YOLOv8 detection + OpenCV obfuscation
│   ├── scheduler.py                 # DynamicScheduler + suitability score
│   ├── server.py                    # FastAPI server (async, thread-pool offload)
│   └── benchmark.py                 # Parameterized benchmark runner
│
├── sample_media/
│   └── test_traffic_pedestrians.jpg
│
└── tests/
    ├── test_gatekeeper.py           # 2 tests
    ├── test_scheduler.py            # 11 tests
    └── test_benchmark.py            # 13 tests
`

---

## Benchmark Results (Empirical – CPU-Only, Python 3.11, Windows)

Test image: sample_media/test_traffic_pedestrians.jpg (1080×810 px, 4 detected entities)
Model: YOLOv8 nano (yolov8n.pt), CPU inference, confidence threshold: 0.35

| Scenario | Reps | Mean GK Latency | p95 GK Latency | Mean Sched Latency | Mean BW Reduction | Cloud Rate |
|---|---|---|---|---|---|---|
| Low Density (0-2 obj)    | 15 | 297.9 ms* | 130.5 ms | 50 µs | 34.8% | 0.0% |
| Medium Density (3-5 obj) | 15 | 118.6 ms  | 122.3 ms | 40 µs | 34.8% | 0.0% |
| High Density (>5 obj)    | 15 | 126.8 ms  | 142.9 ms | 40 µs | 34.8% | 100.0% |

> **Note on Low Density Mean:** Rep 1 included a 2787 ms cold-start (first YOLO inference loads model weights into
> memory). Subsequent reps (rep 2-15) consistently run at ~115-130 ms. The p95 of 130.5 ms excludes the outlier.

**Key observations:**
- **Gatekeeper latency**: ~115-130 ms per frame after warm-up (CPU-only, no CUDA).
- **Scheduler decision**: < 100 µs — negligible overhead.
- **Bandwidth reduction**: 34.8% consistently — Gaussian blurring reduces JPEG entropy, saving ~75 KB per frame.
- **Cloud offload**: 100% for high-density scenario (node CPU loads simulated high enough to gate heavy frames).
- **Raw frame**: 214,997 bytes; **Obfuscated JPEG**: 140,211 bytes; **Saved**: 74,786 bytes per frame.

---

## API Response Schema (v0.3.0 additions)

`json
{
  "semantic_weight": 4,
  "latency_ms": 119.4,
  "class_breakdown": {"bus": 1, "person": 3},
  "detections": [
    {"class_id": 5, "class_name": "bus", "confidence": 0.87, "bbox": [22, 231, 805, 756]},
    ...
  ],
  "routing": {
    "destination": "edge",
    "node_id": "edge-node-01",
    "node_cpu_load": 45.2,
    "priority": 1,
    "suitability_score": 9.1,
    "reason": "Node edge-node-01 selected (CPU=45.2%, score=9.1000, semantic_weight=4)"
  },
  "bandwidth": {
    "raw_frame_bytes": 214997,
    "obfuscated_bytes": 140211,
    "bandwidth_saved_bytes": 74786,
    "bandwidth_reduction_pct": 34.78
  },
  "node_snapshot": [
    {"node_id": "edge-node-01", "region": "us-west", "cpu_load": 45.2, "current_queue": 1, "available": true},
    ...
  ]
}
`

---

## Running the System

`powershell
# Activate virtual environment
.venv\Scripts\Activate.ps1

# Run visual gatekeeper test
python run_test.py

# Run all unit tests
pytest tests/ -v

# Run empirical benchmark → generates benchmark_results.csv
python src/benchmark.py

# Start API server (single process)
uvicorn src.server:app --reload --port 8000

# Run API traffic spike test
python run_api_test.py

# Build and run 3-node Docker cluster
docker-compose up --build -d
`

---

## Test Coverage Summary

| Test Suite | File | Tests | Status |
|---|---|---|---|
| Gatekeeper | 	ests/test_gatekeeper.py | 2 | ✅ All pass |
| Scheduler | 	ests/test_scheduler.py | 11 | ✅ All pass |
| Benchmark | 	ests/test_benchmark.py | 13 | ✅ All pass |
| **Total** | | **26** | ✅ **26/26** |
