# Semantic-Aware Edge Video Analytics

**Privacy-Preserving Inference Pipeline with Dynamic Scheduling and K3s Orchestration**

A modular, production-ready edge video analytics system that detects sensitive entities (people, vehicles) in real-time using YOLOv8, obfuscates them with Gaussian blur before transmission, and routes sanitized frames to optimal compute nodes via a suitability-score scheduling algorithm.

---

## Architecture

`mermaid
graph TD
    subgraph Ingestion
        VS[Video Source<br>Camera / File / RTSP]
        SP[Stream Processor<br>src/stream_processor.py]
    end

    subgraph "FastAPI Server (src/server.py)"
        API[POST /process-frame]
        GK[Privacy Gatekeeper<br>YOLOv8n + Gaussian Blur]
        SCH[Dynamic Scheduler<br>Suitability Score]
        FWD[httpx Forwarder]
    end

    subgraph "Edge Cluster (K3s / Docker)"
        N1[edge-node-alpha<br>2 CPU / 1Gi<br>us-west]
        N2[edge-node-beta<br>1 CPU / 512Mi<br>eu-central]
        N3[edge-node-gamma<br>0.5 CPU / 256Mi<br>ap-southeast]
        CLD[Cloud Fallback]
    end

    VS --> SP --> API
    API -->|asyncio.to_thread| GK
    GK -->|semantic_weight| SCH
    SCH -->|node + endpoint| FWD
    FWD -->|obfuscated JPEG| N1
    FWD -->|obfuscated JPEG| N2
    FWD -->|obfuscated JPEG| N3
    SCH -->|all overloaded| CLD
`

---

## Prerequisites

| Requirement | Version | Notes |
|---|---|---|
| Python | 3.11+ | With pip |
| Docker Desktop | 24+ | For multi-node simulation |
| K3s / kubectl | 1.28+ | Optional: Kubernetes deployment |
| Git | 2.40+ | Repository management |

---

## Quick Start

### 1. Local Development

`powershell
# Clone and enter project
cd c:\Projects\Multimedia

# Create and activate virtual environment
python -m venv .venv
.venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt

# Run visual gatekeeper test
python run_test.py
# -> Saves visual_test_output.jpg (side-by-side comparison)

# Run all unit tests (37 tests)
pytest tests/ -v
`

### 2. Run Benchmarks

`powershell
# Empirical benchmark (45 runs across 3 density scenarios)
python src/benchmark.py
# -> Exports benchmark_results.csv

# Generate publication-ready charts
python src/visualize_metrics.py
# -> reports/latency_distribution.png
# -> reports/bandwidth_savings.png
`

### 3. Start the API Server

`powershell
uvicorn src.server:app --reload --port 8000

# Swagger UI: http://127.0.0.1:8000/docs
# Health:     http://127.0.0.1:8000/health
# Nodes:      http://127.0.0.1:8000/nodes
`

### 4. Video Stream Processing

`powershell
# Start the server (in one terminal)
uvicorn src.server:app --port 8000

# Process a video stream (in another terminal)
python src/stream_processor.py --video sample_media/test_video.mp4

# With live display window
python src/stream_processor.py --video sample_media/test_video.mp4 --display

# Output: reports/demo_output.mp4
`

### 5. Docker Compose Cluster

`powershell
# Build and start 3-node edge cluster
docker-compose up --build -d

# Verify all nodes are healthy
curl http://localhost:8001/health
curl http://localhost:8002/health
curl http://localhost:8003/health

# Submit a frame to the high-capacity node
curl -X POST http://localhost:8001/process-frame -F "file=@sample_media/test_traffic_pedestrians.jpg"

# Tear down
docker-compose down
`

### 6. K3s Kubernetes Deployment

`ash
# Apply manifests
kubectl apply -f k8s/namespace.yaml
kubectl apply -f k8s/services.yaml
kubectl apply -f k8s/node-alpha-deployment.yaml
kubectl apply -f k8s/node-beta-deployment.yaml
kubectl apply -f k8s/node-gamma-deployment.yaml
kubectl apply -f k8s/ingress.yaml

# Verify pods
kubectl -n edge-analytics get pods,svc

# Test via ingress
curl -X POST http://edge-analytics.local/ingest/process-frame -F "file=@sample_media/test_traffic_pedestrians.jpg"
`

### 7. Run Everything (Automated)

`powershell
.\run_all.ps1                        # Full pipeline (tests + benchmark + charts + stream)
.\run_all.ps1 -SkipStream            # Skip video stream simulation
.\run_all.ps1 -MaxFrames 60          # Process more frames
`

---

## Project Structure

`
Multimedia/
+-- .venv/                           # Python 3.11 virtual environment
+-- requirements.txt                 # Dependencies
+-- Dockerfile                       # Multi-stage production build
+-- docker-compose.yml               # 3-node heterogeneous cluster
+-- run_all.ps1                      # Automated execution pipeline
+-- run_test.py                      # Visual gatekeeper test
+-- run_api_test.py                  # API traffic spike test
+-- README.md                        # This file
+-- REPORT.md                        # Full research report
+-- architecture.md                  # System architecture documentation
+-- benchmark_results.csv            # Empirical benchmark data (45 rows)
+-- visual_test_output.jpg           # Side-by-side obfuscation comparison
+--
+-- src/
|   +-- __init__.py
|   +-- gatekeeper.py                # YOLOv8 detection + OpenCV obfuscation
|   +-- scheduler.py                 # DynamicScheduler + suitability score
|   +-- server.py                    # FastAPI server (async, thread-pool, forwarding)
|   +-- benchmark.py                 # Parameterized benchmark runner
|   +-- visualize_metrics.py         # Headless chart generation (matplotlib)
|   +-- stream_processor.py          # Live video stream test harness
+--
+-- k8s/
|   +-- namespace.yaml               # edge-analytics namespace
|   +-- node-alpha-deployment.yaml   # High-tier node (2 CPU / 1Gi)
|   +-- node-beta-deployment.yaml    # Mid-tier node (1 CPU / 512Mi)
|   +-- node-gamma-deployment.yaml   # Constrained node (0.5 CPU / 256Mi)
|   +-- services.yaml                # ClusterIP services + PVC
|   +-- ingress.yaml                 # Traefik Ingress routing
+--
+-- reports/
|   +-- latency_distribution.png     # Inference vs scheduler latency charts
|   +-- bandwidth_savings.png        # Raw vs obfuscated bandwidth chart
|   +-- demo_output.mp4              # Processed video demo
+--
+-- sample_media/
|   +-- test_traffic_pedestrians.jpg # Sample test image
|   +-- test_video.mp4               # Synthetic test video (auto-generated)
+--
+-- tests/
    +-- test_gatekeeper.py           # 2 tests
    +-- test_scheduler.py            # 11 tests
    +-- test_benchmark.py            # 13 tests
    +-- test_server.py               # 11 tests
`

---

## Empirical Results Summary

Test image: us.jpg (1080x810 px, 4 detected entities) | Model: YOLOv8 nano | CPU-only inference

| Metric | Value |
|---|---|
| GK Inference Latency (steady-state) | **~115-130 ms** |
| GK Inference Latency (p95) | **142.9 ms** |
| Scheduler Decision Latency | **< 100 us** |
| Bandwidth Reduction | **34.78%** |
| Raw Frame Size | 210 KB |
| Obfuscated JPEG Size | 137 KB |
| Bytes Saved per Frame | **73 KB** |

| Scenario | Cloud Offload Rate | Avg GK Latency |
|---|---|---|
| Low Density (0-2 obj) | 0% | 118 ms |
| Medium Density (3-5 obj) | 0% | 119 ms |
| High Density (>5 obj) | 100% | 127 ms |

---

## API Endpoints

| Endpoint | Method | Description |
|---|---|---|
| /health | GET | Liveness probe |
| /nodes | GET | Edge node cluster status |
| /stats | GET | Session routing statistics |
| /process-frame | POST | Primary pipeline (gatekeeper + scheduler + forwarding) |
| /api/v1/execute-task | POST | Secondary analytics on sanitized frames |
| /api/v1/obfuscate-image | POST | Returns obfuscated JPEG binary |

---

## Test Suite

`
37 passed in 8.16s
`

| Suite | File | Tests |
|---|---|---|
| Gatekeeper | 	ests/test_gatekeeper.py | 2 |
| Scheduler | 	ests/test_scheduler.py | 11 |
| Benchmark | 	ests/test_benchmark.py | 13 |
| Server | 	ests/test_server.py | 11 |
| **Total** | | **37** |

---

## Routing Algorithm

Suitability Score: score(W, C) = (100 - C) / (W + 1)

Where W = semantic weight (detected objects), C = node CPU load (%).

Ineligibility gate: W > 5 AND C > 75% prevents heavy frames on overloaded nodes.
All nodes ineligible -> **cloud fallback**.

---

## License

MIT
