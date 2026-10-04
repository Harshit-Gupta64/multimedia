"""
FastAPI Server - Semantic-Aware Edge Video Analytics  v0.4.0
-------------------------------------------------------------
New in v0.4:
  - POST /api/v1/execute-task  - secondary analytics endpoint on each node
  - Inter-node forwarder via httpx.AsyncClient: when scheduler picks an edge
    node, the obfuscated payload is forwarded to that node's /api/v1/execute-task
    endpoint. Graceful fallback when a node is unreachable.
  - All CPU-bound work still runs in AnyIO thread pool (asyncio.to_thread).
  - bandwidth_saved_bytes + bandwidth_reduction_pct returned on every frame.
"""
import asyncio
import time
import cv2
import httpx
import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel
from typing import Dict, List, Optional, Any

from src.gatekeeper import PrivacyGatekeeper, GatekeeperResult
from src.scheduler import DynamicScheduler, RoutingDecision

# ---------------------------------------------------------------------------
# App & singletons
# ---------------------------------------------------------------------------

from contextlib import asynccontextmanager

gatekeeper = PrivacyGatekeeper()
scheduler  = DynamicScheduler(num_nodes=3)

# Shared async HTTP client (connection pool, reused across requests)
_http_client: Optional[httpx.AsyncClient] = None


@asynccontextmanager
async def lifespan(app_instance):
    global _http_client
    _http_client = httpx.AsyncClient(timeout=httpx.Timeout(5.0, connect=2.0))
    yield
    if _http_client:
        await _http_client.aclose()


app = FastAPI(
    title="Semantic-Aware Edge Video Analytics",
    version="0.4.0",
    description=(
        "Edge video ingestion with YOLOv8 privacy obfuscation, non-blocking async, "
        "inter-node workload forwarding, and K3s-compatible deployment."
    ),
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# Pydantic schemas
# ---------------------------------------------------------------------------

class DetectionSchema(BaseModel):
    class_id:   int
    class_name: str
    confidence: float
    bbox:       List[int]


class RoutingDecisionSchema(BaseModel):
    destination:       str
    node_id:           Optional[str]
    node_cpu_load:     Optional[float]
    node_endpoint:     Optional[str]
    priority:          int
    suitability_score: Optional[float]
    reason:            str


class BandwidthMetrics(BaseModel):
    raw_frame_bytes:         int
    obfuscated_bytes:        int
    bandwidth_saved_bytes:   int
    bandwidth_reduction_pct: float


class ForwardingResult(BaseModel):
    forwarded:          bool
    target_node:        Optional[str]
    forward_latency_ms: Optional[float]
    execute_task_result: Optional[Dict[str, Any]]
    error:              Optional[str]


class ProcessFrameResponse(BaseModel):
    semantic_weight:  int
    latency_ms:       float
    class_breakdown:  Dict[str, int]
    detections:       List[DetectionSchema]
    routing:          RoutingDecisionSchema
    bandwidth:        BandwidthMetrics
    forwarding:       ForwardingResult
    node_snapshot:    List[Dict]


class ExecuteTaskResponse(BaseModel):
    node_id:            str
    object_count:       int
    class_counts:       Dict[str, int]
    exec_latency_ms:    float
    status:             str


# ---------------------------------------------------------------------------
# CPU-bound pipeline (runs in thread pool)
# ---------------------------------------------------------------------------

def _run_pipeline(
    contents: bytes,
) -> tuple[GatekeeperResult, RoutingDecision, bytes, List[Dict]]:
    """Sync function: YOLO inference + obfuscation + JPEG encode."""
    nparr = np.frombuffer(contents, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if frame is None:
        raise ValueError("Invalid or unreadable image payload")

    result   = gatekeeper.process_frame(frame, obfuscate=True)
    decision = scheduler.route_frame(result.semantic_weight, result.latency_ms)
    snapshot = scheduler.get_node_status()

    ok, buf = cv2.imencode(".jpg", result.obfuscated_frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
    obf_bytes = buf.tobytes() if ok else b""

    return result, decision, obf_bytes, snapshot


def _compute_bandwidth(raw: bytes, obf: bytes) -> BandwidthMetrics:
    raw_sz = len(raw)
    obf_sz = len(obf)
    saved  = max(0, raw_sz - obf_sz)
    pct    = round(saved / raw_sz * 100, 2) if raw_sz > 0 else 0.0
    return BandwidthMetrics(
        raw_frame_bytes=raw_sz,
        obfuscated_bytes=obf_sz,
        bandwidth_saved_bytes=saved,
        bandwidth_reduction_pct=pct,
    )


# ---------------------------------------------------------------------------
# Inter-node forwarder
# ---------------------------------------------------------------------------

async def _forward_to_node(
    node_id: str,
    node_endpoint: str,
    obf_bytes: bytes,
) -> ForwardingResult:
    """
    POST the obfuscated frame to the elected edge node's /api/v1/execute-task.
    Returns graceful error on connection failure.
    """
    target_url = f"{node_endpoint.rstrip('/')}/api/v1/execute-task"
    t0 = time.perf_counter()
    try:
        resp = await _http_client.post(
            target_url,
            files={"file": ("obfuscated.jpg", obf_bytes, "image/jpeg")},
        )
        fwd_lat = (time.perf_counter() - t0) * 1000.0
        if resp.status_code == 200:
            return ForwardingResult(
                forwarded=True,
                target_node=node_id,
                forward_latency_ms=round(fwd_lat, 2),
                execute_task_result=resp.json(),
                error=None,
            )
        else:
            return ForwardingResult(
                forwarded=False,
                target_node=node_id,
                forward_latency_ms=round(fwd_lat, 2),
                execute_task_result=None,
                error=f"HTTP {resp.status_code}: {resp.text[:200]}",
            )
    except (httpx.ConnectError, httpx.TimeoutException, OSError) as exc:
        fwd_lat = (time.perf_counter() - t0) * 1000.0
        return ForwardingResult(
            forwarded=False,
            target_node=node_id,
            forward_latency_ms=round(fwd_lat, 2),
            execute_task_result=None,
            error=f"{type(exc).__name__}: node unreachable (graceful fallback)",
        )


# ---------------------------------------------------------------------------
# Ops Endpoints
# ---------------------------------------------------------------------------

@app.get("/health", tags=["Ops"])
def health_check():
    return {"status": "ok", "service": "Edge Privacy Analytics", "version": "0.4.0"}


@app.get("/nodes", tags=["Ops"])
def list_nodes():
    return {"nodes": scheduler.get_node_status()}


@app.get("/stats", tags=["Ops"])
def routing_stats():
    return scheduler.get_routing_stats()


# ---------------------------------------------------------------------------
# Secondary Analytics Endpoint (runs ON each edge node)
# ---------------------------------------------------------------------------

@app.post("/api/v1/execute-task", response_model=ExecuteTaskResponse, tags=["Edge"])
async def execute_task(file: UploadFile = File(...)):
    """
    Secondary analytics endpoint executed on the elected edge node.
    Receives the already-obfuscated frame and performs lightweight object counting.
    No further obfuscation here – privacy is already guaranteed by the gatekeeper.
    """
    import os
    node_label = os.getenv("NODE_ID", "edge-node-local")

    contents = await file.read()

    def _count_objects(data: bytes) -> tuple[int, Dict[str, int], float]:
        t0    = time.perf_counter()
        nparr = np.frombuffer(data, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if frame is None:
            return 0, {}, 0.0
        # Lightweight re-inference on the already-obfuscated frame
        results = gatekeeper.model(frame, conf=0.3, verbose=False)
        class_counts: Dict[str, int] = {}
        total = 0
        if results and results[0].boxes:
            for box in results[0].boxes:
                cname = gatekeeper.model.names[int(box.cls[0].item())]
                class_counts[cname] = class_counts.get(cname, 0) + 1
                total += 1
        lat = (time.perf_counter() - t0) * 1000.0
        return total, class_counts, lat

    total, class_counts, lat = await asyncio.to_thread(_count_objects, contents)

    return ExecuteTaskResponse(
        node_id=node_label,
        object_count=total,
        class_counts=class_counts,
        exec_latency_ms=round(lat, 2),
        status="ok",
    )


# ---------------------------------------------------------------------------
# Primary Pipeline
# ---------------------------------------------------------------------------

@app.post("/process-frame", response_model=ProcessFrameResponse, tags=["Pipeline"])
async def process_frame(file: UploadFile = File(...)):
    """
    Primary ingestion pipeline:
      1. Read upload (async I/O)
      2. YOLO gatekeeper + obfuscation (thread pool)
      3. DynamicScheduler routing decision
      4. Forward obfuscated frame to elected node's /api/v1/execute-task (async httpx)
      5. Return unified JSON with gatekeeper results, routing, bandwidth, and forward metadata.
    """
    contents = await file.read()

    try:
        result, decision, obf_bytes, snapshot = await asyncio.to_thread(
            _run_pipeline, contents
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    bandwidth = _compute_bandwidth(contents, obf_bytes)

    # Inter-node forwarding
    if decision.destination == "edge" and decision.node_id and decision.node_endpoint and obf_bytes:
        fwd_result = await _forward_to_node(
            decision.node_id, decision.node_endpoint, obf_bytes
        )
    else:
        # Cloud routed or no obfuscated bytes – no forwarding
        fwd_result = ForwardingResult(
            forwarded=False,
            target_node=decision.node_id,
            forward_latency_ms=None,
            execute_task_result=None,
            error="Cloud-routed: no edge node elected" if decision.destination == "cloud" else None,
        )

    return ProcessFrameResponse(
        semantic_weight=result.semantic_weight,
        latency_ms=result.latency_ms,
        class_breakdown=result.class_breakdown,
        detections=[
            DetectionSchema(
                class_id=d.class_id,
                class_name=d.class_name,
                confidence=round(d.confidence, 4),
                bbox=list(d.bbox),
            )
            for d in result.detections
        ],
        routing=RoutingDecisionSchema(
            destination=decision.destination,
            node_id=decision.node_id,
            node_cpu_load=decision.node_cpu_load,
            node_endpoint=decision.node_endpoint,
            priority=decision.priority,
            suitability_score=decision.suitability_score,
            reason=decision.reason,
        ),
        bandwidth=bandwidth,
        forwarding=fwd_result,
        node_snapshot=snapshot,
    )


# ---------------------------------------------------------------------------
# Backward-compat aliases
# ---------------------------------------------------------------------------

@app.post("/api/v1/process-frame", response_model=ProcessFrameResponse, tags=["Legacy"])
async def process_frame_v1(file: UploadFile = File(...)):
    return await process_frame(file)


@app.post("/api/v1/obfuscate-image", tags=["Legacy"])
async def obfuscate_image_endpoint(file: UploadFile = File(...)):
    contents = await file.read()
    try:
        result, _, obf_bytes, _ = await asyncio.to_thread(_run_pipeline, contents)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    if not obf_bytes:
        raise HTTPException(status_code=500, detail="Image encoding failed")
    return Response(content=obf_bytes, media_type="image/jpeg")

