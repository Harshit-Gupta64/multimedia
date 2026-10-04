r"""
src/stream_processor.py - End-to-End Video Stream Test Harness
--------------------------------------------------------------
Processes a local video file (or camera device) frame-by-frame, posting each
frame to the FastAPI /process-frame endpoint, and collecting rolling metrics.

Features:
  - Native FPS playback with frame-level API calls.
  - Rolling FPS, average latency, and cumulative bandwidth saved.
  - Optional live OpenCV GUI window (--display flag) showing raw vs obfuscated
    frames with an overlay banner of routing metadata.
  - Writes the full processed stream to reports/demo_output.mp4.

Usage:
  # Against running server (start uvicorn first):
  .venv\Scripts\python.exe src/stream_processor.py --video sample_media/test_video.mp4
  .venv\Scripts\python.exe src/stream_processor.py --video sample_media/test_video.mp4 --display

  # Use camera:
  .venv\Scripts\python.exe src/stream_processor.py --camera 0 --display
"""
import argparse
import os
import sys
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np
import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


# ---------------------------------------------------------------------------
# Config defaults
# ---------------------------------------------------------------------------

DEFAULT_SERVER = "http://127.0.0.1:8000"
OUTPUT_DIR     = "reports"
OUTPUT_VIDEO   = os.path.join(OUTPUT_DIR, "demo_output.mp4")
MAX_FRAMES     = 300   # safety cap for long videos in test mode


# ---------------------------------------------------------------------------
# Metrics tracker
# ---------------------------------------------------------------------------

class StreamMetrics:
    """Rolling statistics collector over the video stream."""

    def __init__(self, window: int = 30):
        self._window         = window
        self._frame_times    = deque(maxlen=window)
        self.total_frames    = 0
        self.total_latency   = 0.0
        self.total_bw_saved  = 0
        self.total_raw_bytes = 0
        self.total_obf_bytes = 0
        self.node_histogram  = {}
        self.cloud_frames    = 0

    def record(self, latency_ms: float, bw_saved: int, raw_bytes: int,
               obf_bytes: int, node_id: str, destination: str):
        now = time.perf_counter()
        self._frame_times.append(now)
        self.total_frames   += 1
        self.total_latency  += latency_ms
        self.total_bw_saved += bw_saved
        self.total_raw_bytes+= raw_bytes
        self.total_obf_bytes+= obf_bytes
        if destination == "cloud":
            self.cloud_frames += 1
        if node_id:
            self.node_histogram[node_id] = self.node_histogram.get(node_id, 0) + 1

    @property
    def rolling_fps(self) -> float:
        if len(self._frame_times) < 2:
            return 0.0
        elapsed = self._frame_times[-1] - self._frame_times[0]
        return (len(self._frame_times) - 1) / elapsed if elapsed > 0 else 0.0

    @property
    def avg_latency(self) -> float:
        return self.total_latency / self.total_frames if self.total_frames else 0.0

    @property
    def cumulative_bw_pct(self) -> float:
        if self.total_raw_bytes == 0:
            return 0.0
        return max(0, self.total_raw_bytes - self.total_obf_bytes) / self.total_raw_bytes * 100

    def summary(self) -> dict:
        return {
            "total_frames":      self.total_frames,
            "avg_latency_ms":    round(self.avg_latency, 1),
            "rolling_fps":       round(self.rolling_fps, 1),
            "total_bw_saved_kb": round(self.total_bw_saved / 1024, 1),
            "cumulative_bw_pct": round(self.cumulative_bw_pct, 1),
            "cloud_frames":      self.cloud_frames,
            "node_histogram":    self.node_histogram,
        }


# ---------------------------------------------------------------------------
# Frame poster
# ---------------------------------------------------------------------------

def post_frame(server_url: str, frame: np.ndarray) -> dict:
    """Encode a frame as JPEG and POST to /process-frame. Return parsed JSON."""
    ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
    if not ok:
        raise RuntimeError("Failed to JPEG-encode frame")
    jpeg_bytes = buf.tobytes()

    t0 = time.perf_counter()
    resp = requests.post(
        f"{server_url}/process-frame",
        files={"file": ("frame.jpg", jpeg_bytes, "image/jpeg")},
        timeout=30,
    )
    http_lat = (time.perf_counter() - t0) * 1000.0

    if resp.status_code != 200:
        raise RuntimeError(f"Server returned {resp.status_code}: {resp.text[:200]}")

    data = resp.json()
    data["_http_latency_ms"] = round(http_lat, 1)
    data["_raw_jpeg_bytes"]  = len(jpeg_bytes)
    return data


# ---------------------------------------------------------------------------
# Overlay renderer
# ---------------------------------------------------------------------------

def render_overlay(raw: np.ndarray, obfuscated: np.ndarray, meta: dict) -> np.ndarray:
    """Build side-by-side frame with a metadata overlay banner."""
    h, w = raw.shape[:2]
    # Resize obfuscated to match raw if needed
    if obfuscated.shape[:2] != (h, w):
        obfuscated = cv2.resize(obfuscated, (w, h))

    combined = np.hstack([raw, obfuscated])
    banner_h = 45
    overlay  = np.zeros((banner_h, combined.shape[1], 3), dtype=np.uint8)

    routing = meta.get("routing", {})
    bw      = meta.get("bandwidth", {})
    sw      = meta.get("semantic_weight", 0)
    node    = routing.get("node_id") or "cloud"
    cpu     = routing.get("node_cpu_load")
    dest    = routing.get("destination", "?")
    gk_lat  = meta.get("latency_ms", 0)
    bw_pct  = bw.get("bandwidth_reduction_pct", 0)

    cpu_str = f"{cpu:.0f}%" if cpu is not None else "N/A"
    line1 = f"Node: {node}  CPU: {cpu_str}  Dest: {dest.upper()}  SW: {sw}"
    line2 = f"GK: {gk_lat:.0f}ms  BW saved: {bw_pct:.1f}%"

    cv2.putText(overlay, line1, (10, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
    cv2.putText(overlay, line2, (10, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    cv2.putText(overlay, "RAW INPUT", (w // 2 - 50, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    cv2.putText(overlay, "OBFUSCATED", (w + w // 2 - 60, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

    return np.vstack([overlay, combined])


# ---------------------------------------------------------------------------
# Generate synthetic test video if none exists
# ---------------------------------------------------------------------------

def generate_test_video(out_path: str, num_frames: int = 90, fps: int = 15,
                        width: int = 640, height: int = 480):
    """Create a short synthetic video with coloured rectangles and text."""
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_path, fourcc, fps, (width, height))

    for i in range(num_frames):
        frame = np.full((height, width, 3), 180, dtype=np.uint8)
        # Moving rectangle
        x = int((i * 5) % (width - 100))
        y = int(100 + 50 * np.sin(i * 0.1))
        cv2.rectangle(frame, (x, y), (x + 90, y + 130), (0, 180, 0), -1)
        cv2.rectangle(frame, (x + 120, y + 20), (x + 200, y + 80), (200, 0, 0), -1)
        cv2.putText(frame, f"Frame {i+1:03d}/{num_frames}", (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        writer.write(frame)

    writer.release()
    print(f"[GEN] Synthetic test video -> {os.path.abspath(out_path)}  ({num_frames} frames, {fps} fps)")


# ---------------------------------------------------------------------------
# Main stream loop
# ---------------------------------------------------------------------------

def process_stream(
    source,
    server_url: str,
    output_path: str,
    display: bool = False,
    max_frames: int = MAX_FRAMES,
):
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video source: {source}")

    src_fps = cap.get(cv2.CAP_PROP_FPS) or 15.0
    src_w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total   = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    total   = min(total, max_frames) if total > 0 else max_frames

    print(f"Source: {source}  ({src_w}x{src_h} @ {src_fps:.1f} fps, ~{total} frames)")

    # Output video writer (double-wide for side-by-side + banner)
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    out_w = src_w * 2
    out_h = src_h + 45   # banner height
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, min(src_fps, 15), (out_w, out_h))

    metrics = StreamMetrics()
    frame_idx = 0

    print(f"Streaming to {server_url}/process-frame ...")
    print(f"Output video -> {os.path.abspath(output_path)}")
    if display:
        print("Press 'q' in the display window to stop early.\n")
    else:
        print("")

    while frame_idx < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1

        try:
            meta = post_frame(server_url, frame)
        except Exception as e:
            print(f"  [WARN] Frame {frame_idx}: {e}")
            continue

        bw      = meta.get("bandwidth", {})
        routing = meta.get("routing", {})
        metrics.record(
            latency_ms=meta.get("_http_latency_ms", 0),
            bw_saved=bw.get("bandwidth_saved_bytes", 0),
            raw_bytes=bw.get("raw_frame_bytes", 0),
            obf_bytes=bw.get("obfuscated_bytes", 0),
            node_id=routing.get("node_id", ""),
            destination=routing.get("destination", "edge"),
        )

        # Build display frame (use raw frame + draw detections as "obfuscated" proxy)
        obf_frame = frame.copy()
        for det in meta.get("detections", []):
            x1, y1, x2, y2 = det["bbox"]
            # Apply Gaussian blur to simulate what the server produced
            roi = obf_frame[y1:y2, x1:x2]
            if roi.size > 0:
                obf_frame[y1:y2, x1:x2] = cv2.GaussianBlur(roi, (51, 51), 30)

        vis = render_overlay(frame, obf_frame, meta)
        writer.write(vis)

        # Progress
        if frame_idx % 10 == 0 or frame_idx == 1:
            print(
                f"  [{frame_idx:4d}/{total}]  "
                f"fps={metrics.rolling_fps:5.1f}  "
                f"avg_lat={metrics.avg_latency:6.0f}ms  "
                f"bw_saved={metrics.total_bw_saved/1024:7.0f}KB  "
                f"node={routing.get('node_id','?')}"
            )

        if display:
            cv2.imshow("Edge Video Analytics - Live Stream", vis)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                print("\n[STOP] User pressed 'q'.")
                break

    cap.release()
    writer.release()
    if display:
        cv2.destroyAllWindows()

    # Final summary
    summary = metrics.summary()
    print("\n" + "=" * 65)
    print("  STREAM PROCESSING SUMMARY")
    print("=" * 65)
    for k, v in summary.items():
        print(f"    {k:<25}: {v}")
    print("=" * 65)
    print(f"\n[DONE] Output video -> {os.path.abspath(output_path)}")

    return summary


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Edge Video Analytics - Stream Processor")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--video",  type=str, help="Path to input video file")
    group.add_argument("--camera", type=int, help="Camera device index (e.g. 0)")
    parser.add_argument("--server", type=str, default=DEFAULT_SERVER, help="FastAPI server URL")
    parser.add_argument("--output", type=str, default=OUTPUT_VIDEO, help="Output video path")
    parser.add_argument("--display", action="store_true", help="Show live OpenCV window")
    parser.add_argument("--max-frames", type=int, default=MAX_FRAMES, help="Max frames to process")
    args = parser.parse_args()

    # Default: generate synthetic test video if no source specified
    source = args.camera if args.camera is not None else args.video
    if source is None:
        synth_path = "sample_media/test_video.mp4"
        if not Path(synth_path).exists():
            generate_test_video(synth_path, num_frames=60, fps=15)
        source = synth_path

    process_stream(
        source=source,
        server_url=args.server,
        output_path=args.output,
        display=args.display,
        max_frames=args.max_frames,
    )


if __name__ == "__main__":
    main()
