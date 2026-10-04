"""
tests/test_server.py  –  Unit tests for server v0.4 helpers and schemas
"""
import os
import sys
import pytest
import numpy as np
import cv2

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.server import _compute_bandwidth, BandwidthMetrics
from src.scheduler import DynamicScheduler, RoutingDecision


class TestBandwidthMetricsHelper:

    def _raw_and_obf(self):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        ok, raw_buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
        blurred     = cv2.GaussianBlur(frame, (51, 51), 30)
        ok2, obf_buf= cv2.imencode(".jpg", blurred, [cv2.IMWRITE_JPEG_QUALITY, 85])
        return raw_buf.tobytes(), obf_buf.tobytes()

    def test_returns_bandwidth_metrics_model(self):
        raw, obf = self._raw_and_obf()
        result = _compute_bandwidth(raw, obf)
        assert isinstance(result, BandwidthMetrics)

    def test_raw_size_matches_input(self):
        raw, obf = self._raw_and_obf()
        result = _compute_bandwidth(raw, obf)
        assert result.raw_frame_bytes == len(raw)

    def test_obfuscated_size_matches_input(self):
        raw, obf = self._raw_and_obf()
        result = _compute_bandwidth(raw, obf)
        assert result.obfuscated_bytes == len(obf)

    def test_bandwidth_saved_non_negative(self):
        raw, obf = self._raw_and_obf()
        result = _compute_bandwidth(raw, obf)
        assert result.bandwidth_saved_bytes >= 0

    def test_reduction_pct_in_valid_range(self):
        raw, obf = self._raw_and_obf()
        result = _compute_bandwidth(raw, obf)
        assert 0.0 <= result.bandwidth_reduction_pct <= 100.0

    def test_empty_raw_no_division_error(self):
        result = _compute_bandwidth(b"", b"data")
        assert result.bandwidth_reduction_pct == 0.0
        assert result.bandwidth_saved_bytes == 0


class TestSchedulerEndpointRegistry:

    def test_node_endpoint_in_status(self):
        s = DynamicScheduler(num_nodes=3, seed=0)
        status = s.get_node_status()
        for entry in status:
            assert "endpoint" in entry
            assert entry["endpoint"].startswith("http")

    def test_get_node_endpoint_returns_url(self):
        s = DynamicScheduler(num_nodes=3, seed=0)
        ep = s.get_node_endpoint("edge-node-01")
        assert ep is not None
        assert "8001" in ep or "edge-node" in ep

    def test_get_node_endpoint_unknown_returns_none(self):
        s = DynamicScheduler(num_nodes=3, seed=0)
        assert s.get_node_endpoint("nonexistent-node") is None

    def test_routing_decision_has_endpoint(self):
        s = DynamicScheduler(num_nodes=3, seed=42)
        for n in s.nodes:
            n.cpu_load = 20.0
        s._refresh_cpu_loads = lambda: None
        decision = s.route_frame(semantic_weight=2)
        if decision.destination == "edge":
            assert decision.node_endpoint is not None
            assert decision.node_endpoint.startswith("http")

    def test_cloud_routing_has_no_endpoint(self):
        s = DynamicScheduler(num_nodes=3, seed=0)
        for n in s.nodes:
            n.cpu_load = 99.0
        s._refresh_cpu_loads = lambda: None
        decision = s.route_frame(semantic_weight=8)
        if decision.destination == "cloud":
            assert decision.node_endpoint is None
