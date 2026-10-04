"""
tests/test_benchmark.py - Unit tests for benchmark metric calculation logic
"""
import os
import sys
import pytest
import numpy as np
import cv2

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.benchmark import (
    compute_bandwidth,
    encode_jpeg,
    make_synthetic_frame,
    load_real_frame,
    _mean,
    _percentile,
)


class TestBandwidthCalculation:

    def test_compute_bandwidth_reduces_correctly(self):
        """Obfuscated JPEG should be smaller → positive savings."""
        frame = make_synthetic_frame(640, 480)
        raw_bytes = encode_jpeg(frame)
        # Blurred version (lower entropy → smaller JPEG)
        blurred = cv2.GaussianBlur(frame, (51, 51), 30)
        obf_bytes = encode_jpeg(blurred)
        result = compute_bandwidth(raw_bytes, obf_bytes)
        assert result["raw_bytes"] == len(raw_bytes)
        assert result["obf_bytes"] == len(obf_bytes)
        assert result["bandwidth_saved_bytes"] >= 0
        assert 0.0 <= result["bandwidth_reduction_pct"] <= 100.0

    def test_compute_bandwidth_zero_raw_size(self):
        """Edge case: empty raw payload should not divide by zero."""
        result = compute_bandwidth(b"", b"data")
        assert result["bandwidth_reduction_pct"] == 0.0

    def test_compute_bandwidth_identical_frames(self):
        """Same frame before and after → 0 bytes saved."""
        frame = make_synthetic_frame(320, 240)
        raw = encode_jpeg(frame)
        result = compute_bandwidth(raw, raw)
        assert result["bandwidth_saved_bytes"] == 0
        assert result["bandwidth_reduction_pct"] == 0.0

    def test_compute_bandwidth_obfuscated_larger_not_negative(self):
        """If obfuscated frame is somehow larger, saved must be clamped to 0."""
        result = compute_bandwidth(b"small", b"much_larger_payload")
        assert result["bandwidth_saved_bytes"] == 0


class TestFrameGeneration:

    def test_synthetic_frame_shape(self):
        frame = make_synthetic_frame(640, 480, num_rects=3)
        assert frame.shape == (480, 640, 3)
        assert frame.dtype == np.uint8

    def test_synthetic_frame_no_rects(self):
        frame = make_synthetic_frame(320, 240, num_rects=0)
        # Should be a uniform grey frame
        assert frame.shape == (240, 320, 3)

    def test_encode_jpeg_returns_bytes(self):
        frame = make_synthetic_frame(320, 240)
        result = encode_jpeg(frame)
        assert isinstance(result, bytes)
        assert len(result) > 0


class TestStatisticsHelpers:

    def test_mean_basic(self):
        assert _mean([10.0, 20.0, 30.0]) == 20.0

    def test_mean_empty(self):
        assert _mean([]) == 0.0

    def test_mean_single(self):
        assert _mean([42.5]) == 42.5

    def test_percentile_p95(self):
        values = list(range(1, 101))  # 1..100
        p95 = _percentile([float(v) for v in values], 95)
        assert p95 == 95.0

    def test_percentile_empty(self):
        assert _percentile([], 95) == 0.0

    def test_percentile_p50_median(self):
        # _percentile uses floor-index: idx = int(len * p/100) - 1
        # For 5 elements at p50: idx = int(5 * 0.5) - 1 = 1  → value = 20.0
        values = [10.0, 20.0, 30.0, 40.0, 50.0]
        assert _percentile(values, 50) == 20.0
