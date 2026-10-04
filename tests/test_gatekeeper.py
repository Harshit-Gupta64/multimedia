import os
import sys
import pytest
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.gatekeeper import PrivacyGatekeeper, GatekeeperResult


def test_gatekeeper_initialization():
    gatekeeper = PrivacyGatekeeper(model_path="yolov8n.pt")
    assert gatekeeper is not None
    assert "person" in gatekeeper.sensitive_classes
    assert "car" in gatekeeper.sensitive_classes


def test_gatekeeper_process_synthetic_frame():
    gatekeeper = PrivacyGatekeeper(model_path="yolov8n.pt")
    frame = np.zeros((640, 640, 3), dtype=np.uint8)
    result = gatekeeper.process_frame(frame)
    
    assert isinstance(result, GatekeeperResult)
    assert result.obfuscated_frame.shape == frame.shape
    assert isinstance(result.latency_ms, float)
    assert isinstance(result.semantic_weight, int)
    assert result.semantic_weight >= 0
