import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union
import cv2
import numpy as np
from ultralytics import YOLO


@dataclass
class Detection:
    class_id: int
    class_name: str
    confidence: float
    bbox: Tuple[int, int, int, int]  # (x1, y1, x2, y2)


@dataclass
class GatekeeperResult:
    obfuscated_frame: np.ndarray
    semantic_weight: int
    latency_ms: float
    detections: List[Detection] = field(default_factory=list)
    class_breakdown: Dict[str, int] = field(default_factory=dict)


class PrivacyGatekeeper:
    """
    Privacy Gatekeeper performs real-time edge detection of sensitive entities
    (e.g., person, vehicles) and applies heavy Gaussian blur/pixelation
    over detected bounding boxes while extracting semantic metrics.
    """

    DEFAULT_SENSITIVE_CLASSES = ["person", "car", "bus", "truck"]

    def __init__(
        self,
        model_path: str = "yolov8n.pt",
        conf_threshold: float = 0.35,
        sensitive_classes: Optional[List[str]] = None,
        blur_kernel_size: int = 51,
        blur_sigma: float = 30.0,
        device: Optional[str] = None,
    ):
        """
        Initialize the PrivacyGatekeeper with a YOLO model.

        :param model_path: Path or identifier of the YOLOv8 model (default: yolov8n.pt).
        :param conf_threshold: Minimum detection confidence score.
        :param sensitive_classes: List of class names to consider sensitive and obfuscate.
        :param blur_kernel_size: Gaussian blur kernel size (odd integer).
        :param blur_sigma: Gaussian blur standard deviation in X and Y directions.
        :param device: Device to run inference on ('cpu', 'cuda', 0, etc.).
        """
        self.conf_threshold = conf_threshold
        self.sensitive_classes = sensitive_classes or self.DEFAULT_SENSITIVE_CLASSES
        self.blur_kernel_size = blur_kernel_size if blur_kernel_size % 2 == 1 else blur_kernel_size + 1
        self.blur_sigma = blur_sigma
        self.device = device

        # Load YOLO model
        self.model = YOLO(model_path)
        self._target_class_ids = self._resolve_target_class_ids()

    def _resolve_target_class_ids(self) -> List[int]:
        """Map sensitive class names to model class indices."""
        target_ids = []
        for class_id, class_name in self.model.names.items():
            if class_name.lower() in [cls.lower() for cls in self.sensitive_classes]:
                target_ids.append(class_id)
        return target_ids

    def obfuscate_region(
        self,
        frame: np.ndarray,
        bbox: Tuple[int, int, int, int],
        mode: str = "blur",
    ) -> np.ndarray:
        """
        Apply heavy blur or pixelation to a specific bounding box within a frame.

        :param frame: Image/frame as a numpy array.
        :param bbox: (x1, y1, x2, y2) coordinates.
        :param mode: Obfuscation technique ('blur' or 'pixelate').
        :return: Modified frame slice or updated frame.
        """
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = bbox

        # Clamp bounding box coordinates to image boundaries
        x1 = max(0, min(x1, w - 1))
        y1 = max(0, min(y1, h - 1))
        x2 = max(x1 + 1, min(x2, w))
        y2 = max(y1 + 1, min(y2, h))

        roi = frame[y1:y2, x1:x2]
        if roi.size == 0:
            return frame

        if mode == "pixelate":
            # Downscale and upscale to create pixelation effect
            roi_h, roi_w = roi.shape[:2]
            scale_w = max(1, roi_w // 16)
            scale_h = max(1, roi_h // 16)
            temp = cv2.resize(roi, (scale_w, scale_h), interpolation=cv2.INTER_LINEAR)
            obfuscated_roi = cv2.resize(temp, (roi_w, roi_h), interpolation=cv2.INTER_NEAREST)
        else:
            # Gaussian blur
            ksize = self.blur_kernel_size
            obfuscated_roi = cv2.GaussianBlur(roi, (ksize, ksize), self.blur_sigma)

        frame[y1:y2, x1:x2] = obfuscated_roi
        return frame

    def process_frame(
        self,
        frame: np.ndarray,
        obfuscate: bool = True,
        obfuscation_mode: str = "blur",
    ) -> GatekeeperResult:
        """
        Detect sensitive entities in the frame, obfuscate their regions,
        and calculate semantic metrics and latency.

        :param frame: Raw OpenCV image (numpy array, BGR format).
        :param obfuscate: Whether to apply obfuscation to the returned frame.
        :param obfuscation_mode: 'blur' or 'pixelate'.
        :return: GatekeeperResult containing obfuscated frame, semantic weight, and metadata.
        """
        start_time = time.perf_counter()

        # Copy frame to prevent mutating input array unexpectedly
        output_frame = frame.copy()

        # Run inference
        results = self.model(
            output_frame,
            conf=self.conf_threshold,
            classes=self._target_class_ids,
            verbose=False,
            device=self.device,
        )

        detections: List[Detection] = []
        class_breakdown: Dict[str, int] = {}

        if results and len(results) > 0:
            boxes = results[0].boxes
            for box in boxes:
                cls_id = int(box.cls[0].item())
                cls_name = self.model.names.get(cls_id, str(cls_id))
                conf = float(box.conf[0].item())
                xyxy = [int(v) for v in box.xyxy[0].tolist()]
                bbox = (xyxy[0], xyxy[1], xyxy[2], xyxy[3])

                detections.append(
                    Detection(
                        class_id=cls_id,
                        class_name=cls_name,
                        confidence=conf,
                        bbox=bbox,
                    )
                )

                class_breakdown[cls_name] = class_breakdown.get(cls_name, 0) + 1

                if obfuscate:
                    self.obfuscate_region(output_frame, bbox, mode=obfuscation_mode)

        latency_ms = (time.perf_counter() - start_time) * 1000.0
        semantic_weight = len(detections)

        return GatekeeperResult(
            obfuscated_frame=output_frame,
            semantic_weight=semantic_weight,
            latency_ms=round(latency_ms, 2),
            detections=detections,
            class_breakdown=class_breakdown,
        )
