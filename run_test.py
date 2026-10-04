"""
Standalone Verification Script for Privacy Gatekeeper.
Downloads a sample test image (or generates synthetic scene), runs it through PrivacyGatekeeper,
saves side-by-side visual comparison to visual_test_output.jpg, and prints metadata.
"""
import os
import sys
import urllib.request
import cv2
import numpy as np

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from src.gatekeeper import PrivacyGatekeeper


def fetch_or_create_test_image(destination_path: str = "sample_media/test_traffic_pedestrians.jpg") -> np.ndarray:
    """Download a realistic test image with people/cars or generate a synthetic one if offline."""
    os.makedirs(os.path.dirname(destination_path), exist_ok=True)

    # Standard public test image from Ultralytics (bus.jpg contains person and bus)
    url = "https://ultralytics.com/images/bus.jpg"
    
    if not os.path.exists(destination_path):
        print(f"Downloading sample image from {url} ...")
        try:
            headers = {"User-Agent": "Mozilla/5.0"}
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=10) as response, open(destination_path, "wb") as out_file:
                out_file.write(response.read())
            print(f"Saved test image to {destination_path}")
        except Exception as e:
            print(f"Warning: Could not download image ({e}). Generating synthetic test scene...")
            # Create synthetic test image
            h, w = 640, 640
            img = np.full((h, w, 3), 200, dtype=np.uint8)
            # Add synthetic road & buildings
            cv2.rectangle(img, (0, 400), (w, h), (70, 70, 70), -1)
            cv2.putText(img, "Synthetic Test Scene", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 200), 2)
            cv2.imwrite(destination_path, img)

    image = cv2.imread(destination_path)
    if image is None:
        raise RuntimeError(f"Failed to read test image from {destination_path}")
    return image


def create_side_by_side_comparison(original: np.ndarray, obfuscated: np.ndarray, detections: list) -> np.ndarray:
    """Create a side-by-side visualization with headers and detection boxes on the original."""
    annotated_orig = original.copy()

    # Draw bounding boxes and class names on original image for comparison
    for det in detections:
        x1, y1, x2, y2 = det.bbox
        cv2.rectangle(annotated_orig, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label = f"{det.class_name}: {det.confidence:.2f}"
        cv2.putText(
            annotated_orig,
            label,
            (x1, max(y1 - 10, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
        )

    # Resize both to same height if different
    h1, w1 = annotated_orig.shape[:2]
    h2, w2 = obfuscated.shape[:2]
    target_h = max(h1, h2)
    
    if h1 != target_h:
        annotated_orig = cv2.resize(annotated_orig, (int(w1 * (target_h / h1)), target_h))
    if h2 != target_h:
        obfuscated = cv2.resize(obfuscated, (int(w2 * (target_h / h2)), target_h))

    # Add header banner
    header_height = 50
    header_w = annotated_orig.shape[1] + obfuscated.shape[1]
    header = np.zeros((header_height, header_w, 3), dtype=np.uint8)
    
    # Text on header
    cv2.putText(header, "ORIGINAL (Detected Entities)", (40, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
    cv2.putText(header, "OBFUSCATED (Privacy Gatekeeper)", (annotated_orig.shape[1] + 40, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    combined_body = np.hstack([annotated_orig, obfuscated])
    combined_full = np.vstack([header, combined_body])
    return combined_full


def main():
    print("=" * 60)
    print("Initializing PrivacyGatekeeper ...")
    gatekeeper = PrivacyGatekeeper(
        model_path="yolov8n.pt",
        conf_threshold=0.35,
        sensitive_classes=["person", "car", "bus", "truck"],
        blur_kernel_size=51,
        blur_sigma=35.0,
    )
    print(f"Target Sensitive Classes: {gatekeeper.sensitive_classes}")

    # Load test image
    test_img = fetch_or_create_test_image()
    print(f"Loaded test frame of shape: {test_img.shape}")

    # Run gatekeeper
    print("Running Privacy Gatekeeper inference and obfuscation...")
    result = gatekeeper.process_frame(test_img, obfuscate=True, obfuscation_mode="blur")

    # Output metadata
    print("\n" + "=" * 60)
    print("PIPELINE EXECUTION METADATA:")
    print(f" - Latency: {result.latency_ms:.2f} ms")
    print(f" - Semantic Weight (Sensitive Objects Count): {result.semantic_weight}")
    print(f" - Class Breakdown: {result.class_breakdown}")
    print(" - Detections List:")
    for idx, det in enumerate(result.detections, start=1):
        print(f"    [{idx}] {det.class_name.upper()} (conf: {det.confidence:.2f}) -> BBox: {det.bbox}")
    print("=" * 60)

    # Save visual test output
    output_filename = "visual_test_output.jpg"
    comparison = create_side_by_side_comparison(test_img, result.obfuscated_frame, result.detections)
    cv2.imwrite(output_filename, comparison)
    print(f"\n[SUCCESS] Visual comparison saved to: {os.path.abspath(output_filename)}")


if __name__ == "__main__":
    main()
