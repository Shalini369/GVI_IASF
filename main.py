"""
Assisted Vision: Real-Time Spatial Hazard Detection & Universal Scene OCR
Optimized for Lenovo Snapdragon X Plus (ARM64 Windows) CPU inference.

Usage:
    python main.py
    python main.py --source 0 --conf 0.50 --ocr-interval 10
    python main.py --source path/to/video.mp4
"""

import sys
import os
import time
import argparse
import logging
from typing import Dict, Set, List
import cv2
import numpy as np

# Ensure project root is in sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from modules.object_detection import (
    HazardDetector,
    PROXIMITY_CLOSE,
    PROXIMITY_MEDIUM,
    PROXIMITY_FAR,
    CAT_TRANSIT,
    CAT_SIGNAGE,
    CAT_SCENE,
)

from modules.tts_engine import speak_text

# Set up logging format
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("AssistedVision")


class AssistVoiceManager:
    """
    Manages and throttles console audio/speech alert triggers for both
    imminent hazards and newly detected scene/transit text.
    """

    def __init__(self, hazard_cooldown: float = 1.2, text_cooldown: float = 3.5) -> None:
        self.hazard_cooldown = hazard_cooldown
        self.text_cooldown = text_cooldown
        self.last_hazard_times: Dict[str, float] = {}
        self.last_text_times: Dict[str, float] = {}

    def process_alerts(self, structured_labels: List[str], detections: List[dict]) -> None:
        """
        Parses structured messages and prints clean console alerts for speech synthesis.
        """
        now = time.time()

        for msg in structured_labels:
            is_close_hazard = ("detected CLOSE" in msg) or ("detected close" in msg)
            is_text_event = ("Scene Text Detected:" in msg) or ("Read:" in msg) or ("Route:" in msg)

            if is_close_hazard:
                last_time = self.last_hazard_times.get(msg, 0.0)

                if now - last_time >= self.hazard_cooldown:
                    print(f"\n[!!! PROXIMITY HAZARD ALERT !!!] >>> {msg} <<<")

                    speak_text(msg, priority=1)

                    self.last_hazard_times[msg] = now

            elif is_text_event:
                last_time = self.last_text_times.get(msg, 0.0)

                if now - last_time >= self.text_cooldown:
                    print(f"[OCR VOICE ASSIST] >>> {msg} <<<")

                    speak_text(msg, priority=2)

                    self.last_text_times[msg] = now


def draw_telemetry_hud(
    frame: np.ndarray,
    fps: float,
    latency_ms: float,
    conf_threshold: float,
    close_hazards: int,
    total_hazards: int,
    ocr_count: int,
) -> None:
    """
    Renders telemetry HUD (FPS, Latency, Hazards, and Active OCR detections)
    in a semi-transparent HUD overlay.
    """
    h, w = frame.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX

    panel_w = 340
    panel_h = 95
    px1, py1 = 12, 12
    px2, py2 = px1 + panel_w, py1 + panel_h

    # Backdrop
    hud_bg = frame.copy()
    cv2.rectangle(hud_bg, (px1, py1), (px2, py2), (15, 15, 15), -1)
    border_color = (0, 0, 255) if close_hazards > 0 else (0, 220, 255)
    cv2.rectangle(hud_bg, (px1, py1), (px2, py2), border_color, 2)
    cv2.addWeighted(hud_bg, 0.75, frame, 0.25, 0, frame)

    # Line 1: FPS and Latency
    line1 = f"FPS: {fps:4.1f} | Latency: {latency_ms:3.0f}ms (Snapdragon X Plus)"
    cv2.putText(frame, line1, (px1 + 10, py1 + 22), font, 0.44, (255, 255, 255), 1, cv2.LINE_AA)

    # Line 2: Active Confidence & OCR Engine
    line2 = f"YOLO Conf: {conf_threshold:.2f} | RapidOCR Engine: ACTIVE"
    cv2.putText(frame, line2, (px1 + 10, py1 + 44), font, 0.42, (0, 220, 255), 1, cv2.LINE_AA)

    # Line 3: Hazards Summary
    h_col = (0, 0, 255) if close_hazards > 0 else (0, 255, 120)
    line3 = f"CLOSE Hazards: {close_hazards} | Total Objects: {total_hazards}"
    cv2.putText(frame, line3, (px1 + 10, py1 + 66), font, 0.44, h_col, 1, cv2.LINE_AA)

    # Line 4: OCR Text Count
    line4 = f"Visible Text Elements: {ocr_count} (Cyan/Magenta)"
    cv2.putText(frame, line4, (px1 + 10, py1 + 86), font, 0.40, (255, 255, 0), 1, cv2.LINE_AA)


def run_assisted_vision(
    source: str = "0",
    model_path: str = "models/yolov8n.pt",
    conf_threshold: float = 0.50,
    ocr_conf: float = 0.45,
    ocr_interval: int = 10,
    resolution: tuple = (640, 480),
) -> None:
    """
    Main real-time detection & OCR loop.
    """
    logger.info("Initializing HazardDetector with Universal OCR...")
    detector = HazardDetector(
        model_path=model_path,
        conf_threshold=conf_threshold,
        ocr_conf_threshold=ocr_conf,
        ocr_interval_frames=ocr_interval,
    )
    voice_manager = AssistVoiceManager(hazard_cooldown=1.2, text_cooldown=3.5)

    video_source = int(source) if source.isdigit() else source
    logger.info("Opening video source: %s", video_source)

    cap = cv2.VideoCapture(video_source)
    if isinstance(video_source, int):
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, resolution[0])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, resolution[1])
        cap.set(cv2.CAP_PROP_FPS, 30)

    if not cap.isOpened():
        logger.error("Could not open video source '%s'. Check camera device or file permissions.", source)
        return

    window_title = "Assisted Vision: Spatial Hazards & Universal Scene OCR (ARM64)"
    cv2.namedWindow(window_title, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_title, resolution[0], resolution[1])

    logger.info("Real-time stream started. Press 'q' to quit.")
    print("=" * 80)
    print("  ASSISTED VISION: HAZARD DETECTION & UNIVERSAL SCENE OCR (RapidOCR)")
    print(f"  Confidence Threshold: {conf_threshold:.2f} (Adjust live with '+' or '-')")
    print(f"  Full-Scene OCR Scan : Every {ocr_interval} frames | Immediate ROI for Bus/Signage")
    print("  Hotkeys             : '+' increase conf | '-' decrease conf | 'c' clear OCR cache | 'q' quit")
    print("=" * 80)

    prev_time = time.perf_counter()
    smoothed_fps = 0.0
    alpha = 0.15
    current_conf = conf_threshold

    try:
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                if not isinstance(video_source, int):
                    logger.info("End of video stream reached.")
                    break
                logger.warning("Empty frame from camera, waiting...")
                time.sleep(0.05)
                continue

            frame_start = time.perf_counter()

            # Execute combined YOLO + RapidOCR pipeline
            structured_labels, detections, annotated_frame = detector.detect_objects(
                frame, conf_threshold=current_conf
            )

            frame_end = time.perf_counter()
            latency_ms = (frame_end - frame_start) * 1000.0

            # Compute smoothed FPS
            instant_fps = 1.0 / max(1e-5, (frame_end - prev_time))
            prev_time = frame_end
            smoothed_fps = instant_fps if smoothed_fps == 0.0 else (alpha * instant_fps + (1.0 - alpha) * smoothed_fps)

            # Separate hazard vs OCR counts
            hazard_items = [d for d in detections if d.get("type") == "hazard"]
            ocr_items = [d for d in detections if d.get("type") == "ocr"]
            close_count = sum(1 for h in hazard_items if h.get("proximity") == PROXIMITY_CLOSE)

            # Process audio/console alerts
            voice_manager.process_alerts(structured_labels, detections)

            # Draw HUD
            draw_telemetry_hud(
                annotated_frame,
                fps=smoothed_fps,
                latency_ms=latency_ms,
                conf_threshold=current_conf,
                close_hazards=close_count,
                total_hazards=len(hazard_items),
                ocr_count=len(ocr_items),
            )

            # Show annotated stream
            cv2.imshow(window_title, annotated_frame)

            # Handle interactive keys
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), ord("Q"), 27):
                logger.info("User requested exit.")
                break
            elif key in (ord("+"), ord("=")):
                current_conf = min(0.95, round(current_conf + 0.05, 2))
                logger.info("Confidence threshold adjusted UP to: %.2f", current_conf)
            elif key in (ord("-"), ord("_")):
                current_conf = max(0.10, round(current_conf - 0.05, 2))
                logger.info("Confidence threshold adjusted DOWN to: %.2f", current_conf)
            elif key in (ord("c"), ord("C")):
                detector.cached_scene_text.clear()
                logger.info("OCR scene cache cleared.")

    except KeyboardInterrupt:
        logger.info("Interrupted by user (Ctrl+C).")
    except Exception as e:
        logger.error("Runtime exception in detection loop: %s", e, exc_info=True)
    finally:
        logger.info("Releasing resources and closing windows...")
        cap.release()
        cv2.destroyAllWindows()
        logger.info("Shutdown complete.")


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Assisted Vision Real-Time Spatial Hazards & Universal Scene OCR."
    )
    parser.add_argument(
        "--source",
        type=str,
        default="0",
        help="Video capture source: camera index (default: 0) or path to video file.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="models/yolov8n.pt",
        help="Path to YOLO weights (default: models/yolov8n.pt).",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.50,
        help="Confidence threshold for YOLO hazard detection (default: 0.50).",
    )
    parser.add_argument(
        "--ocr-conf",
        type=float,
        default=0.45,
        help="Minimum confidence threshold for RapidOCR text recognition (default: 0.45).",
    )
    parser.add_argument(
        "--ocr-interval",
        type=int,
        default=10,
        help="Frame interval for periodic full-frame OCR scan (default: 10).",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=640,
        help="Capture frame width (default: 640).",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=480,
        help="Capture frame height (default: 480).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    run_assisted_vision(
        source=args.source,
        model_path=args.model,
        conf_threshold=args.conf,
        ocr_conf=args.ocr_conf,
        ocr_interval=args.ocr_interval,
        resolution=(args.width, args.height),
    )
