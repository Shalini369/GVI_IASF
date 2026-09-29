"""
Assisted Vision: Real-Time Spatial Hazard Detection & Universal Scene OCR
Optimized for Lenovo Snapdragon X Plus (ARM64 Windows) CPU inference.

Features:
- Ultralytics YOLOv8 object & hazard detection with whitelisting and spatial rules
- Universal Scene Text Reading (RapidOCR) with dual strategy:
  1. Periodic full-frame downscaled scan (every 10 frames)
  2. Immediate full-resolution ROI crop OCR for transit (bus/truck) and signage (stop sign, door, traffic light)
- Perspective Box Geometry proximity (CLOSE, MEDIUM, FAR) & spatial zones (OVERHEAD / UPPER BODY vs GROUND PATH)
- 10-frame persistence cache for full-scene text overlays for smooth 20+ FPS performance
"""

from typing import List, Dict, Any, Tuple, Optional, Set
import os
import sys
import re
import time
import logging
import cv2
import numpy as np
import torch
from ultralytics import YOLO
from rapidocr_onnxruntime import RapidOCR

# Configure module logger
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("HazardDetector")

# ---------------------------------------------------------------------------
# Hardware & CPU Optimization (Snapdragon X Plus / ARM64 Windows)
# ---------------------------------------------------------------------------
def optimize_cpu_runtime() -> int:
    """Configures PyTorch thread settings for Qualcomm Snapdragon X Plus ARM64."""
    total_cores = os.cpu_count() or 4
    optimal_threads = max(2, min(8, total_cores - 2 if total_cores > 4 else total_cores))
    torch.set_num_threads(optimal_threads)
    torch.set_grad_enabled(False)
    return optimal_threads


_ACTIVE_THREADS = optimize_cpu_runtime()

# ---------------------------------------------------------------------------
# Class Whitelists & Blacklists
# ---------------------------------------------------------------------------
# Whitelisted COCO IDs:
# 0: person, 1: bicycle, 2: car, 3: motorcycle, 5: bus, 7: truck,
# 9: traffic light, 10: fire hydrant, 11: stop sign, 13: bench,
# 56: chair, 58: potted plant
WHITELIST_COCO_IDS: List[int] = [0, 1, 2, 3, 5, 7, 9, 10, 11, 13, 56, 58]

WHITELIST_CLASS_NAMES: Set[str] = {
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "bus",
    "truck",
    "traffic light",
    "fire hydrant",
    "stop sign",
    "bench",
    "chair",
    "potted plant",
    # Custom / extended classes
    "door",
    "pole",
    "branch",
    "tree branch",
    "stairs",
    "step",
    "curb",
}

BLACKLIST_CLASS_NAMES: Set[str] = {
    "tv",
    "cell phone",
    "laptop",
    "book",
    "remote",
    "mouse",
    "keyboard",
    "microwave",
    "refrigerator",
    "toaster",
    "sink",
    "oven",
}

CLASS_LABEL_MAPPING: Dict[str, str] = {
    "person": "Pedestrian",
    "bicycle": "Bicycle",
    "car": "Car",
    "motorcycle": "Motorcycle",
    "bus": "Bus",
    "truck": "Truck",
    "traffic light": "Traffic Light",
    "fire hydrant": "Fire Hydrant",
    "stop sign": "Stop Sign",
    "bench": "Bench",
    "chair": "Chair",
    "potted plant": "Potted Plant",
    "door": "Door",
    "pole": "Pole",
    "branch": "Branch",
    "tree branch": "Branch",
    "stairs": "Stairs",
    "step": "Step",
    "curb": "Curb",
}

DYNAMIC_CLASSES: Set[str] = {
    "bicycle",
    "car",
    "motorcycle",
    "bus",
    "truck",
}

# Immediate ROI OCR target categories
TRANSIT_ROI_CLASSES: Set[str] = {"bus", "truck"}
SIGNAGE_ROI_CLASSES: Set[str] = {"stop sign", "traffic light", "door"}
ALL_ROI_CLASSES: Set[str] = TRANSIT_ROI_CLASSES | SIGNAGE_ROI_CLASSES

# Spatial Zones
ZONE_OVERHEAD: str = "OVERHEAD / UPPER BODY"
ZONE_GROUND: str = "GROUND PATH"

# Proximity Levels
PROXIMITY_CLOSE: str = "CLOSE"
PROXIMITY_MEDIUM: str = "MEDIUM"
PROXIMITY_FAR: str = "FAR"

# OCR Text Categories
CAT_TRANSIT: str = "BUS ROUTE / TRANSIT TEXT"
CAT_SIGNAGE: str = "SIGNAGE / NAVIGATION TEXT"
CAT_SCENE: str = "SCENE TEXT"

# Colors (BGR)
COLOR_CLOSE: Tuple[int, int, int] = (0, 0, 255)         # Red
COLOR_MEDIUM: Tuple[int, int, int] = (0, 215, 255)      # Amber / Yellow
COLOR_FAR: Tuple[int, int, int] = (0, 220, 0)           # Green
COLOR_OCR_CYAN: Tuple[int, int, int] = (255, 255, 0)    # Bright Cyan for Scene Text
COLOR_OCR_MAGENTA: Tuple[int, int, int] = (255, 0, 255) # Bright Magenta for Signage
COLOR_OCR_BLUE: Tuple[int, int, int] = (255, 180, 0)    # Deep Sky Cyan for Transit
COLOR_BOX_BG: Tuple[int, int, int] = (15, 15, 15)


class TextCleaner:
    """Utility for cleaning and validating detected OCR strings."""

    @staticmethod
    def clean(text: str) -> str:
        """Strips noise, normalizes whitespace, and filters junk."""
        if not text:
            return ""
        # Replace non-printable characters and collapse whitespace
        cleaned = re.sub(r"[\r\n\t]+", " ", text).strip()
        cleaned = re.sub(r"\s{2,}", " ", cleaned)
        return cleaned

    @classmethod
    def is_valid(cls, text: str, confidence: float, min_conf: float = 0.45, min_len: int = 2) -> bool:
        """
        Validates text detection:
        - confidence >= min_conf (0.45)
        - length >= min_len (ignore single isolated characters)
        - contains at least one alphanumeric character
        """
        if confidence < min_conf:
            return False
        cleaned = cls.clean(text)
        if len(cleaned) < min_len:
            return False
        if not re.search(r"[A-Za-z0-9]", cleaned):
            return False
        return True


class HazardDetector:
    """
    Unified Real-Time Spatial Hazard & Scene OCR Engine.
    Executes YOLOv8 object detection alongside dual-mode RapidOCR.
    """

    def __init__(
        self,
        model_path: str = "models/yolov8n.pt",
        conf_threshold: float = 0.50,
        ocr_conf_threshold: float = 0.45,
        ocr_interval_frames: int = 10,
        ocr_max_width: int = 800,
        device: str = "cpu",
        imgsz: int = 640,
    ) -> None:
        """
        Initializes YOLOv8 detector and RapidOCR engine.

        Args:
            model_path: Path to YOLO weights.
            conf_threshold: Default confidence threshold for YOLO objects.
            ocr_conf_threshold: Minimum confidence for OCR text.
            ocr_interval_frames: Frame frequency for full-frame OCR (default 10 frames).
            ocr_max_width: Maximum width for downscaled full-frame OCR scan.
            device: Compute device ('cpu').
            imgsz: Input resolution for YOLO.
        """
        self.model_path = model_path
        self.conf_threshold = conf_threshold
        self.ocr_conf_threshold = ocr_conf_threshold
        self.ocr_interval_frames = ocr_interval_frames
        self.ocr_max_width = ocr_max_width
        self.device = device
        self.imgsz = imgsz

        # Frame counter & cache for full-scene OCR
        self.frame_count: int = 0
        self.cached_scene_text: List[Dict[str, Any]] = []

        logger.info(
            "Initializing HazardDetector (model=%s, conf=%.2f, ocr_interval=%d frames, max_ocr_w=%d)",
            model_path,
            conf_threshold,
            ocr_interval_frames,
            ocr_max_width,
        )

        # 1. Load YOLOv8 Model
        try:
            self.model = YOLO(self.model_path)
        except Exception as e:
            logger.error("Failed loading from %s. Downloading 'yolov8n.pt': %s", self.model_path, e)
            self.model = YOLO("yolov8n.pt")

        # 2. Initialize RapidOCR Engine
        logger.info("Initializing RapidOCR ONNXRuntime engine...")
        try:
            self.ocr_engine = RapidOCR()
            logger.info("RapidOCR initialized successfully.")
        except Exception as e:
            logger.error("Failed initializing RapidOCR engine: %s", e)
            raise RuntimeError(f"RapidOCR engine initialization error: {e}")

        # Warm up YOLO model
        self._warmup()

    def _warmup(self) -> None:
        """Runs a synthetic warm-up pass for YOLO and OCR engines."""
        try:
            dummy = np.zeros((self.imgsz, self.imgsz, 3), dtype=np.uint8)
            with torch.inference_mode():
                _ = self.model.predict(
                    source=dummy,
                    device=self.device,
                    imgsz=self.imgsz,
                    conf=self.conf_threshold,
                    classes=WHITELIST_COCO_IDS,
                    verbose=False,
                )
            _ = self.ocr_engine(dummy[:200, :400])
            logger.info("Warmup complete. Detection & OCR pipelines ready.")
        except Exception as e:
            logger.warning("Warmup notice: %s", e)

    @staticmethod
    def calculate_hazard_zone(y1: float, y2: float, box_height_ratio: float, frame_height: int) -> str:
        """
        Determines the spatial hazard zone:
        - "OVERHEAD / UPPER BODY": if top edge y1 < frame_height * 0.35 AND box_height_ratio > 0.25
        - "GROUND PATH": if bottom edge y2 >= frame_height * 0.50
        - Fallback: based on vertical center.
        """
        if (y1 < frame_height * 0.35) and (box_height_ratio > 0.25):
            return ZONE_OVERHEAD
        elif y2 >= frame_height * 0.50:
            return ZONE_GROUND
        else:
            y_center = (y1 + y2) / 2.0
            return ZONE_OVERHEAD if y_center < (frame_height * 0.50) else ZONE_GROUND

    @staticmethod
    def calculate_proximity(box_height_ratio: float, bottom_y_ratio: float) -> str:
        """
        Computes proximity based on Perspective Box Geometry:
        - "CLOSE": box_height_ratio > 0.45 OR bottom_y_ratio > 0.85
        - "MEDIUM": box_height_ratio between 0.20 and 0.45 OR bottom_y_ratio between 0.60 and 0.85
        - "FAR": box_height_ratio < 0.20 AND bottom_y_ratio < 0.60
        """
        if box_height_ratio > 0.45 or bottom_y_ratio > 0.85:
            return PROXIMITY_CLOSE
        elif box_height_ratio >= 0.20 or bottom_y_ratio >= 0.60:
            return PROXIMITY_MEDIUM
        else:
            return PROXIMITY_FAR

    def _run_ocr_on_image(
        self, img: np.ndarray, offset_x: int = 0, offset_y: int = 0, scale_factor: float = 1.0
    ) -> List[Dict[str, Any]]:
        """
        Executes RapidOCR on an image array and maps bounding boxes back to global coordinates.
        """
        if img is None or img.size == 0 or img.shape[0] < 12 or img.shape[1] < 12:
            return []

        try:
            results, _ = self.ocr_engine(img)
        except Exception as e:
            logger.debug("RapidOCR inference exception: %s", e)
            return []

        if not results:
            return []

        extracted_text: List[Dict[str, Any]] = []

        for item in results:
            # item format: [box_points, text, confidence]
            # box_points is 4 corners [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]
            if len(item) < 3:
                continue

            box_pts, raw_text, conf = item[0], item[1], float(item[2])
            cleaned = TextCleaner.clean(raw_text)

            if not TextCleaner.is_valid(cleaned, conf, min_conf=self.ocr_conf_threshold):
                continue

            pts = np.array(box_pts, dtype=np.float32)
            # Rescale back if image was downscaled
            if scale_factor != 1.0:
                pts /= scale_factor

            # Offset by ROI coordinate if cropped
            pts[:, 0] += offset_x
            pts[:, 1] += offset_y

            # Derive bounding rect [x1, y1, x2, y2]
            x1 = int(np.min(pts[:, 0]))
            y1 = int(np.min(pts[:, 1]))
            x2 = int(np.max(pts[:, 0]))
            y2 = int(np.max(pts[:, 1]))

            extracted_text.append({
                "text": cleaned,
                "confidence": round(conf, 3),
                "bbox": [x1, y1, x2, y2],
                "polygon": pts.tolist(),
            })

        return extracted_text

    def _scan_full_frame_ocr(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """
        Runs full-frame OCR scan on a downscaled representation for high efficiency.
        """
        h, w = frame.shape[:2]
        if w > self.ocr_max_width:
            scale = self.ocr_max_width / float(w)
            new_w = self.ocr_max_width
            new_h = int(h * scale)
            downscaled = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            return self._run_ocr_on_image(downscaled, scale_factor=scale)
        else:
            return self._run_ocr_on_image(frame, scale_factor=1.0)

    @torch.inference_mode()
    def detect_objects(
        self, frame: np.ndarray, conf_threshold: Optional[float] = None
    ) -> Tuple[List[str], List[Dict[str, Any]], np.ndarray]:
        """
        Performs simultaneous YOLOv8 hazard detection and Universal Scene OCR.

        Returns:
            Tuple of:
              - structured_labels: List of natural language warnings & OCR reads.
              - detections: Comprehensive list of detected hazards and text objects.
              - annotated_frame: Frame with color-coded hazard boxes, bright Cyan/Magenta OCR boxes, and HUD.
        """
        if frame is None or frame.size == 0:
            raise ValueError("Input frame is invalid or empty.")

        frame_h, frame_w = frame.shape[:2]
        effective_conf = conf_threshold if conf_threshold is not None else self.conf_threshold
        self.frame_count += 1

        # -------------------------------------------------------------------
        # 1. YOLOv8 Spatial Hazard Detection
        # -------------------------------------------------------------------
        results = self.model.predict(
            source=frame,
            device=self.device,
            conf=effective_conf,
            classes=WHITELIST_COCO_IDS,
            imgsz=self.imgsz,
            verbose=False,
        )

        hazard_detections: List[Dict[str, Any]] = []

        if results and len(results) > 0:
            result = results[0]
            boxes = result.boxes

            if boxes is not None and len(boxes) > 0:
                xyxy_coords = boxes.xyxy.cpu().numpy()
                confidences = boxes.conf.cpu().numpy()
                class_ids = boxes.cls.cpu().numpy().astype(int)
                names_dict = result.names or {}

                for i in range(len(xyxy_coords)):
                    conf = float(confidences[i])
                    cls_id = int(class_ids[i])
                    raw_name = names_dict.get(cls_id, f"obj_{cls_id}").lower().strip()

                    # Blacklist / Noise filter
                    if raw_name in BLACKLIST_CLASS_NAMES:
                        continue
                    if (cls_id not in WHITELIST_COCO_IDS) and (raw_name not in WHITELIST_CLASS_NAMES):
                        continue

                    x1 = max(0, int(xyxy_coords[i][0]))
                    y1 = max(0, int(xyxy_coords[i][1]))
                    x2 = min(frame_w, int(xyxy_coords[i][2]))
                    y2 = min(frame_h, int(xyxy_coords[i][3]))

                    box_h = max(1, y2 - y1)
                    box_height_ratio = box_h / float(frame_h)
                    bottom_y_ratio = y2 / float(frame_h)

                    zone = self.calculate_hazard_zone(y1, y2, box_height_ratio, frame_h)
                    proximity = self.calculate_proximity(box_height_ratio, bottom_y_ratio)
                    mapped_label = CLASS_LABEL_MAPPING.get(raw_name, raw_name.capitalize())

                    hazard_detections.append({
                        "type": "hazard",
                        "raw_name": raw_name,
                        "label": mapped_label,
                        "confidence": round(conf, 3),
                        "bbox": [x1, y1, x2, y2],
                        "zone": zone,
                        "proximity": proximity,
                        "box_height_ratio": round(box_height_ratio, 3),
                        "bottom_y_ratio": round(bottom_y_ratio, 3),
                        "associated_text": [],
                    })

        # -------------------------------------------------------------------
        # 2. Universal Scene OCR Pipeline (Dual Strategy)
        # -------------------------------------------------------------------
        # Strategy A: Periodic Full-Frame OCR Scan (every 10 frames)
        if (self.frame_count % self.ocr_interval_frames == 1) or not self.cached_scene_text:
            self.cached_scene_text = self._scan_full_frame_ocr(frame)

        # Strategy B: Immediate ROI Crop OCR on Transit / Signage Hazards
        roi_text_detections: List[Dict[str, Any]] = []

        for haz in hazard_detections:
            raw_name = haz["raw_name"]
            if raw_name in ALL_ROI_CLASSES:
                hx1, hy1, hx2, hy2 = haz["bbox"]
                roi_w = hx2 - hx1
                roi_h = hy2 - hy1

                # Ensure valid ROI size
                if roi_w >= 24 and roi_h >= 24:
                    crop = frame[hy1:hy2, hx1:hx2]
                    texts_in_roi = self._run_ocr_on_image(crop, offset_x=hx1, offset_y=hy1)

                    category = (
                        CAT_TRANSIT
                        if raw_name in TRANSIT_ROI_CLASSES
                        else CAT_SIGNAGE
                    )

                    for t_item in texts_in_roi:
                        t_item["category"] = category
                        t_item["parent_hazard"] = haz["label"]
                        haz["associated_text"].append(t_item["text"])
                        roi_text_detections.append(t_item)

        # -------------------------------------------------------------------
        # 3. Categorize & Merge Full-Frame Scene Text
        # -------------------------------------------------------------------
        all_text_detections: List[Dict[str, Any]] = list(roi_text_detections)

        for scene_item in self.cached_scene_text:
            sx1, sy1, sx2, sy2 = scene_item["bbox"]
            s_center_x = (sx1 + sx2) / 2.0
            s_center_y = (sy1 + sy2) / 2.0

            # Check if this scene text is already captured by an ROI detection
            is_duplicate = False
            for roi_item in roi_text_detections:
                rx1, ry1, rx2, ry2 = roi_item["bbox"]
                # Spatial overlap check
                if (abs(sx1 - rx1) < 20 and abs(sy1 - ry1) < 20) or (
                    scene_item["text"].upper() == roi_item["text"].upper()
                ):
                    is_duplicate = True
                    break

            if is_duplicate:
                continue

            # Determine category: check if center falls inside any hazard bbox
            assigned_category = CAT_SCENE
            parent_label = None

            for haz in hazard_detections:
                hx1, hy1, hx2, hy2 = haz["bbox"]
                if hx1 <= s_center_x <= hx2 and hy1 <= s_center_y <= hy2:
                    if haz["raw_name"] in TRANSIT_ROI_CLASSES:
                        assigned_category = CAT_TRANSIT
                    elif haz["raw_name"] in SIGNAGE_ROI_CLASSES:
                        assigned_category = CAT_SIGNAGE
                    parent_label = haz["label"]
                    if scene_item["text"] not in haz["associated_text"]:
                        haz["associated_text"].append(scene_item["text"])
                    break

            item_copy = dict(scene_item)
            item_copy["type"] = "ocr"
            item_copy["category"] = assigned_category
            item_copy["parent_hazard"] = parent_label
            all_text_detections.append(item_copy)

        # -------------------------------------------------------------------
        # 4. Generate Structured Language Labels (for TTS Audio Assistance)
        # -------------------------------------------------------------------
        structured_labels: List[str] = []

        # A. Hazard & Transit Audio Alerts
        for haz in hazard_detections:
            lbl = haz["label"]
            prox = haz["proximity"]
            zone = haz["zone"]
            raw_cls = haz["raw_name"]
            texts = haz["associated_text"]

            if texts and raw_cls in TRANSIT_ROI_CLASSES:
                route_str = " ".join(texts)
                structured_labels.append(f"{lbl} detected {prox} with Route: '{route_str}'")
            elif texts and raw_cls in SIGNAGE_ROI_CLASSES:
                sign_str = " ".join(texts)
                structured_labels.append(f"{lbl} Read: '{sign_str}' detected {prox}")
            else:
                # Standard spatial hazard alert
                if zone == ZONE_OVERHEAD:
                    structured_labels.append(f"Overhead {lbl} detected {prox.lower()}")
                elif raw_cls in DYNAMIC_CLASSES:
                    structured_labels.append(f"Moving {lbl} detected {prox.lower()} on {zone}")
                else:
                    structured_labels.append(f"{lbl} detected {prox.lower()} on {zone}")

        # B. General Scene Text Alerts
        for txt in all_text_detections:
            if txt["category"] == CAT_SCENE:
                structured_labels.append(f"Scene Text Detected: '{txt['text']}'")
            elif txt["category"] == CAT_SIGNAGE and not txt.get("parent_hazard"):
                structured_labels.append(f"Street Sign Read: '{txt['text']}'")

        # -------------------------------------------------------------------
        # 5. Visual Annotations (annotated_frame)
        # -------------------------------------------------------------------
        annotated_frame = frame.copy()
        self._draw_hazard_dividers(annotated_frame, frame_w, frame_h)

        # Draw YOLO Hazard Bounding Boxes
        for haz in hazard_detections:
            self._draw_hazard_box(annotated_frame, haz)

        # Draw RapidOCR Text Bounding Boxes (Cyan / Magenta)
        for txt in all_text_detections:
            self._draw_ocr_box(annotated_frame, txt)

        # Combine all detections for programmatic callers
        all_detections: List[Dict[str, Any]] = hazard_detections + all_text_detections

        return structured_labels, all_detections, annotated_frame

    def _draw_hazard_dividers(self, frame: np.ndarray, frame_w: int, frame_h: int) -> None:
        """Draws horizon and perspective guidance lines."""
        overlay = frame.copy()
        y_overhead = int(frame_h * 0.35)
        y_ground = int(frame_h * 0.50)
        y_critical = int(frame_h * 0.85)

        cv2.line(overlay, (0, y_overhead), (frame_w, y_overhead), (180, 180, 255), 1, cv2.LINE_AA)
        cv2.line(overlay, (0, y_ground), (frame_w, y_ground), (255, 180, 0), 1, cv2.LINE_AA)
        cv2.line(overlay, (0, y_critical), (frame_w, y_critical), (0, 0, 255), 1, cv2.LINE_AA)

        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(overlay, "UPPER BODY BOUNDARY (y < 35%)", (10, y_overhead - 6), font, 0.38, (200, 200, 255), 1, cv2.LINE_AA)
        cv2.putText(overlay, "GROUND PATH HORIZON (y >= 50%)", (10, y_ground - 6), font, 0.38, (255, 200, 100), 1, cv2.LINE_AA)
        cv2.putText(overlay, "CRITICAL CLOSE DEPTH (y >= 85%)", (10, y_critical - 6), font, 0.38, (100, 100, 255), 1, cv2.LINE_AA)

        cv2.addWeighted(overlay, 0.45, frame, 0.55, 0, frame)

    def _draw_hazard_box(self, frame: np.ndarray, haz: Dict[str, Any]) -> None:
        """Draws color-coded hazard box (Red for CLOSE, Yellow for MEDIUM, Green for FAR)."""
        x1, y1, x2, y2 = haz["bbox"]
        prox = haz["proximity"]
        label = haz["label"]
        conf = haz["confidence"]

        if prox == PROXIMITY_CLOSE:
            box_color = COLOR_CLOSE
            thickness = 3
        elif prox == PROXIMITY_MEDIUM:
            box_color = COLOR_MEDIUM
            thickness = 2
        else:
            box_color = COLOR_FAR
            thickness = 2

        cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, thickness, cv2.LINE_AA)

        tag_main = f"{label} [{prox}] {int(conf * 100)}%"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.50
        (tw, th), _ = cv2.getTextSize(tag_main, font, font_scale, 1)

        pill_y1 = max(0, y1 - th - 6)
        pill_y2 = pill_y1 + th + 6
        pill_x2 = min(frame.shape[1], x1 + tw + 10)

        cv2.rectangle(frame, (x1, pill_y1), (pill_x2, pill_y2), COLOR_BOX_BG, -1)
        cv2.rectangle(frame, (x1, pill_y1), (pill_x2, pill_y2), box_color, 1)
        cv2.putText(frame, tag_main, (x1 + 5, pill_y2 - 4), font, font_scale, box_color, 1, cv2.LINE_AA)

        if prox == PROXIMITY_CLOSE:
            alert_lbl = "! CLOSE HAZARD !"
            (aw, ah), _ = cv2.getTextSize(alert_lbl, font, 0.42, 1)
            ax1 = max(0, x2 - aw - 8)
            ay1 = max(0, y1 - ah - 6)
            cv2.rectangle(frame, (ax1, ay1), (x2, y1), COLOR_CLOSE, -1)
            cv2.putText(frame, alert_lbl, (ax1 + 4, y1 - 3), font, 0.42, (255, 255, 255), 1, cv2.LINE_AA)

    def _draw_ocr_box(self, frame: np.ndarray, txt_item: Dict[str, Any]) -> None:
        """
        Draws bright Cyan / Magenta bounding boxes around detected text regions
        with high-contrast legible text label overlays.
        """
        x1, y1, x2, y2 = txt_item["bbox"]
        category = txt_item.get("category", CAT_SCENE)
        text = txt_item["text"]
        conf = txt_item.get("confidence", 0.0)

        # Color coding: Magenta for Signage, Deep Cyan for Transit, Bright Cyan for General Scene Text
        if category == CAT_SIGNAGE:
            box_color = COLOR_OCR_MAGENTA
            prefix = "SIGN"
        elif category == CAT_TRANSIT:
            box_color = COLOR_OCR_BLUE
            prefix = "ROUTE"
        else:
            box_color = COLOR_OCR_CYAN
            prefix = "TEXT"

        # Draw text boundary
        cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2, cv2.LINE_AA)

        # Draw polygon if available for rotated/perspective text
        if "polygon" in txt_item:
            poly_pts = np.array(txt_item["polygon"], dtype=np.int32)
            cv2.polylines(frame, [poly_pts], isClosed=True, color=box_color, thickness=1, lineType=cv2.LINE_AA)

        # Label pill above text
        tag = f"[{prefix}] '{text}' ({int(conf * 100)}%)"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.42
        (tw, th), _ = cv2.getTextSize(tag, font, font_scale, 1)

        pill_y1 = max(0, y1 - th - 6)
        pill_y2 = pill_y1 + th + 6
        pill_x2 = min(frame.shape[1], x1 + tw + 10)

        cv2.rectangle(frame, (x1, pill_y1), (pill_x2, pill_y2), (10, 10, 10), -1)
        cv2.rectangle(frame, (x1, pill_y1), (pill_x2, pill_y2), box_color, 1)
        cv2.putText(frame, tag, (x1 + 4, pill_y2 - 3), font, font_scale, box_color, 1, cv2.LINE_AA)


# ---------------------------------------------------------------------------
# Module-Level Convenience Function
# ---------------------------------------------------------------------------
_GLOBAL_DETECTOR: Optional[HazardDetector] = None

def get_default_detector(
    model_path: str = "models/yolov8n.pt", conf_threshold: float = 0.50
) -> HazardDetector:
    """Returns or creates a cached singleton instance of HazardDetector."""
    global _GLOBAL_DETECTOR
    if _GLOBAL_DETECTOR is None:
        _GLOBAL_DETECTOR = HazardDetector(model_path=model_path, conf_threshold=conf_threshold)
    return _GLOBAL_DETECTOR


def detect_objects(
    frame: np.ndarray, conf_threshold: float = 0.50
) -> Tuple[List[str], List[Dict[str, Any]], np.ndarray]:
    """
    Standard function API:
    Takes an OpenCV BGR frame and returns (structured_labels, detections, annotated_frame).
    """
    detector = get_default_detector(conf_threshold=conf_threshold)
    return detector.detect_objects(frame, conf_threshold=conf_threshold)
