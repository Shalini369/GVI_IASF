# Assisted Vision: Real-Time Spatial Hazards & Universal Scene OCR (GVI)

Production-grade real-time obstacle hazard detection and **Universal Scene Text Reading (OCR)** pipeline engineered for visually impaired navigation assistance. Specially optimized for CPU execution on **Lenovo Snapdragon X Plus** laptops running **Windows ARM64**.

---

## Architecture Overview

```
                      +------------------------------------------+
                      |         Real-Time Camera Stream          |
                      +------------------------------------------+
                                           |
                    +----------------------+----------------------+
                    |                                             |
                    v                                             v
     +------------------------------+             +-------------------------------+
     |   YOLOv8 Hazard Detection    |             |   Universal Scene OCR Engine  |
     |   (Spatial Proximity & Zones)|             |         (RapidOCR ONNX)       |
     +------------------------------+             +-------------------------------+
                    |                                             |
                    |                             +---------------+---------------+
                    |                             |                               |
                    v                             v                               v
     +------------------------------+  +----------------------+      +--------------------+
     | Whitelist: Person, Vehicle,  |  | Periodic Full Scan   |      | Immediate ROI Crop |
     | Transit, Signs, Curbs, Doors |  | (Every 10 frames)    |      | (Bus, Stop Sign,   |
     +------------------------------+  +----------------------+      |  Door, Light)      |
                    |                             |                  +--------------------+
                    +----------------------+      |                            |
                                           |      |                            |
                                           v      v                            v
                                  +-----------------------------------------------+
                                  |   Categorization & Spatial Association        |
                                  |   - BUS ROUTE / TRANSIT TEXT                  |
                                  |   - SIGNAGE / NAVIGATION TEXT                 |
                                  |   - SCENE TEXT (Street, Store, Doors)         |
                                  +-----------------------------------------------+
                                                          |
                                                          v
                                  +-----------------------------------------------+
                                  |  Assistive TTS Speech & High-Contrast Visual  |
                                  |  - Cyan/Magenta OCR Overlays                  |
                                  |  - Red (CLOSE), Yellow (MEDIUM), Green (FAR)  |
                                  +-----------------------------------------------+
```

---

## Key Features

### 1. Universal Scene OCR Pipeline (RapidOCR ONNXRuntime)
- **Dual Execution Strategy**:
  - **Periodic Full-Frame Scan**: Runs RapidOCR on the downscaled frame (max width 800px) every 10 frames (~0.3–0.4s interval). Reads all outdoor/indoor environmental text: store names, street signs, building numbers, warning banners.
  - **Immediate ROI Crop OCR**: Instant sub-frame extraction whenever a **Bus**, **Truck**, **Stop Sign**, **Door**, or **Traffic Light** is detected by YOLO, enabling instant route/sign reading with near-zero latency.
- **Smart Text Categorization**:
  - `BUS ROUTE / TRANSIT TEXT` (Deep Cyan box)
  - `SIGNAGE / NAVIGATION TEXT` (Magenta box)
  - `SCENE TEXT` (Bright Cyan box)
- **Noise Rejection**: Strips non-alphanumerics, rejects single-character noise, and filters text below $0.45$ confidence.
- **Persistence Caching**: Full-frame OCR boxes are cached for 10 frames so overlays remain stable and continuous on-screen while sustaining $20-30+$ FPS on CPU.

### 2. Spatial Hazard Detection (YOLOv8)
- **Whitelisted Hazard Classes**: `['person', 'bicycle', 'car', 'motorcycle', 'bus', 'truck', 'traffic light', 'fire hydrant', 'stop sign', 'bench', 'chair', 'potted plant', 'door']`.
- **Perspective Proximity Rules**:
  - **CLOSE**: $\text{box\_height\_ratio} > 0.45 \quad\mathbf{OR}\quad \text{bottom\_y\_ratio} > 0.85$ (Red Bounding Box)
  - **MEDIUM**: $0.20 \le \text{box\_height\_ratio} \le 0.45 \quad\mathbf{OR}\quad 0.60 \le \text{bottom\_y\_ratio} \le 0.85$ (Yellow Bounding Box)
  - **FAR**: $\text{box\_height\_ratio} < 0.20 \quad\mathbf{AND}\quad \text{bottom\_y\_ratio} < 0.60$ (Green Bounding Box)
- **Spatial Hazard Zones**:
  - **OVERHEAD / UPPER BODY**: $y_1 < 0.35 \times H$ and $\text{box\_height\_ratio} > 0.25$ (Head-level hazard protection).
  - **GROUND PATH**: $y_2 \ge 0.50 \times H$ (Trip hazards, curbs, pavement obstacles).

---

## Installation & Setup

```bash
conda activate gvi_iasf
pip install -r requirements.txt
```

---

## Running the Application

### Real-Time Live Webcam Stream (Camera 0)
```bash
python main.py
```

### Video File or Custom Camera Index
```bash
python main.py --source 0 --conf 0.50 --ocr-interval 10
python main.py --source path/to/video.mp4
```

### Interactive Hotkeys
- **`+`** / **`=`**: Increase YOLO confidence threshold by `0.05`.
- **`-`** / **`_`**: Decrease YOLO confidence threshold by `0.05`.
- **`c`**: Clear OCR scene text cache immediately.
- **`q`** / **`ESC`**: Safely exit stream.

---

## Programmatic API Usage

```python
import cv2
from modules.object_detection import HazardDetector, detect_objects

# Option 1: Module-level convenience function
frame = cv2.imread("street_scene.jpg")
structured_labels, detections, annotated_frame = detect_objects(frame, conf_threshold=0.50)

# Natural language output for TTS engine
for label in structured_labels:
    print("Voice assist:", label)
    # Output examples:
    # "Bus detected CLOSE with Route: '42B'"
    # "Scene Text Detected: 'STARBUCKS COFFEE'"
    # "Street Sign Read: 'MAIN ST'"
    # "Pedestrian detected CLOSE on Ground Path"

# Option 2: Stateful detector instance
detector = HazardDetector(
    model_path="models/yolov8n.pt",
    conf_threshold=0.50,
    ocr_interval_frames=10,
    ocr_max_width=800,
)
structured_labels, detections, annotated_frame = detector.detect_objects(frame)
```
