# CrowdAware AI: Person and Crowd Awareness System

![Python 3.8+](https://img.shields.io/badge/Python-3.8%2B-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-MobileNetV3--SSDLite-orange)
![OpenCV](https://img.shields.io/badge/OpenCV-4.8%2B-green)
![Zero-YOLO](https://img.shields.io/badge/Zero--YOLO-Independent-blueviolet)
![License](https://img.shields.io/badge/License-MIT-purple)

**CrowdAware AI** is a real-time, end-to-end computer-vision application designed to enhance spatial awareness for the user. By processing live camera or video feeds, the system detects nearby people, estimates their approximate physical distance, calculates their movement direction, and provides intelligent, non-intrusive audio alerts when individuals are approaching rapidly or blocking the user's path.

This system is completely independent of external YOLO or Ultralytics libraries. It features a custom trainable **PyTorch MobileNetV3-SSDLite** pedestrian detection architecture, an automated dataset ingestion pipeline, Kalman filtering for trajectory tracking, and a dedicated, thread-safe Text-to-Speech (TTS) audio engine.

---

## Key Features

* **Custom Trainable Deep Learning Detector (Zero-YOLO):**
  A native PyTorch MobileNetV3-Large + SSDLite architecture built directly using PyTorch/Torchvision primitives (no Ultralytics or black-box YOLO dependencies). The model can be trained and fine-tuned from scratch on any pedestrian dataset.
* **Automated Dataset Downloader & Loader:**
  Includes automated ingestion for standard pedestrian datasets (such as the Penn-Fudan Pedestrian Dataset with 345 annotated pedestrians) as well as an offline synthetic dataset generator.
* **Monocular Distance Estimation:**
  Utilizes a calibrated pinhole camera model. By comparing the detected pixel bounding-box height of a person to a known real-world average height (1.70 m) and the calibrated focal length, it estimates depth in real time without requiring LiDAR or stereo cameras.
* **Movement and Trajectory Tracking:**
  Employs a custom implementation of SORT (Simple Online and Realtime Tracking) using a 7-dimensional Kalman filter state vector `[cx, cy, scale, ratio, vx, vy, vscale]` paired with the Hungarian assignment algorithm (via SciPy) to track individuals across frames, calculate lateral/radial velocities, and classify movement into 8 compass headings.
* **Intelligent Audio Engine:**
  * **Smart Alerts:** Dispatches contextual warnings (e.g., "Warning! Person 1.5 meters ahead", "Someone approaching from your left", "Path ahead is blocked").
  * **Spatial Audio Panning:** Dynamically pans the audio output to the left or right stereo channel based on the tracked person's horizontal position in the camera view.
  * **Asynchronous Execution:** Runs on a dedicated background thread using `queue.PriorityQueue` to ensure audio rendering never bottlenecks the OpenCV video processing pipeline.
  * **Anti-Spam Cooldowns:** Enforces strict timing constraints per alert category to prevent alert fatigue.
* **Zone and Corridor Analysis:**
  Classifies targets into configurable proximity zones (Critical, Close, Near, Medium, Far). It actively monitors the center corridor of the screen to detect if the path is obstructed and counts total active tracks to warn about high crowd density.
* **Rich UI Dashboard:**
  An OpenCV-rendered Heads-Up Display (HUD) featuring color-coded bounding boxes, movement vectors (arrows), fading trajectory trails, and a live statistics dashboard detailing system FPS and active track states.

---

## System Architecture

The pipeline processes video frames continuously through a coordinated loop located in `src/pipeline.py`:

```
  [ Video Feed / Camera ]
             │
             ▼
  ┌─────────────────────────────────────────────────────────┐
  │ 1. Custom MobileNetV3-SSDLite Detector                  │
  │    - Normalized tensor input                            │
  │    - Multi-scale anchor box predictions                 │
  │    - Non-Maximum Suppression (NMS)                      │
  └──────────────────────────┬──────────────────────────────┘
                             │ Detected BBoxes [x1, y1, x2, y2]
                             ▼
  ┌─────────────────────────────────────────────────────────┐
  │ 2. Monocular Distance & Depth Estimator                 │
  │    - Pinhole formula: d = (Focal_Length_px * H) / H_px  │
  │    - 5 Proximity Zones: Critical, Close, Near, Med, Far │
  └──────────────────────────┬──────────────────────────────┘
                             │ BBoxes + Distances
                             ▼
  ┌─────────────────────────────────────────────────────────┐
  │ 3. Kalman Filter & SORT Tracker                         │
  │    - State vector: [cx, cy, s, r, vx, vy, vs]           │
  │    - Hungarian assignment algorithm                     │
  │    - Radial approach rate: dr/dt                        │
  │    - Movement direction angle: atan2(-dy, dx)           │
  └──────────────────────────┬──────────────────────────────┘
                             │ Active Tracks with Velocity & Headings
                             ▼
  ┌─────────────────────────────────────────────────────────┐
  │ 4. Hazard & Path-Corridor Analysis Engine               │
  │    - Central corridor boundary intersection             │
  │    - Time-to-Collision (TTC) metric                     │
  │    - Crowd density threshold monitor                    │
  └──────────────────────────┬──────────────────────────────┘
                             │ Alert Events
                             ▼
  ┌──────────────────────────┴──────────────────────────────┐
  │                                                         │
  ▼                                                         ▼
┌─────────────────────────────────┐   ┌───────────────────────────────────┐
│ 5. Asynchronous Audio Alert Sys │   │ 6. High-Contrast HUD Visualizer   │
│    - Priority Queue             │   │    - Zone-colored bounding boxes  │
│    - Spatial Stereo Panning     │   │    - Path corridor overlay        │
│    - Category Anti-Spam Timers  │   │    - Trajectory trails & vectors  │
│    - Native TTS Engine          │   │    - Real-time telemetry dashboard│
└─────────────────────────────────┘   └───────────────────────────────────┘
```

---

## Installation Guide

### 1. Prerequisites
* **Operating System:** Windows, macOS, or Linux.
* **Python:** Version 3.8 to 3.12.
* **Hardware:** A standard webcam. NVIDIA GPU (CUDA) or Apple Silicon (MPS) supported, but runs smoothly on CPU.

### 2. Environment Setup
```bash
git clone https://github.com/kushagrakushwah/CAS.git
cd CAS

# Create and activate a virtual environment
python -m venv venv
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# Install the Python dependencies
pip install -r requirements.txt
```

---

## Dataset Acquisition & Model Training

### 1. Download the Dataset
Download and extract the standard Penn-Fudan Pedestrian Dataset (or generate a fallback dataset):
```bash
python scripts/download_dataset.py
```
This stores the images in `data/PennFudanPed/PNGImages` and bounding box annotations in `data/PennFudanPed/Annotation`.

### 2. Train Your Custom Detector
Train the MobileNetV3-SSDLite architecture on the dataset:
```bash
python train.py --epochs 5 --batch-size 4 --lr 0.001
```

Options for `train.py`:
* `--epochs, -e`: Number of training epochs (default: 5).
* `--batch-size, -b`: Batch size (default: 4).
* `--lr`: Learning rate (default: 0.001).
* `--data, -d`: Path to dataset directory (default: `data/PennFudanPed`).
* `--output-dir, -o`: Directory where `.pth` checkpoints are saved (default: `models`).
* `--device`: Compute device (`cpu`, `cuda`, `mps`).

Once training finishes, the best checkpoint is automatically saved to:
`models/best_person_detector.pth`

---

## Running the Real-Time System

To run the awareness system using your newly trained model and webcam:
```bash
python main.py --model models/best_person_detector.pth
```
*Press **Q** or **ESC** in the video window to quit.*

### Command-Line Arguments
```text
Options:
  -c, --camera INDEX     Camera device index (default: 0).
  -s, --source PATH      Path to video file (.mp4, .avi) instead of live camera.
  -m, --model MODEL      Path to model checkpoint (default: models/best_person_detector.pth).
  --no-audio             Disable audio voice alerts.
  --no-display           Headless mode (no OpenCV GUI window).
  --confidence FLOAT     Override detection confidence threshold (0.0 to 1.0).
  --device STRING        Force compute device ('cpu', 'cuda', 'mps').
```

---

## Distance Calibration

For physical distance estimation to be accurate across different webcams and lenses, calibrate the camera's focal length:

1. Stand a person of average height (e.g., 1.70 meters) at an exact measured distance (e.g., 2.0 meters) in front of the camera.
2. Start the application: `python main.py`
3. Press the **C** key on your keyboard to note the measured bounding box pixel height.
4. Run the calibration script:
   ```bash
   python scripts/calibrate.py --distance 2.0 --height <measured_pixel_height>
   ```
5. Update the `focal_length_px` in `config/settings.yaml` with the output value.

---

## Demo Modes

To test without a physical camera or live pedestrians:

**Synthetic Animated Demo:**
Simulates moving persons and verifies tracking, corridor blocking, and HUD rendering:
```bash
python scripts/demo.py --synthetic --frames 300
```

**Video File Demo:**
Run the entire pipeline on a recorded video:
```bash
python scripts/demo.py --video path/to/video.mp4
```

---

## Testing Framework

The project includes 34 PyTest unit tests covering model architecture, dataset loading, detection parsing, distance estimation geometry, and SORT tracking:
```bash
pytest tests/ -v
```

---

## Project Structure

```text
CAS/
├── main.py                  # Real-time application entry point
├── train.py                 # Custom deep learning model training script
├── requirements.txt         # Core dependencies
├── config/
│   └── settings.yaml        # Runtime configuration (detector, camera, audio, zones)
├── data/
│   └── PennFudanPed/        # Downloaded pedestrian training dataset
├── models/
│   └── best_person_detector.pth # Custom trained PyTorch model checkpoint
├── scripts/
│   ├── download_dataset.py  # Ingestion script for pedestrian datasets
│   ├── calibrate.py         # Focal length calibration utility
│   ├── demo.py              # Demo script (synthetic or video)
│   └── generate_pdf.py      # Educational PDF guide generator
├── src/
│   ├── pipeline.py          # Master real-time execution loop
│   ├── training/
│   │   ├── dataset.py       # PennFudanPedDataset loader and collate functions
│   │   ├── model.py         # MobileNetV3 + SSDLite 2-class architecture
│   │   └── trainer.py       # Training loop with multi-task loss tracking
│   ├── detection/
│   │   ├── mobilenet_detector.py # Zero-YOLO PyTorch detector for inference
│   │   ├── detector.py      # Detection dataclass and legacy fallback
│   │   └── distance.py      # Pinhole camera distance estimation
│   ├── tracking/
│   │   └── tracker.py       # Kalman-filter object tracking and direction mapping
│   ├── audio/
│   │   ├── alert_coordinator.py # Logic evaluating when and what to speak
│   │   └── alert_engine.py  # Thread-safe TTS with spatial panning
│   └── ui/
│       └── visualizer.py    # OpenCV HUD and Dashboard rendering
└── tests/
    ├── test_detection.py    # Tests for distance, geometry, Kalman, audio
    └── test_mobilenet_training.py # Tests for custom model, dataset, and inference
```

---

## License
This project is licensed under the MIT License - see the LICENSE file for details.
