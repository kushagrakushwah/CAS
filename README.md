# 🚶 CrowdAware AI: Person & Crowd Awareness System

![Python 3.8+](https://img.shields.io/badge/Python-3.8%2B-blue.svg)
![YOLOv8](https://img.shields.io/badge/YOLO-v8-yellow)
![OpenCV](https://img.shields.io/badge/OpenCV-4.8%2B-green)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c)
![License](https://img.shields.io/badge/License-MIT-purple)

**CrowdAware AI** is a real-time, end-to-end computer-vision system designed to enhance spatial awareness. It detects nearby people, estimates their approximate distance and movement direction, and provides intelligent audio alerts when individuals are approaching or blocking your path.

Built with performance in mind, the system leverages YOLOv8 for rapid object detection, Kalman filtering for trajectory tracking, and an intelligent Text-to-Speech (TTS) audio engine to provide situational awareness without requiring you to look at a screen.

---

## ✨ Key Features

*   🤖 **Real-Time Person Detection:** Powered by Ultralytics YOLOv8 (supports Nano to X-Large models) for high-speed, accurate detection.
*   📏 **Monocular Distance Estimation:** Uses a calibrated pinhole camera model to estimate real-world distance based on bounding box height, without needing expensive LiDAR or stereo cameras.
*   🎯 **Movement & Tracking (SORT-based):** Employs Kalman filters and the Hungarian algorithm to track individuals across frames, calculating their speed, approach rate, and compass direction.
*   🔊 **Intelligent Audio Engine:**
    *   **Smart Alerts:** Provides critical warnings (`"Warning! Person 1.5 meters ahead"`), approach notifications, and path-blocking alerts.
    *   **Spatial Audio:** Pans audio left/right based on the person's position in the camera view.
    *   **Anti-Spam Cooldowns:** Priority queues and intelligent cooldowns prevent alert fatigue.
*   🛑 **Zone & Crowd Analysis:** Defines proximity zones (Critical, Close, Near, Medium, Far) and alerts you to high crowd density or path obstructions.
*   📊 **Rich UI Dashboard:** OpenCV visualization overlay featuring color-coded bounding boxes, movement arrows, trajectory trails, and a live statistics dashboard.
*   🧪 **Developer Friendly:** Includes synthetic demos, comprehensive unit tests, and a highly customizable YAML configuration.

---

## 🏗️ System Architecture

The pipeline processes video frames continuously through a coordinated, single-threaded loop (with optional background inference threading):

1.  **Camera Capture:** Grabs frames via OpenCV.
2.  **Detection (`detector.py`):** YOLOv8 infers bounding boxes and confidences.
3.  **Distance Estimation (`distance.py`):** Calculates depth using `(focal_length × known_height) / bbox_height`.
4.  **Tracking (`tracker.py`):** Predicts trajectories and associates tracks, classifying movement (approaching, retreating, stationary).
5.  **Alert Coordination (`alert_coordinator.py` & `alert_engine.py`):** Evaluates track states against zones and safely enqueues TTS alerts.
6.  **Visualization (`visualizer.py`):** Renders the HUD and dashboard.

---

## 🚀 Getting Started

### 1. Prerequisites
*   **OS:** Windows, macOS, or Linux
*   **Python:** Version 3.8 or higher
*   **Hardware:** A webcam. A GPU (CUDA/MPS) is highly recommended for higher FPS, but the `yolov8n.pt` model runs comfortably on modern CPUs.

### 2. Installation
Clone the repository and install the required dependencies:
```bash
git clone https://github.com/yourusername/crowd-awareness-system.git
cd crowd-awareness-system

# It is recommended to use a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Quick Start (Live Camera)
Run the main script to start the system using your default webcam:
```bash
python main.py
```
*Press **`Q`** or **`ESC`** in the video window to quit.*

---

## 🛠️ Configuration & CLI Options

The system is highly configurable via the `config/settings.yaml` file and command-line overrides.

### Command-Line Arguments
```bash
python main.py --help

Options:
  -c, --camera INDEX     Camera device index (default: 0)
  -s, --source PATH      Path to video file instead of live camera
  -m, --model MODEL      YOLOv8 model size (e.g., yolov8n.pt, yolov8s.pt)
  --no-audio             Disable voice alerts
  --no-display           Headless mode (no video window)
  --config PATH          Path to alternate YAML configuration file
```

### Key Configurations (`config/settings.yaml`)
*   **Zones:** Define proximity thresholds in meters (Critical: 1.0, Close: 2.5, Near: 4.0).
*   **Audio:** Adjust TTS volume, speech rate, and toggle spatial audio or beep alerts.
*   **Tracking:** Tweak Kalman filter noise parameters and max track age.

---

## 🎯 Distance Calibration

For the distance estimation to be accurate, you **must calibrate the camera's focal length**. The default in `settings.yaml` is an approximation.

1.  Stand a person of average height (e.g., 1.70m) at a **known, exact distance** from the camera (e.g., 2.0 meters).
2.  Run the main script: `python main.py`
3.  Press **`C`** on your keyboard. Note the bounding box height (in pixels) printed to the console.
4.  Run the calibration script:
    ```bash
    python scripts/calibrate.py --distance 2.0 --height <measured_pixel_height>
    ```
5.  Update the `focal_length_px` variable in your `config/settings.yaml` with the output value.

---

## 🎮 Demo Modes

Don't have a camera or want to test without walking around? Use the built-in demo script!

**Synthetic Animated Demo:**
Generates a virtual 2D scene with animated boxes representing people, allowing you to test tracking and audio logic instantly.
```bash
python scripts/demo.py --synthetic
```

**Video File Demo:**
Run the pipeline on a pre-recorded video.
```bash
python scripts/demo.py --video path/to/test_video.mp4
```

---

## 🧪 Testing

The project includes a robust suite of unit tests for detection, distance math, tracking algorithms, and audio queueing.

Run the test suite using `pytest`:
```bash
pytest tests/ -v
```
To generate a coverage report:
```bash
pytest tests/ -v --cov=src
```

---

## 📁 Project Structure

```text
crowd-awareness-system/
├── main.py                  # Main entry point and CLI
├── requirements.txt         # Python dependencies
├── config/
│   └── settings.yaml        # Master configuration file
├── scripts/
│   ├── benchmark.py         # Performance profiling script
│   ├── calibrate.py         # Focal length calibration utility
│   └── demo.py              # Synthetic and video demo runners
├── tests/
│   └── test_detection.py    # Unit tests
└── src/
    ├── pipeline.py          # Orchestrates detection, tracking, & UI
    ├── audio/
    │   ├── alert_coordinator.py # Decides *when* and *what* to alert
    │   └── alert_engine.py      # Thread-safe TTS & audio playback
    ├── detection/
    │   ├── detector.py      # YOLOv8 integration
    │   ├── distance.py      # Pinhole camera distance estimation
    │   └── tracker.py       # Kalman-filter object tracking
    └── ui/
        └── visualizer.py    # OpenCV HUD and Dashboard rendering
```

---

## 🔮 Future Enhancements
*   **Depth Camera Support:** Integration with Intel RealSense or Oak-D for absolute depth measurement.
*   **Pose Estimation:** Utilizing YOLOv8-Pose to determine if a person is facing the user or looking away.
*   **Mobile Deployment:** Exporting models via ONNX/CoreML for Android and iOS devices.

---

## 📄 License
This project is licensed under the MIT License - see the LICENSE file for details.

---
*Built with ❤️ by the CrowdAware AI Team.*
