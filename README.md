# CrowdAware AI: Person and Crowd Awareness System

![Python 3.8+](https://img.shields.io/badge/Python-3.8%2B-blue.svg)
![YOLOv8](https://img.shields.io/badge/YOLO-v8-yellow)
![OpenCV](https://img.shields.io/badge/OpenCV-4.8%2B-green)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c)
![License](https://img.shields.io/badge/License-MIT-purple)

**CrowdAware AI** is a real-time, end-to-end computer-vision application designed to enhance spatial awareness for the user. By processing live video feeds, the system detects nearby people, estimates their approximate physical distance, calculates their movement direction, and provides intelligent, non-intrusive audio alerts when individuals are approaching rapidly or blocking the user's path.

This system was built with high performance and modularity in mind. It leverages the Ultralytics YOLOv8 architecture for rapid object detection, Kalman filtering for trajectory tracking, and a dedicated, thread-safe Text-to-Speech (TTS) audio engine. It can be used for accessibility purposes (e.g., assisting visually impaired individuals), robotic navigation context, or autonomous security monitoring.

---

## Key Features

* **Real-Time Person Detection:** 
  Utilizes the YOLOv8 model (compatible with Nano through X-Large weights) to achieve high-speed, accurate bounding-box detection. It includes dynamic resolution scaling and strict Non-Maximum Suppression (NMS) to eliminate duplicate overlapping bounding boxes.
* **Monocular Distance Estimation:** 
  Instead of relying on expensive LiDAR or stereo cameras, this system uses a calibrated pinhole camera model. By comparing the detected pixel height of a person to a known real-world average height and the camera's calibrated focal length, it estimates depth in real-time.
* **Movement and Trajectory Tracking:** 
  Employs a custom implementation of SORT (Simple Online and Realtime Tracking). It uses a 7-dimensional Kalman filter state vector `[x, y, scale, ratio, dx, dy, dscale]` paired with the Hungarian assignment algorithm (via SciPy) to track individuals across frames, even through minor occlusions.
* **Intelligent Audio Engine:**
  * **Smart Alerts:** Dispatches contextual warnings (e.g., "Warning! Person 1.5 meters ahead", "Path ahead is blocked").
  * **Spatial Audio Panning:** Dynamically pans the audio output to the left or right stereo channel based on the tracked person's horizontal position in the camera view.
  * **Asynchronous Execution:** Runs on a dedicated background thread using `queue.PriorityQueue` to ensure audio rendering never bottlenecks the OpenCV video processing pipeline.
  * **Anti-Spam Cooldowns:** Enforces strict timing constraints per alert category to prevent alert fatigue.
* **Zone and Crowd Analysis:** 
  Classifies targets into configurable proximity zones (Critical, Close, Near, Medium, Far). It actively monitors the center corridor of the screen to detect if the path is obstructed and counts total active tracks to warn about high crowd density.
* **Rich UI Dashboard:** 
  An OpenCV-rendered Heads-Up Display (HUD) featuring color-coded bounding boxes, movement vectors (arrows), fading trajectory trails, and a live statistics dashboard detailing system FPS and active track states.

---

## System Architecture

The pipeline processes video frames continuously through a coordinated loop located in `src/pipeline.py`:

1. **Camera Capture:** Fetches the latest frame from the specified video device or file stream.
2. **Detection (`detector.py`):** The frame is passed through YOLOv8. The resulting tensors are parsed into `Detection` dataclasses containing coordinates, confidence scores, and class labels.
3. **Distance Estimation (`distance.py`):** For each detection, the system calculates depth using the mathematical formula: `Distance = (Focal_Length_px * Known_Height_m) / Bounding_Box_Height_px`.
4. **Tracking (`tracker.py`):** 
   * Predicts the next location of existing tracks using the Kalman Filter.
   * Computes an Intersection-over-Union (IoU) matrix between predictions and new detections.
   * Resolves assignments using the linear sum assignment algorithm.
   * Computes velocity (`dx/dy`) to classify movement as approaching, retreating, or stationary.
5. **Alert Coordination (`alert_coordinator.py` & `alert_engine.py`):** Evaluates all updated tracks. If a track breaches a zone threshold, enters the path corridor, or exhibits an approaching velocity, a request is sent to the TTS priority queue.
6. **Visualization (`visualizer.py`):** Overlays data onto the original frame and pushes it to the display buffer.

---

## Installation Guide

### 1. Prerequisites
* **Operating System:** Windows, macOS, or Linux.
* **Python:** Version 3.8 or higher is required.
* **Hardware:** A standard webcam. A CUDA-capable NVIDIA GPU or Apple Silicon (MPS) is highly recommended for optimal inference speeds, though the `yolov8n.pt` (Nano) model is optimized for CPU execution.

### 2. Environment Setup
Clone the repository and install the required dependencies:

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

*Note on PyTorch:* Depending on your hardware, you may want to install a specific version of PyTorch with CUDA support. Visit the [PyTorch Get Started](https://pytorch.org/get-started/locally/) page for the exact command for your system before running the requirements installation.

---

## Usage and CLI Options

To run the system using your default webcam, execute the main entry point:

```bash
python main.py
```
*Press **Q** or **ESC** while focused on the video window to safely shut down the system.*

### Command-Line Arguments

The `main.py` script accepts several overrides to customize runtime behavior without modifying the configuration files:

```text
Options:
  -c, --camera INDEX     Specify the camera device index (default: 0).
  -s, --source PATH      Provide a path to a video file (.mp4, .avi) instead of a live camera.
  -m, --model MODEL      Specify the YOLOv8 model size (e.g., yolov8n.pt, yolov8s.pt, yolov8m.pt).
  --no-audio             Disable the background audio TTS engine completely.
  --no-display           Run in headless mode (no OpenCV GUI window).
  --config PATH          Provide a path to a custom YAML configuration file.
  --confidence FLOAT     Override the default detection confidence threshold (0.0 to 1.0).
  --device STRING        Force inference on a specific device ('cpu', 'cuda', 'mps').
```

---

## Distance Calibration

For the distance estimation mathematics to be accurate across different webcams and lenses, you must calibrate the camera's focal length. The default value in `config/settings.yaml` is a generalized approximation.

### Calibration Steps:
1. Stand a person of average height (e.g., 1.70 meters) at an exact, measured distance from the camera (e.g., 2.0 meters).
2. Start the application: `python main.py`
3. Press the **C** key on your keyboard. Note the bounding box height (in pixels) that is printed to your console output.
4. Run the included calibration utility script with your measurements:
    ```bash
    python scripts/calibrate.py --distance 2.0 --height <measured_pixel_height>
    ```
5. The script will output a new `focal_length_px` value. Open `config/settings.yaml`, locate the `distance:` block, and update `focal_length_px` with this new number.

---

## Configuration (`settings.yaml`)

The system is highly configurable via the `config/settings.yaml` file. Key sections include:

* **Camera & Detection:** Define target FPS, frame resolution, model weights, and NMS thresholds.
* **Distance Zones:** Tune the precise meter thresholds for the `critical`, `close`, `near`, `medium`, and `far` proximity zones.
* **Audio Constraints:** Adjust the TTS engine properties (volume, WPM rate) and configure the cooldown limits (in seconds) to prevent overlapping audio spam.
* **Tracking Physics:** Modify the Kalman filter's process noise and measurement noise variables, and dictate the maximum frames a track is kept alive when occluded.

---

## Demo Modes

To test the system without requiring physical movement or a live camera feed, you can utilize the built-in demo script.

**Synthetic Animated Demo:**
This mode generates a virtual 2D canvas with animated, colored rectangles representing people moving using sine-wave velocity patterns. It allows you to verify the tracking logic, audio queueing, and UI rendering instantly.
```bash
python scripts/demo.py --synthetic --frames 500
```

**Video File Demo:**
This mode runs the entire pipeline on a pre-recorded video file for testing in specific environments.
```bash
python scripts/demo.py --video path/to/test_video.mp4
```

---

## Testing Framework

The project includes a robust suite of PyTest unit tests. These tests cover the detection math, distance estimation formulas, SORT tracking data structures, and the logic within the audio priority queue, ensuring no regressions are introduced during development.

To execute the test suite:
```bash
pytest tests/ -v
```

To generate a codebase coverage report:
```bash
pytest tests/ -v --cov=src
```

---

## Project Structure

```text
CAS/
├── main.py                  # Application entry point and CLI parser
├── requirements.txt         # Python package dependencies
├── config/
│   └── settings.yaml        # Master configuration file containing all tunable variables
├── scripts/
│   ├── benchmark.py         # Performance profiling script for FPS evaluation
│   ├── calibrate.py         # Focal length mathematical calibration utility
│   └── demo.py              # Synthetic and video demo execution runners
├── tests/
│   └── test_detection.py    # Comprehensive PyTest unit tests
└── src/
    ├── pipeline.py          # Master orchestrator connecting all sub-modules
    ├── audio/
    │   ├── alert_coordinator.py # Logic engine evaluating *when* and *what* to alert
    │   └── alert_engine.py      # Thread-safe TTS, spatial panning, and queue management
    ├── detection/
    │   ├── detector.py      # YOLOv8 tensor processing and bounding box extraction
    │   ├── distance.py      # Pinhole camera distance mathematical estimation
    │   └── tracker.py       # Kalman-filter object tracking and velocity mapping
    └── ui/
        └── visualizer.py    # OpenCV HUD, zone coloring, and Dashboard rendering
```

---

## Future Enhancements
* **Depth Camera Integration:** Implementation of direct SDK support for hardware like Intel RealSense or Luxonis Oak-D for absolute millimeter-accurate depth measurement.
* **Pose Estimation:** Integrating YOLOv8-Pose to determine the vector of a person's gaze, avoiding alerts for individuals walking away from the user.
* **Mobile Deployment:** Exporting the PyTorch models to ONNX/CoreML formats for eventual execution on Android and iOS devices.
