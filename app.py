"""
app.py
------
Streamlit frontend for CrowdAware AI.

Run with:
    .\\venv\\Scripts\\streamlit run app.py

Features
--------
- Upload images and run detection → returns annotated image.
- Webcam live feed with real-time detection.
- Signal panel: zone status, crowd density, path-blocked indicator.
- Audio alerts via pyttsx3 (plays on your PC speakers).
- Stats bar: confidence, detection count, inference time.
"""

import sys
import os
import time
import logging
from pathlib import Path
from io import BytesIO

import streamlit as st
import cv2
import numpy as np
from PIL import Image
import yaml

# ---------------------------------------------------------------------------
# Path setup — ensure project root is importable
# ---------------------------------------------------------------------------
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

# ---------------------------------------------------------------------------
# Logging — suppress noisy library logs in the UI
# ---------------------------------------------------------------------------
logging.basicConfig(level=logging.WARNING)
logging.getLogger("torch").setLevel(logging.ERROR)
logging.getLogger("torchvision").setLevel(logging.ERROR)

# ---------------------------------------------------------------------------
# Load config
# ---------------------------------------------------------------------------
CONFIG_PATH = ROOT / "config" / "settings.yaml"

@st.cache_resource
def load_config() -> dict:
    with open(CONFIG_PATH, "r") as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Load backend subsystems (cached — loaded once on first call)
# ---------------------------------------------------------------------------
@st.cache_resource
def load_subsystems():
    """Load detector, estimator, tracker, visualizer once and reuse."""
    config = load_config()

    from src.detection.mobilenet_detector import MobileNetPersonDetector
    from src.detection.distance import DistanceEstimator
    from src.detection.tracker import MultiPersonTracker
    from src.ui.visualizer import Visualizer
    from src.audio.alert_engine import AlertEngine
    from src.audio.alert_coordinator import AlertCoordinator

    detector    = MobileNetPersonDetector(config)
    estimator   = DistanceEstimator(config)
    tracker     = MultiPersonTracker(config)
    visualizer  = Visualizer(config)
    alert_eng   = AlertEngine(config)
    coordinator = AlertCoordinator(config, alert_eng)

    # Do NOT call alert_eng.start() here — pyttsx3 crashes when started
    # inside Streamlit's cache thread. It is started lazily in process_frame()
    # only when the user enables audio.
    return detector, estimator, tracker, visualizer, alert_eng, coordinator, config


# ---------------------------------------------------------------------------
# Core processing function — one frame in, annotated frame out
# ---------------------------------------------------------------------------
def process_frame(frame_bgr: np.ndarray, enable_audio: bool = False):
    """
    Run detection → distance → track → visualize on a single frame.

    Returns
    -------
    annotated   : BGR numpy array with overlays
    tracks      : list of Track objects
    detections  : list of Detection objects
    alert_summary : str summary for the signal panel
    stats       : dict with timing info
    """
    detector, estimator, tracker, visualizer, alert_eng, coordinator, config = load_subsystems()

    h, w = frame_bgr.shape[:2]
    t0 = time.perf_counter()

    # 1. Detect
    detections = detector.detect(frame_bgr)

    # 2. Estimate distances
    estimates = [estimator.estimate(det, h) for det in detections]

    # 3. Track
    tracks = tracker.update(detections, estimates, (h, w))

    t1 = time.perf_counter()
    elapsed_ms = (t1 - t0) * 1000.0

    # 4. Audio alerts (only if user enabled them)
    if enable_audio:
        # Start the audio engine lazily on first use (avoids crash at Streamlit boot)
        if not st.session_state.get("audio_started", False):
            try:
                alert_eng.start()
                st.session_state["audio_started"] = True
            except Exception:
                pass  # pyttsx3 not available — silently skip
        try:
            coordinator.evaluate(tracks, w)
        except Exception:
            pass  # Audio error — don't crash the detection loop

    # 5. Alert summary text
    alert_summary = coordinator.get_active_alert_summary(tracks)

    # 6. Draw overlays
    det_fps = detector.current_fps
    annotated = visualizer.draw(frame_bgr, tracks, fps=0.0,
                                alert_summary=alert_summary, detector_fps=det_fps)

    stats = {
        "detections": len(detections),
        "tracks": len(tracks),
        "inference_ms": round(elapsed_ms, 1),
        "det_fps": round(det_fps, 1),
        "avg_conf": round(
            float(np.mean([d.confidence for d in detections])) if detections else 0.0, 3
        ),
    }

    return annotated, tracks, detections, alert_summary, stats


# ---------------------------------------------------------------------------
# Signal panel renderer
# ---------------------------------------------------------------------------
def render_signal_panel(tracks, alert_summary: str, stats: dict):
    """Render the live signal / status cards in the sidebar."""
    st.sidebar.markdown("---")
    st.sidebar.markdown("### Signal Panel")

    n = stats["detections"]
    inf_ms = stats["inference_ms"]

    # --- Crowd density ---
    if n == 0:
        density_label, density_color = "CLEAR", "green"
    elif n < 5:
        density_label, density_color = "LOW", "green"
    elif n < 10:
        density_label, density_color = "MEDIUM", "orange"
    else:
        density_label, density_color = "HIGH", "red"

    st.sidebar.markdown(
        f"<div style='background:{density_color};padding:8px 12px;border-radius:6px;"
        f"color:white;font-weight:bold;text-align:center;margin-bottom:6px;'>"
        f"CROWD: {density_label} ({n} people)</div>",
        unsafe_allow_html=True,
    )

    # --- Zone status (closest person) ---
    if tracks:
        from src.detection.distance import Zone
        closest = min(tracks, key=lambda t: t.distance_m)
        zone = closest.zone
        zone_colors = {
            Zone.CRITICAL: "red",
            Zone.CLOSE: "orange",
            Zone.NEAR: "#ccaa00",
            Zone.MEDIUM: "green",
            Zone.FAR: "#336699",
        }
        zone_col = zone_colors.get(zone, "gray")
        zone_name = zone.value.upper() if zone else "FAR"
        st.sidebar.markdown(
            f"<div style='background:{zone_col};padding:8px 12px;border-radius:6px;"
            f"color:white;font-weight:bold;text-align:center;margin-bottom:6px;'>"
            f"CLOSEST: {zone_name} ({closest.distance_m:.1f} m)</div>",
            unsafe_allow_html=True,
        )
    else:
        st.sidebar.markdown(
            "<div style='background:green;padding:8px 12px;border-radius:6px;"
            "color:white;font-weight:bold;text-align:center;margin-bottom:6px;'>"
            "ZONE: ALL CLEAR</div>",
            unsafe_allow_html=True,
        )

    # --- Path blocked? ---
    blockers = [t for t in tracks if t.is_path_blocker]
    path_status = "BLOCKED" if blockers else "CLEAR"
    path_color = "red" if blockers else "green"
    st.sidebar.markdown(
        f"<div style='background:{path_color};padding:8px 12px;border-radius:6px;"
        f"color:white;font-weight:bold;text-align:center;margin-bottom:6px;'>"
        f"PATH: {path_status}</div>",
        unsafe_allow_html=True,
    )

    # --- Approaching alert ---
    approaching = [t for t in tracks if t.is_approaching]
    if approaching:
        st.sidebar.markdown(
            f"<div style='background:orange;padding:8px 12px;border-radius:6px;"
            f"color:white;font-weight:bold;text-align:center;margin-bottom:6px;'>"
            f"APPROACHING: {len(approaching)} person(s)</div>",
            unsafe_allow_html=True,
        )

    # --- Inference stats ---
    st.sidebar.markdown("---")
    st.sidebar.markdown("### Stats")
    st.sidebar.metric("Inference time", f"{inf_ms} ms")
    st.sidebar.metric("Detection FPS", f"{stats['det_fps']}")
    st.sidebar.metric("Avg confidence", f"{stats['avg_conf']:.1%}")
    st.sidebar.metric("Tracks active", stats["tracks"])

    # --- Alert summary text ---
    if alert_summary:
        st.sidebar.markdown("---")
        st.sidebar.markdown("### Alert Summary")
        st.sidebar.info(alert_summary)


# ---------------------------------------------------------------------------
# Convert BGR numpy → PIL Image for Streamlit display
# ---------------------------------------------------------------------------
def bgr_to_pil(bgr: np.ndarray) -> Image.Image:
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)


def pil_to_bgr(img: Image.Image) -> np.ndarray:
    rgb = np.array(img.convert("RGB"))
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


# ---------------------------------------------------------------------------
# Page: Image Upload
# ---------------------------------------------------------------------------
def page_image_upload():
    st.header("Image Detection")
    st.write("Upload an image. The model will detect people and show bounding boxes, distances, and zone labels.")

    col1, col2 = st.columns([2, 1])
    with col1:
        uploaded = st.file_uploader(
            "Choose an image",
            type=["jpg", "jpeg", "png", "bmp", "webp"],
            help="Supports JPG, PNG, BMP, WEBP",
        )

    with col2:
        enable_audio = st.checkbox("Enable voice alerts", value=False,
                                   help="Plays audio alerts on your PC speakers via pyttsx3")
        st.info("Audio only works if pyttsx3 is set up correctly on this machine.")

    # --- Quick test image picker from test_images folder ---
    test_dir = ROOT / "test_images"
    test_files = sorted(test_dir.glob("*")) if test_dir.exists() else []
    image_files = [f for f in test_files if f.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}]

    selected_test = None
    if image_files:
        st.markdown("**Or pick a test image:**")
        cols = st.columns(min(5, len(image_files)))
        for i, img_path in enumerate(image_files[:10]):
            with cols[i % 5]:
                thumb = Image.open(img_path).convert("RGB")
                thumb.thumbnail((120, 120))
                if st.button(img_path.name, key=f"test_{i}"):
                    selected_test = img_path

    # --- Determine which image to process ---
    source_img = None
    if selected_test is not None:
        source_img = Image.open(selected_test).convert("RGB")
        st.success(f"Using test image: {selected_test.name}")
    elif uploaded is not None:
        source_img = Image.open(uploaded).convert("RGB")

    if source_img is not None:
        frame_bgr = pil_to_bgr(source_img)

        with st.spinner("Running detection..."):
            annotated, tracks, detections, alert_summary, stats = process_frame(
                frame_bgr, enable_audio=enable_audio
            )

        # --- Show results ---
        result_col, orig_col = st.columns(2)
        with orig_col:
            st.subheader("Original")
            st.image(source_img, use_container_width=True)
        with result_col:
            st.subheader(f"Detected ({stats['detections']} people)")
            st.image(bgr_to_pil(annotated), use_container_width=True)

        # --- Detection table ---
        if detections:
            st.markdown("### Detection Details")
            rows = []
            for i, (det, track) in enumerate(zip(detections, tracks[:len(detections)])):
                rows.append({
                    "Person #": i + 1,
                    "Confidence": f"{det.confidence:.1%}",
                    "Distance (m)": f"{track.distance_m:.2f}",
                    "Zone": track.zone.value.upper() if track.zone else "FAR",
                    "Path Blocked": "YES" if track.is_path_blocker else "no",
                    "Bbox [x1,y1,x2,y2]": str(det.bbox),
                })
            st.dataframe(rows, use_container_width=True)
        else:
            st.warning("No people detected. Try a different image or lower the confidence threshold.")

        # --- Download annotated image ---
        result_pil = bgr_to_pil(annotated)
        buf = BytesIO()
        result_pil.save(buf, format="PNG")
        st.download_button(
            label="Download annotated image",
            data=buf.getvalue(),
            file_name="crowdaware_result.png",
            mime="image/png",
        )

        # --- Signal panel ---
        render_signal_panel(tracks, alert_summary, stats)

    else:
        st.info("Upload an image or pick a test image above to start.")


# ---------------------------------------------------------------------------
# Page: Live Webcam
# ---------------------------------------------------------------------------
def page_webcam():
    st.header("Live Webcam Detection")
    st.write("Uses your webcam for real-time detection. Click Start to begin, Stop to end.")

    col1, col2 = st.columns([2, 1])
    with col1:
        cam_id = st.number_input("Camera device ID", min_value=0, max_value=10, value=0, step=1)
    with col2:
        enable_audio = st.checkbox("Enable voice alerts", value=False, key="webcam_audio")

    start = st.button("Start Webcam", type="primary")
    stop  = st.button("Stop Webcam")

    FRAME_PLACEHOLDER = st.empty()
    stats_placeholder = st.empty()

    if "webcam_running" not in st.session_state:
        st.session_state.webcam_running = False

    if start:
        st.session_state.webcam_running = True

    if stop:
        st.session_state.webcam_running = False

    if st.session_state.webcam_running:
        cap = cv2.VideoCapture(int(cam_id))
        if not cap.isOpened():
            st.error(f"Cannot open camera {cam_id}. Check device ID.")
            st.session_state.webcam_running = False
            return

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        frame_count = 0
        MAX_FRAMES = 300  # Auto-stop after 300 frames (~10s at 30fps) to avoid infinite loop

        while st.session_state.webcam_running and frame_count < MAX_FRAMES:
            ret, frame = cap.read()
            if not ret:
                st.warning("Frame read failed.")
                break

            annotated, tracks, detections, alert_summary, stats = process_frame(
                frame, enable_audio=enable_audio
            )

            FRAME_PLACEHOLDER.image(bgr_to_pil(annotated), caption="Live Feed", use_container_width=True)

            # Compact stats row
            stats_placeholder.markdown(
                f"**Detections:** {stats['detections']} | "
                f"**Inference:** {stats['inference_ms']} ms | "
                f"**FPS:** {stats['det_fps']} | "
                f"**Avg Conf:** {stats['avg_conf']:.1%}"
            )

            render_signal_panel(tracks, alert_summary, stats)
            frame_count += 1

        cap.release()
        st.session_state.webcam_running = False
        st.info("Webcam stopped.")


# ---------------------------------------------------------------------------
# Page: Batch Folder
# ---------------------------------------------------------------------------
def page_batch():
    st.header("Batch Folder Scan")
    st.write("Point to a folder and run detection on every image inside it.")

    folder_path = st.text_input("Folder path", value=str(ROOT / "test_images"))
    enable_audio = st.checkbox("Enable voice alerts", value=False, key="batch_audio")

    if st.button("Run Batch Detection", type="primary"):
        folder = Path(folder_path)
        if not folder.exists():
            st.error(f"Folder not found: {folder}")
            return

        image_files = [f for f in sorted(folder.iterdir())
                       if f.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}]
        if not image_files:
            st.warning("No images found in that folder.")
            return

        st.write(f"Found {len(image_files)} images. Processing...")
        progress = st.progress(0)
        summary_rows = []

        for i, img_path in enumerate(image_files):
            pil_img = Image.open(img_path).convert("RGB")
            frame_bgr = pil_to_bgr(pil_img)

            annotated, tracks, detections, alert_summary, stats = process_frame(
                frame_bgr, enable_audio=enable_audio
            )

            summary_rows.append({
                "File": img_path.name,
                "People detected": stats["detections"],
                "Avg confidence": f"{stats['avg_conf']:.1%}",
                "Inference (ms)": stats["inference_ms"],
                "Alert": alert_summary,
            })

            # Show annotated thumbnail
            thumb = bgr_to_pil(annotated)
            thumb.thumbnail((320, 240))
            st.image(thumb, caption=f"{img_path.name} — {stats['detections']} people", width=320)

            progress.progress((i + 1) / len(image_files))

        st.markdown("### Batch Summary")
        st.dataframe(summary_rows, use_container_width=True)


# ---------------------------------------------------------------------------
# Page: Model Info
# ---------------------------------------------------------------------------
def page_model_info():
    st.header("Model Information")

    config = load_config()
    det_cfg = config.get("detection", {})
    ckpt = det_cfg.get("model", "models/best_person_detector.pth")
    ckpt_path = ROOT / ckpt

    st.markdown("### Architecture")
    st.markdown("""
| Property | Value |
|---|---|
| Model | MobileNetV3-Large + SSDLite320 |
| Framework | PyTorch / Torchvision (Zero-YOLO) |
| Input size | 320 × 320 |
| Classes | 2 (Background, Person) |
| Training | Fine-tuned on Penn-Fudan Pedestrian Dataset |
| Training method | Transfer Learning (ImageNet backbone) |
""")

    st.markdown("### Config")
    is_coco = str(ckpt).lower() in ("coco", "default")
    ckpt_status = "Pretrained COCO (Torchvision)" if is_coco else ("Found" if ckpt_path.exists() else "Not Found")
    ckpt_size_str = "45 MB (COCO weights)" if is_coco else (f"{round(ckpt_path.stat().st_size / 1e6, 1)} MB" if ckpt_path.exists() else "N/A")
    st.markdown(f"""
| Setting | Value |
|---|---|
| Engine | `{det_cfg.get('engine')}` |
| Checkpoint | `{ckpt}` |
| Checkpoint status | `{ckpt_status}` |
| Checkpoint size | `{ckpt_size_str}` |
| Confidence threshold | `{det_cfg.get('confidence_threshold')}` |
| NMS threshold | `{det_cfg.get('nms_threshold')}` |
| Device | `{det_cfg.get('device')}` |
""")

    st.markdown("### Distance Zones")
    dist_cfg = config.get("distance", {}).get("zones", {})
    st.markdown(f"""
| Zone | Threshold |
|---|---|
| CRITICAL (Red) | < {dist_cfg.get('critical', 1.0)} m |
| CLOSE (Orange) | < {dist_cfg.get('close', 2.5)} m |
| NEAR (Yellow) | < {dist_cfg.get('near', 4.0)} m |
| MEDIUM (Green) | < {dist_cfg.get('medium', 7.0)} m |
| FAR (Blue) | > 7 m |
""")

    st.markdown("### Audio Settings")
    audio_cfg = config.get("audio", {})
    st.markdown(f"""
| Setting | Value |
|---|---|
| Engine | `{audio_cfg.get('engine')}` |
| Volume | `{audio_cfg.get('volume')}` |
| Rate (WPM) | `{audio_cfg.get('rate')}` |
| Spatial audio | `{audio_cfg.get('spatial_audio')}` |
""")


# ---------------------------------------------------------------------------
# Main app layout
# ---------------------------------------------------------------------------
def main():
    st.set_page_config(
        page_title="CrowdAware AI",
        page_icon="👁",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # --- Sidebar nav ---
    st.sidebar.title("CrowdAware AI")
    st.sidebar.markdown("Person detection with distance estimation and audio alerts.")
    st.sidebar.markdown("---")

    page = st.sidebar.radio(
        "Navigate",
        ["Image Upload", "Live Webcam", "Batch Scan", "Model Info"],
    )

    # --- Load subsystems on startup (shows spinner once) ---
    with st.spinner("Loading model (first run takes a few seconds)..."):
        try:
            load_subsystems()
            st.sidebar.success("Model loaded")
        except Exception as e:
            st.sidebar.error(f"Model load error: {e}")
            st.error(f"Failed to load model: {e}")
            st.stop()

    # --- Route to page ---
    if page == "Image Upload":
        page_image_upload()
    elif page == "Live Webcam":
        page_webcam()
    elif page == "Batch Scan":
        page_batch()
    elif page == "Model Info":
        page_model_info()


if __name__ == "__main__":
    main()
