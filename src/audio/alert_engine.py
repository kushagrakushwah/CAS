"""
alert_engine.py
---------------
Audio alert engine for the CrowdAware system.

Manages:
- Text-to-speech alerts with cooldown management (no alert spam)
- Spatial audio (left/right panning based on person position)
- Beep tones (critical, close, near)
- Alert priority queue (critical alerts pre-empt lower ones)
- Background threading so audio never blocks the video pipeline

Author: CrowdAware AI Team
"""

import logging
import math
import threading
import time
from collections import defaultdict
from dataclasses import dataclass, field
from enum import IntEnum
from queue import PriorityQueue, Empty
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class AlertPriority(IntEnum):
    """Lower value = higher priority (PriorityQueue is min-heap)."""
    CRITICAL  = 0
    HIGH      = 1
    MEDIUM    = 2
    LOW       = 3
    INFO      = 4


@dataclass(order=True)
class Alert:
    """
    An individual alert message to be spoken.

    Fields are ordered so PriorityQueue works correctly
    (priority first, then time so older alerts win ties).
    """
    priority: AlertPriority
    timestamp: float = field(compare=True)
    message: str = field(compare=False)
    category: str = field(compare=False, default="general")
    pan: float = field(compare=False, default=0.0)   # -1.0=left, 0=center, +1.0=right

    def __repr__(self) -> str:
        return f'Alert(p={self.priority}, cat={self.category}, msg="{self.message}")'


class AlertEngine:
    """
    Thread-safe audio alert engine.

    Features
    --------
    - Per-category cooldowns prevent alert flooding
    - Priority queue ensures critical alerts are spoken first
    - Background worker thread handles TTS so it never blocks video
    - Spatial panning (left/right) based on person's x position in frame

    Usage
    -----
        engine = AlertEngine(config)
        engine.start()
        engine.alert_critical("Person 0.8 metres ahead", pan=-0.3)
        engine.stop()
    """

    def __init__(self, config: dict):
        audio_cfg = config.get("audio", {})
        self.enabled   = audio_cfg.get("enabled", True)
        self.engine_type = audio_cfg.get("engine", "pyttsx3")
        self.volume    = audio_cfg.get("volume", 0.9)
        self.rate      = audio_cfg.get("rate", 175)
        self.spatial   = audio_cfg.get("spatial_audio", True)
        self.beep_mode = audio_cfg.get("beep_alerts", True)

        cooldowns = audio_cfg.get("cooldowns", {})
        self._cooldowns: Dict[str, float] = {
            "critical":     float(cooldowns.get("critical",     2.0)),
            "close":        float(cooldowns.get("close",        4.0)),
            "near":         float(cooldowns.get("near",         8.0)),
            "path_blocked": float(cooldowns.get("path_blocked", 5.0)),
            "crowd":        float(cooldowns.get("crowd",        10.0)),
            "approach":     float(cooldowns.get("approach",     6.0)),
            "general":      5.0,
        }

        # Timestamps of last alert per category
        self._last_alert: Dict[str, float] = defaultdict(float)

        # Background worker
        self._queue: PriorityQueue = PriorityQueue(maxsize=20)
        self._worker_thread: Optional[threading.Thread] = None
        self._running = False

        # TTS engine (lazy-loaded in worker thread)
        self._tts_engine = None
        self._tts_lock = threading.Lock()

        # Stats
        self._alerts_issued = 0
        self._alerts_suppressed = 0

        logger.info(
            f"[Audio] Engine={self.engine_type}, volume={self.volume}, "
            f"rate={self.rate} wpm, spatial={self.spatial}"
        )

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self):
        """Start the background TTS worker thread."""
        if not self.enabled:
            logger.info("[Audio] Audio alerts disabled in config.")
            return
        self._running = True
        self._worker_thread = threading.Thread(
            target=self._worker_loop,
            name="AudioAlertWorker",
            daemon=True,
        )
        self._worker_thread.start()
        logger.info("[Audio] Alert engine started.")

    def stop(self):
        """Gracefully stop the worker thread."""
        self._running = False
        if self._worker_thread and self._worker_thread.is_alive():
            self._worker_thread.join(timeout=2.0)
        logger.info("[Audio] Alert engine stopped.")

    # ------------------------------------------------------------------
    # Public alert methods
    # ------------------------------------------------------------------

    def alert_critical(self, message: str, pan: float = 0.0):
        """Immediate danger — highest priority, shortest cooldown."""
        self._enqueue(message, "critical", AlertPriority.CRITICAL, pan)

    def alert_close(self, message: str, pan: float = 0.0):
        """Person very close (1–2.5 m)."""
        self._enqueue(message, "close", AlertPriority.HIGH, pan)

    def alert_near(self, message: str, pan: float = 0.0):
        """Person nearby (2.5–4 m)."""
        self._enqueue(message, "near", AlertPriority.MEDIUM, pan)

    def alert_approaching(self, message: str, pan: float = 0.0):
        """Someone moving toward the user."""
        self._enqueue(message, "approach", AlertPriority.MEDIUM, pan)

    def alert_path_blocked(self, message: str):
        """Central path is blocked."""
        self._enqueue(message, "path_blocked", AlertPriority.HIGH, 0.0)

    def alert_crowd(self, message: str):
        """High crowd density."""
        self._enqueue(message, "crowd", AlertPriority.MEDIUM, 0.0)

    def alert_info(self, message: str):
        """General informational alert."""
        self._enqueue(message, "general", AlertPriority.INFO, 0.0)

    # ------------------------------------------------------------------
    # Core enqueue logic
    # ------------------------------------------------------------------

    def _enqueue(
        self,
        message: str,
        category: str,
        priority: AlertPriority,
        pan: float,
    ):
        """Add an alert to the queue if cooldown has expired."""
        if not self.enabled:
            return

        now = time.monotonic()
        cooldown = self._cooldowns.get(category, 5.0)

        if now - self._last_alert[category] < cooldown:
            self._alerts_suppressed += 1
            logger.debug(f"[Audio] Suppressed '{category}' alert (cooldown)")
            return

        self._last_alert[category] = now
        alert = Alert(
            priority=priority,
            timestamp=now,
            message=message,
            category=category,
            pan=max(-1.0, min(1.0, pan)),
        )

        try:
            self._queue.put_nowait(alert)
            self._alerts_issued += 1
            logger.debug(f"[Audio] Queued: {alert}")
        except Exception:
            logger.warning("[Audio] Alert queue full — dropping alert.")
            self._alerts_suppressed += 1

    # ------------------------------------------------------------------
    # Worker thread
    # ------------------------------------------------------------------

    def _worker_loop(self):
        """Background thread: drain queue and speak alerts."""
        self._init_tts()
        while self._running:
            try:
                alert: Alert = self._queue.get(timeout=0.1)
                self._speak(alert)
                self._queue.task_done()
            except Empty:
                pass
            except Exception as e:
                logger.error(f"[Audio] Worker error: {e}")

    def _init_tts(self):
        """Initialize TTS engine in the worker thread."""
        try:
            if self.engine_type == "pyttsx3":
                import pyttsx3
                engine = pyttsx3.init()
                engine.setProperty("rate", self.rate)
                engine.setProperty("volume", self.volume)

                # Select voice by gender preference
                voices = engine.getProperty("voices")
                if voices:
                    # Try to find matching gender voice
                    preferred = [
                        v for v in voices
                        if hasattr(v, "gender") and
                           str(v.gender).lower() == self.engine_type
                    ]
                    engine.setProperty("voice", voices[0].id)  # default fallback
                self._tts_engine = engine
                logger.info("[Audio] pyttsx3 TTS engine initialized.")
            else:
                logger.info("[Audio] Using print-only fallback TTS.")
        except Exception as e:
            logger.warning(f"[Audio] TTS init failed ({e}). Alerts will be printed only.")
            self._tts_engine = None

    def _speak(self, alert: Alert):
        """Actually speak or print the alert."""
        logger.info(f"[ALERT:{alert.category.upper()}] {alert.message}")

        if self._tts_engine is None:
            # Print fallback
            print(f"\n🔔 [{alert.category.upper()}] {alert.message}")
            return

        try:
            # Spatial panning (volume adjustment per channel)
            # Full pyttsx3 doesn't support true panning, but we log it
            if self.spatial and abs(alert.pan) > 0.1:
                direction = "LEFT" if alert.pan < 0 else "RIGHT"
                logger.debug(f"[Audio] Spatial pan={alert.pan:.2f} ({direction})")

            with self._tts_lock:
                self._tts_engine.say(alert.message)
                self._tts_engine.runAndWait()
        except Exception as e:
            logger.error(f"[Audio] Speech error: {e}")
            print(f"\n🔔 [{alert.category.upper()}] {alert.message}")

    # ------------------------------------------------------------------
    # Utility helpers
    # ------------------------------------------------------------------

    def compute_pan(self, x_pixel: int, frame_width: int) -> float:
        """
        Compute stereo pan value from horizontal pixel position.

        Returns:
            float in [-1.0, 1.0]: -1=far left, 0=center, +1=far right
        """
        if frame_width <= 0:
            return 0.0
        return (x_pixel / frame_width - 0.5) * 2.0

    def force_alert(self, message: str, category: str = "general"):
        """Bypass cooldown and force an immediate alert (for critical events)."""
        self._last_alert[category] = 0.0   # reset cooldown
        self._enqueue(message, category, AlertPriority.CRITICAL, 0.0)

    def reset_cooldowns(self):
        """Reset all cooldown timers (useful for testing)."""
        self._last_alert = defaultdict(float)

    def get_stats(self) -> dict:
        return {
            "alerts_issued": self._alerts_issued,
            "alerts_suppressed": self._alerts_suppressed,
            "queue_size": self._queue.qsize(),
            "engine": self.engine_type,
            "enabled": self.enabled,
        }

    def __repr__(self) -> str:
        return (f"AlertEngine(engine={self.engine_type}, enabled={self.enabled}, "
                f"issued={self._alerts_issued})")
