"""
alert_coordinator.py
--------------------
The "brain" that decides WHEN to fire audio alerts based on
tracked persons' states.

Takes the list of active tracks and applies alert logic:
1. Critical proximity (<1m)
2. Close proximity (1–2.5m)
3. Someone actively approaching
4. Path blocked
5. Crowd density warning
6. All-clear when area is empty

This is intentionally separated from the alert engine so the
logic can be tested independently.

Author: CrowdAware AI Team
"""

import logging
from typing import List, Optional, Tuple

from ..detection.tracker import Track
from ..detection.distance import Zone
from .alert_engine import AlertEngine

logger = logging.getLogger(__name__)


class AlertCoordinator:
    """
    Evaluates track states and fires the appropriate audio alerts.

    Usage:
        coordinator = AlertCoordinator(config, alert_engine)
        coordinator.evaluate(tracks, frame_width)
    """

    def __init__(self, config: dict, alert_engine: AlertEngine):
        self.engine    = alert_engine
        audio_cfg      = config.get("audio", {})
        zone_cfg       = config.get("zones", {})

        # Load message templates
        msgs = audio_cfg.get("messages", {})
        self._msg_critical    = msgs.get("critical",     "Warning! Person {distance:.1f} metres ahead")
        self._msg_close       = msgs.get("close",        "Person very close, {distance:.1f} metres")
        self._msg_near        = msgs.get("near",         "Person nearby at {distance:.1f} metres")
        self._msg_approaching = msgs.get("approaching",  "Someone approaching from your {direction}")
        self._msg_blocked     = msgs.get("path_blocked", "Path ahead is blocked")
        self._msg_crowd       = msgs.get("crowd_alert",  "{count} people detected nearby")
        self._msg_clear       = msgs.get("clear",        "Path clear")

        self.crowd_threshold      = int(zone_cfg.get("crowd_threshold", 5))
        self.high_density_thresh  = int(zone_cfg.get("high_density_threshold", 10))

        # State for "clear" alert (only announce once when it becomes clear)
        self._was_occupied = False
        self._last_crowd_count = 0

        logger.info("[Coordinator] Alert coordinator initialized.")

    # ------------------------------------------------------------------
    # Main evaluation loop (call every frame)
    # ------------------------------------------------------------------

    def evaluate(self, tracks: List[Track], frame_width: int) -> List[str]:
        """
        Inspect all active tracks and fire alerts as needed.
        Returns list of alert message strings fired in this evaluation.

        Args:
            tracks      : List of confirmed Track objects
            frame_width : Frame width in pixels (for pan calculation)
        """
        fired_alerts: List[str] = []

        if not tracks:
            if self._was_occupied:
                self.engine.alert_info(self._msg_clear)
                fired_alerts.append(self._msg_clear)
                self._was_occupied = False
            return fired_alerts

        self._was_occupied = True

        # Sort tracks by distance — handle closest ones first
        sorted_tracks = sorted(tracks, key=lambda t: t.distance_m)

        # 1. Proximity alerts — up to 4.0m (Critical <1m, Close <2.5m, Near <4m)
        closest = sorted_tracks[0]
        if closest.distance_m <= 4.0:
            p_msg = self._proximity_alert(closest, frame_width)
            if p_msg:
                fired_alerts.append(p_msg)

        # 2. Approaching alerts — up to 4.0m
        for track in sorted_tracks:
            if track.is_approaching and track.distance_m <= 4.0:
                app_msg = self._approaching_alert(track, frame_width)
                if app_msg:
                    fired_alerts.append(app_msg)
                break   # Only one approaching alert per frame

        # 3. Path-blocked alert
        blockers = [t for t in tracks if t.is_path_blocker]
        if blockers:
            self.engine.alert_path_blocked(self._msg_blocked)
            fired_alerts.append(self._msg_blocked)

        # 4. Crowd density alert
        n = len(tracks)
        if n >= self.high_density_thresh:
            c_msg = f"High density crowd alert! {n} people around you"
            self.engine.alert_crowd(c_msg)
            fired_alerts.append(c_msg)
        elif n >= self.crowd_threshold:
            c_msg = self._msg_crowd.format(count=n)
            self.engine.alert_crowd(c_msg)
            fired_alerts.append(c_msg)

        self._last_crowd_count = n
        return fired_alerts

    # ------------------------------------------------------------------
    # Per-track alert helpers
    # ------------------------------------------------------------------

    def _proximity_alert(self, track: Track, frame_width: int) -> str:
        """Fire distance-based alert for person within 4 metres."""
        dist = track.distance_m
        cx   = track.center[0]
        pan  = self.engine.compute_pan(cx, frame_width)

        if dist < 1.0:
            msg = self._msg_critical.format(distance=dist)
            self.engine.alert_critical(msg, pan=pan)
            return msg
        elif dist <= 2.5:
            msg = self._msg_close.format(distance=dist)
            self.engine.alert_close(msg, pan=pan)
            return msg
        elif dist <= 4.0:
            msg = self._msg_near.format(distance=dist)
            self.engine.alert_near(msg, pan=pan)
            return msg
        return ""

    def _approaching_alert(self, track: Track, frame_width: int) -> str:
        """Fire alert when someone is actively moving toward the user."""
        direction = track.movement_direction or "ahead"
        msg = self._msg_approaching.format(direction=direction)
        cx  = track.center[0]
        pan = self.engine.compute_pan(cx, frame_width)
        self.engine.alert_approaching(msg, pan=pan)
        return msg

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def get_active_alert_summary(self, tracks: List[Track]) -> str:
        """
        Return a human-readable summary of the current alert state.
        Useful for the UI dashboard display.
        """
        if not tracks:
            return "✅ Area clear"

        lines = []
        closest = min(tracks, key=lambda t: t.distance_m)
        lines.append(f"Closest: {closest.distance_m:.1f}m [{closest.zone.value.upper()}]")

        blockers = [t for t in tracks if t.is_path_blocker]
        if blockers:
            lines.append("⚠ Path BLOCKED")

        approaching = [t for t in tracks if t.is_approaching]
        if approaching:
            lines.append(f"⚠ {len(approaching)} approaching")

        if len(tracks) >= self.crowd_threshold:
            lines.append(f"⚠ Crowd: {len(tracks)} people")

        return " | ".join(lines)
