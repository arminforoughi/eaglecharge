"""Perch / charge state machine.

States:
  SEARCH    : no line, or low confidence
  TRACK     : line locked, drone closing distance
  ALIGN     : line stable & centered, ready to commit
  PERCH     : contact made, awaiting charge handshake
  CHARGE    : harvesting power, battery rising
  DEPART    : disengaging, returning to altitude
"""
from __future__ import annotations

import time
from dataclasses import dataclass

from .detector import Detection
from .telemetry import Telemetry


STATES = ("SEARCH", "TRACK", "ALIGN", "PERCH", "CHARGE", "DEPART")


@dataclass
class FSMSnapshot:
    state: str
    since_ms: int
    align_score: float
    charge_rate_w: float
    harvested_wh: float


class PerchFSM:
    def __init__(self):
        self.state = "SEARCH"
        self._entered = time.time()
        self._aligned_streak = 0
        self._perched_at: float | None = None
        self._charge_started: float | None = None
        self.harvested_wh = 0.0
        self.last_charge_w = 0.0

    def _goto(self, s: str):
        if s != self.state:
            self.state = s
            self._entered = time.time()
            if s == "PERCH":
                self._perched_at = time.time()
            if s == "CHARGE":
                self._charge_started = time.time()
            if s == "DEPART":
                self._perched_at = None
                self._charge_started = None

    def step(self, det: Detection, tel: Telemetry, frame_w: int, frame_h: int) -> FSMSnapshot:
        # Alignment score: confidence weighted by how centered + level the line is.
        align_score = 0.0
        if det.primary is not None:
            ox, oy = det.centroid_offset_px
            radial = (ox * ox + oy * oy) ** 0.5
            radial_norm = min(1.0, radial / (max(frame_w, frame_h) * 0.25))
            angle_pen = min(1.0, abs(det.primary_angle_deg) / 30.0)
            align_score = max(0.0, det.confidence * (1.0 - 0.6 * radial_norm) * (1.0 - 0.4 * angle_pen))

        s = self.state
        if s == "SEARCH":
            if det.confidence > 0.45:
                self._goto("TRACK")
        elif s == "TRACK":
            if det.confidence < 0.25:
                self._goto("SEARCH")
            elif align_score > 0.55:
                self._aligned_streak += 1
                if self._aligned_streak > 6:
                    self._goto("ALIGN")
            else:
                self._aligned_streak = max(0, self._aligned_streak - 1)
        elif s == "ALIGN":
            if align_score < 0.35:
                self._aligned_streak = 0
                self._goto("TRACK")
            elif time.time() - self._entered > 1.5:
                self._goto("PERCH")
        elif s == "PERCH":
            # Wait for "contact" — in mock, ~1.5 s; in live, the user wires
            # a contact sensor to flip a flag.
            if time.time() - self._entered > 1.5:
                self._goto("CHARGE")
        elif s == "CHARGE":
            # Track harvested energy. In mock, current is negative when charging.
            now = time.time()
            dt = max(0.0, now - getattr(self, "_last_step", now))
            self._last_step = now
            if tel.current_a < 0:
                p_w = abs(tel.current_a) * tel.voltage_v
                self.last_charge_w = p_w
                self.harvested_wh += p_w * (dt / 3600.0)
            else:
                self.last_charge_w = 0.0
            if tel.battery_pct >= 95.0:
                self._goto("DEPART")
        elif s == "DEPART":
            if time.time() - self._entered > 2.0:
                self._aligned_streak = 0
                self._goto("SEARCH")

        return FSMSnapshot(
            state=self.state,
            since_ms=int((time.time() - self._entered) * 1000),
            align_score=align_score,
            charge_rate_w=self.last_charge_w,
            harvested_wh=round(self.harvested_wh, 3),
        )

    def wants_charging(self) -> bool:
        return self.state in ("PERCH", "CHARGE")
