"""Powerline detector. Canny + probabilistic Hough, clustered into a single
primary line. Confidence = supporting edge pixels / line length."""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import cv2
import numpy as np


@dataclass
class Detection:
    lines: list[tuple[int, int, int, int]] = field(default_factory=list)
    primary: tuple[int, int, int, int] | None = None
    primary_angle_deg: float = 0.0
    confidence: float = 0.0
    centroid_offset_px: tuple[int, int] = (0, 0)


class PowerlineDetector:
    def __init__(self, canny_lo: int = 60, canny_hi: int = 180):
        self.canny_lo = canny_lo
        self.canny_hi = canny_hi

    def detect(self, frame: np.ndarray) -> Detection:
        h, w = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 1.2)
        edges = cv2.Canny(gray, self.canny_lo, self.canny_hi, apertureSize=3)

        min_len = int(min(w, h) * 0.35)
        segs = cv2.HoughLinesP(
            edges,
            rho=1,
            theta=np.pi / 360,
            threshold=80,
            minLineLength=min_len,
            maxLineGap=20,
        )
        if segs is None:
            return Detection()

        segs = segs.reshape(-1, 4)
        # Cluster by angle; keep the dominant orientation cluster.
        angles = np.degrees(np.arctan2(segs[:, 3] - segs[:, 1], segs[:, 2] - segs[:, 0]))
        angles = ((angles + 90) % 180) - 90  # wrap to (-90, 90]
        # Histogram vote in 5 deg bins
        bins = np.linspace(-90, 90, 37)
        hist, edges_b = np.histogram(angles, bins=bins)
        peak = int(np.argmax(hist))
        peak_lo, peak_hi = edges_b[peak], edges_b[peak + 1]
        mask = (angles >= peak_lo) & (angles <= peak_hi)
        cluster = segs[mask]
        if len(cluster) == 0:
            return Detection()

        # Primary line: longest segment in dominant cluster, extended to frame edges.
        lengths = np.hypot(cluster[:, 2] - cluster[:, 0], cluster[:, 3] - cluster[:, 1])
        idx = int(np.argmax(lengths))
        x1, y1, x2, y2 = cluster[idx]
        primary_angle = float(np.degrees(math.atan2(y2 - y1, x2 - x1)))

        # Confidence = supporting edge density along primary
        line_mask = np.zeros_like(edges)
        cv2.line(line_mask, (x1, y1), (x2, y2), 255, 6)
        support = int(np.count_nonzero(cv2.bitwise_and(edges, line_mask)))
        denom = max(1, int(np.count_nonzero(line_mask)))
        conf = float(np.clip(support / denom * 2.0, 0.0, 1.0))

        # Centroid offset from frame center (used by FSM for alignment).
        mid_x = (x1 + x2) // 2
        mid_y = (y1 + y2) // 2
        offset = (int(mid_x - w // 2), int(mid_y - h // 2))

        return Detection(
            lines=[tuple(int(v) for v in s) for s in cluster.tolist()],
            primary=(int(x1), int(y1), int(x2), int(y2)),
            primary_angle_deg=primary_angle,
            confidence=conf,
            centroid_offset_px=offset,
        )

    @staticmethod
    def annotate(frame: np.ndarray, det: Detection) -> np.ndarray:
        out = frame.copy()
        h, w = out.shape[:2]
        for x1, y1, x2, y2 in det.lines:
            cv2.line(out, (x1, y1), (x2, y2), (60, 180, 60), 1, cv2.LINE_AA)
        if det.primary is not None:
            x1, y1, x2, y2 = det.primary
            cv2.line(out, (x1, y1), (x2, y2), (40, 220, 255), 3, cv2.LINE_AA)
        # Crosshair
        cv2.line(out, (w // 2 - 18, h // 2), (w // 2 + 18, h // 2), (200, 200, 200), 1)
        cv2.line(out, (w // 2, h // 2 - 18), (w // 2, h // 2 + 18), (200, 200, 200), 1)
        # HUD strip
        cv2.rectangle(out, (0, 0), (w, 28), (0, 0, 0), -1)
        txt = f"CONF {det.confidence*100:5.1f}%   ANG {det.primary_angle_deg:+6.1f} deg   OFF {det.centroid_offset_px[0]:+4d},{det.centroid_offset_px[1]:+4d}"
        cv2.putText(out, txt, (8, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (40, 220, 255), 1, cv2.LINE_AA)
        return out
