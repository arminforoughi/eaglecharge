"""Synthetic video source for stage demos. Renders a sky-over-terrain scene
with a moving powerline so the detector and FSM have something to chew on."""
from __future__ import annotations

import math
import time

import cv2
import numpy as np


class MockVideo:
    def __init__(self, width: int = 960, height: int = 540):
        self.w = width
        self.h = height
        self.t0 = time.time()

    def read(self) -> np.ndarray:
        t = time.time() - self.t0
        img = np.zeros((self.h, self.w, 3), dtype=np.uint8)

        # Sky gradient
        for y in range(self.h):
            f = y / self.h
            img[y, :] = (int(180 - 90 * f), int(150 - 60 * f), int(110 - 40 * f))

        # Terrain band
        horizon = int(self.h * 0.62)
        cv2.rectangle(img, (0, horizon), (self.w, self.h), (45, 70, 55), -1)
        # Noise-y ground texture
        rng = np.random.default_rng(int(t * 2))
        noise = rng.integers(-12, 12, size=(self.h - horizon, self.w, 3), dtype=np.int16)
        img[horizon:, :, :] = np.clip(img[horizon:, :, :].astype(np.int16) + noise, 0, 255).astype(np.uint8)

        # Pylons
        for px in (int(self.w * 0.15), int(self.w * 0.85)):
            cv2.line(img, (px, horizon - 120), (px, horizon + 10), (30, 30, 30), 4)
            cv2.line(img, (px - 30, horizon - 110), (px + 30, horizon - 110), (30, 30, 30), 3)
            cv2.line(img, (px - 25, horizon - 90), (px + 25, horizon - 90), (30, 30, 30), 3)

        # Powerline — sags between pylons, drifts vertically with time.
        drift = int(8 * math.sin(t * 0.6))
        x_l, y_l = int(self.w * 0.15), horizon - 110 + drift
        x_r, y_r = int(self.w * 0.85), horizon - 110 + drift
        sag = 24
        pts = []
        for i in range(64):
            u = i / 63
            x = int(x_l + (x_r - x_l) * u)
            y = int(y_l + (y_r - y_l) * u + sag * math.sin(math.pi * u))
            pts.append((x, y))
        for i in range(len(pts) - 1):
            cv2.line(img, pts[i], pts[i + 1], (15, 15, 15), 2, cv2.LINE_AA)

        # Second parallel line below
        for i in range(len(pts) - 1):
            p1 = (pts[i][0], pts[i][1] + 22)
            p2 = (pts[i + 1][0], pts[i + 1][1] + 22)
            cv2.line(img, p1, p2, (15, 15, 15), 2, cv2.LINE_AA)

        # Subtle motion blur
        img = cv2.GaussianBlur(img, (3, 3), 0.6)
        return img
