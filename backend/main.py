"""EAGLE — drone perch & charge ground station.

Run:
    pip install -r requirements.txt
    python -m backend.main

Open http://localhost:8000

Env:
    EAGLE_VIDEO     path to video file or camera index (default: mock)
    EAGLE_MAVLINK   pymavlink connection string, e.g. udpin:0.0.0.0:14550
                    (default: mock telemetry)
"""
from __future__ import annotations

import asyncio
import json
import os
import time
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, StreamingResponse

from .detector import Detection, PowerlineDetector
from .mock_video import MockVideo
from .perch_fsm import PerchFSM
from .telemetry import make_source


app = FastAPI(title="EAGLE")
FRONTEND = Path(__file__).resolve().parent.parent / "frontend"


class VideoSource:
    def __init__(self):
        src = os.environ.get("EAGLE_VIDEO")
        self.mode = "MOCK"
        self._cap = None
        self._mock = None
        if src:
            cap = cv2.VideoCapture(int(src) if src.isdigit() else src)
            if cap.isOpened():
                self._cap = cap
                self.mode = "LIVE"
            else:
                print(f"[video] could not open {src}; using mock")
        if self._cap is None:
            self._mock = MockVideo()

    def read(self):
        if self._cap is not None:
            ok, frame = self._cap.read()
            if not ok:
                # loop file sources
                self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ok, frame = self._cap.read()
            return frame if ok else None
        return self._mock.read()


# Shared singletons
video = VideoSource()
detector = PowerlineDetector()
telemetry = make_source()
fsm = PerchFSM()

# Latest detection — written by MJPEG loop, read by WS pump.
_latest_det: Detection = Detection()
_latest_frame_size: tuple[int, int] = (960, 540)


def _gen_mjpeg():
    global _latest_det, _latest_frame_size
    boundary = b"--frame"
    while True:
        frame = video.read()
        if frame is None:
            time.sleep(0.05)
            continue
        det = detector.detect(frame)
        _latest_det = det
        _latest_frame_size = (frame.shape[1], frame.shape[0])
        annotated = PowerlineDetector.annotate(frame, det)
        ok, buf = cv2.imencode(".jpg", annotated, [int(cv2.IMWRITE_JPEG_QUALITY), 78])
        if not ok:
            continue
        chunk = (boundary + b"\r\n"
                 b"Content-Type: image/jpeg\r\n"
                 b"Content-Length: " + str(len(buf)).encode() + b"\r\n\r\n"
                 + buf.tobytes() + b"\r\n")
        yield chunk
        time.sleep(1 / 25.0)


@app.get("/video.mjpg")
def video_stream():
    return StreamingResponse(_gen_mjpeg(),
                             media_type="multipart/x-mixed-replace; boundary=frame")


@app.websocket("/ws")
async def telemetry_ws(ws: WebSocket):
    await ws.accept()
    try:
        while True:
            tel = telemetry.snapshot()
            w, h = _latest_frame_size
            snap = fsm.step(_latest_det, tel, w, h)
            telemetry.set_charging(fsm.wants_charging() and snap.state == "CHARGE")
            payload = {
                "ts": time.time(),
                "video_mode": video.mode,
                "telemetry": tel.as_dict(),
                "detection": {
                    "confidence": _latest_det.confidence,
                    "angle_deg": _latest_det.primary_angle_deg,
                    "offset_px": list(_latest_det.centroid_offset_px),
                    "n_lines": len(_latest_det.lines),
                },
                "fsm": {
                    "state": snap.state,
                    "since_ms": snap.since_ms,
                    "align_score": snap.align_score,
                    "charge_rate_w": snap.charge_rate_w,
                    "harvested_wh": snap.harvested_wh,
                },
            }
            await ws.send_text(json.dumps(payload))
            await asyncio.sleep(0.1)
    except WebSocketDisconnect:
        return


@app.get("/")
def index():
    return FileResponse(FRONTEND / "index.html")


@app.get("/{name}")
def static_file(name: str):
    p = FRONTEND / name
    if p.is_file():
        return FileResponse(p)
    return FileResponse(FRONTEND / "index.html")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.main:app", host="0.0.0.0", port=8000, reload=False)
