"""Telemetry source. Real mode talks to the autopilot via pymavlink;
mock mode generates a plausible flight + perch + charge cycle so the
dashboard demos without hardware."""
from __future__ import annotations

import math
import os
import threading
import time
from dataclasses import asdict, dataclass


@dataclass
class Telemetry:
    ts: float = 0.0
    mode: str = "MOCK"
    armed: bool = False
    battery_pct: float = 0.0
    voltage_v: float = 0.0
    current_a: float = 0.0
    alt_m: float = 0.0
    lat: float = 0.0
    lon: float = 0.0
    roll_deg: float = 0.0
    pitch_deg: float = 0.0
    yaw_deg: float = 0.0
    link: str = "down"

    def as_dict(self) -> dict:
        return asdict(self)


class MockTelemetry:
    """Drives a battery curve and attitude that the FSM can react to."""

    def __init__(self):
        self._origin = (38.8895, -77.0353)
        self._t = Telemetry(mode="MOCK", armed=True, battery_pct=42.0,
                            voltage_v=22.6, current_a=9.4, alt_m=18.0,
                            lat=self._origin[0], lon=self._origin[1], link="up")
        self._lock = threading.Lock()
        self._charging = False
        self._t0 = time.time()
        threading.Thread(target=self._run, daemon=True).start()

    def set_origin(self, lat: float, lon: float):
        with self._lock:
            self._origin = (float(lat), float(lon))

    def _run(self):
        while True:
            with self._lock:
                t = self._t
                t.ts = time.time()
                phase = time.time() - self._t0
                if self._charging:
                    t.battery_pct = min(100.0, t.battery_pct + 0.25)
                    t.current_a = -6.5  # negative = charging
                    t.voltage_v = 24.6
                else:
                    t.battery_pct = max(0.0, t.battery_pct - 0.04)
                    t.current_a = 9.0 + 1.5 * math.sin(phase * 0.4)
                    t.voltage_v = 22.0 + 0.3 * math.sin(phase * 0.2)
                t.roll_deg = 4.0 * math.sin(phase * 0.7)
                t.pitch_deg = 2.5 * math.sin(phase * 0.5 + 1.0)
                # Slow circular drift around the operator-set origin.
                w = phase * 0.05
                lat0, lon0 = self._origin
                t.lat = lat0 + 0.0009 * math.sin(w)
                t.lon = lon0 + 0.0011 * math.cos(w)
                t.yaw_deg = (math.degrees(w) + 90) % 360 - 180
                t.alt_m = 18.0 + 0.6 * math.sin(phase * 0.3)
            time.sleep(0.1)

    def set_charging(self, charging: bool):
        with self._lock:
            self._charging = charging

    def snapshot(self) -> Telemetry:
        with self._lock:
            return Telemetry(**asdict(self._t))


class MavlinkTelemetry:
    """Thin pymavlink wrapper. Falls back to mock if connection fails."""

    def __init__(self, conn_str: str):
        from pymavlink import mavutil  # local import — heavy
        self._t = Telemetry(mode="LIVE", link="connecting")
        self._lock = threading.Lock()
        self._charging = False
        self._mav = mavutil.mavlink_connection(conn_str, autoreconnect=True)
        threading.Thread(target=self._run, daemon=True).start()

    def _run(self):
        # Wait for heartbeat
        try:
            self._mav.wait_heartbeat(timeout=10)
            with self._lock:
                self._t.link = "up"
        except Exception:
            with self._lock:
                self._t.link = "down"

        while True:
            msg = self._mav.recv_match(blocking=True, timeout=1.0)
            if msg is None:
                continue
            mtype = msg.get_type()
            with self._lock:
                t = self._t
                t.ts = time.time()
                if mtype == "SYS_STATUS":
                    t.battery_pct = float(getattr(msg, "battery_remaining", -1))
                    t.voltage_v = getattr(msg, "voltage_battery", 0) / 1000.0
                    t.current_a = getattr(msg, "current_battery", 0) / 100.0
                elif mtype == "BATTERY_STATUS":
                    if getattr(msg, "battery_remaining", -1) >= 0:
                        t.battery_pct = float(msg.battery_remaining)
                elif mtype == "GLOBAL_POSITION_INT":
                    t.lat = msg.lat / 1e7
                    t.lon = msg.lon / 1e7
                    t.alt_m = msg.relative_alt / 1000.0
                elif mtype == "ATTITUDE":
                    t.roll_deg = math.degrees(msg.roll)
                    t.pitch_deg = math.degrees(msg.pitch)
                    t.yaw_deg = math.degrees(msg.yaw)
                elif mtype == "HEARTBEAT":
                    t.armed = bool(msg.base_mode & 0x80)
                    t.link = "up"

    def set_charging(self, charging: bool):
        # In live mode this would trigger a servo / relay command.
        # Stubbed here — the user wires it to their hardware.
        with self._lock:
            self._charging = charging

    def set_origin(self, lat: float, lon: float):
        # Live GPS comes from the autopilot; nothing to do.
        pass

    def snapshot(self) -> Telemetry:
        with self._lock:
            return Telemetry(**asdict(self._t))


def make_source():
    """Pick MAVLink if EAGLE_MAVLINK is set, otherwise mock."""
    conn = os.environ.get("EAGLE_MAVLINK")
    if conn:
        try:
            return MavlinkTelemetry(conn)
        except Exception as e:
            print(f"[telemetry] MAVLink init failed ({e}); falling back to mock")
    return MockTelemetry()
