# pan_tilt.py
"""Serial-controlled Pan-Tilt turret interface."""
from __future__ import annotations

import re
import threading
import time
from dataclasses import dataclass
from enum import Enum, auto
from pathlib import Path
from typing import Optional, Tuple

import serial


# ------------------- Exceptions / Enums -------------------
class FirmwareError(RuntimeError):
    """Raised when the firmware replies with an unexpected line."""


class _Ack(str, Enum):
    HOME_OK = auto()
    REST_OK = auto()
    MOVE_OK = auto()


_ACK_PATTERN = re.compile(r"^(HOME_OK|REST_OK|MOVE_OK)$")
_POS_STEPS   = re.compile(r"^(-?\d+),(-?\d+)$")
_POS_DEG     = re.compile(r"^(-?\d+(?:\.\d+)?),(-?\d+(?:\.\d+)?)$")


# ------------------ Internal dataclass -------------------
@dataclass(slots=True)
class _SerialCfg:
    port: str
    baudrate: int = 250_000
    timeout: float = 2.0


# ---------------------- Main class ----------------------
class PanTilt:
    """High-level wrapper around the firmware’s ASCII protocol."""
    def __init__(
        self,
        port: str | Path,
        baudrate: int = 250_000,
        timeout: float = 10.0,
        write_timeout: float | None = None,
        *,
        eol: str = "\n",
        auto_flush: bool = True,
    ):
        self._cfg = _SerialCfg(str(port), baudrate, timeout)
        self._eol = eol.encode()
        self._auto_flush = auto_flush
        self._write_timeout = write_timeout
        self._ser: Optional[serial.Serial] = None
        self._lock = threading.Lock()

    # ---------------- Serial plumbing ----------------
    def open(self) -> None:
        if self._ser and self._ser.is_open:
            return
        wt = self._write_timeout if self._write_timeout is not None else self._cfg.timeout
        self._ser = serial.Serial(
            port=self._cfg.port,
            baudrate=self._cfg.baudrate,
            timeout=self._cfg.timeout,
            write_timeout=wt,
            bytesize=serial.EIGHTBITS,
            parity=serial.PARITY_NONE,
            stopbits=serial.STOPBITS_ONE,
        )
        time.sleep(0.2)
        if self._ser.is_open:
            self._ser.reset_input_buffer()

    def close(self) -> None:
        if self._ser and self._ser.is_open:
            self._ser.close()
        self._ser = None

    def is_open(self) -> bool:
        return bool(self._ser and self._ser.is_open)

    # ------------------ Public API -------------------
    def home(self) -> None:
        self._cmd("HOME", expect=_Ack.HOME_OK)

    def rest(self) -> None:
        self._cmd("REST", expect=_Ack.REST_OK)

    def move_steps(self, pan: int, tilt: int) -> None:
        self._cmd(f"MOVEW {pan} {tilt}", expect=_Ack.MOVE_OK)

    def move_deg(self, pan_deg: float, tilt_deg: float) -> None:
        self._cmd(f"MOVE_DEGW {pan_deg:.3f} {tilt_deg:.3f}", expect=_Ack.MOVE_OK)

    def async_move_steps(self, pan: int, tilt: int) -> None:
        self._cmd(f"MOVE {pan} {tilt}", wait=False)

    def async_move_deg(self, pan_deg: float, tilt_deg: float) -> None:
        self._cmd(f"MOVE_DEG {pan_deg:.3f} {tilt_deg:.3f}", wait=False)

    def position_steps(self) -> Tuple[int, int]:
        line = self._cmd("POSITION", expect_pattern=_POS_STEPS)
        return tuple(map(int, line.split(",")))  # type: ignore[arg-type]

    def position_deg(self) -> Tuple[float, float]:
        line = self._cmd("POSITION_DEG", expect_pattern=_POS_DEG)
        return tuple(map(float, line.split(",")))  # type: ignore[arg-type]

    # -------------------- Tuning ---------------------
    def set_max_speed(self, value: float) -> None:
        self._cmd(f"SET MAX_SPEED {value}", wait=False)

    def set_max_accel(self, value: float) -> None:
        self._cmd(f"SET MAX_ACCEL {value}", wait=False)

    def set_rms_current(self, mA: int) -> None:
        self._cmd(f"SET RMS_CURRENT {mA}", wait=False)

    def set_stealth_chop(self, on: bool = True) -> None:
        self._cmd(f"SET STEALTHCHOP {'ON' if on else 'OFF'}", wait=False)

    # ----------------- Internal core -----------------
    def _cmd(
        self,
        cmd: str,
        expect: Optional[_Ack] = None,
        *,
        expect_pattern: Optional[re.Pattern[str]] = None,
        wait: bool = True,
    ) -> Optional[str]:
        if not self.is_open():
            raise RuntimeError("Serial port is not open")
        with self._lock:
            self._ser.write(cmd.upper().encode() + self._eol)
            if self._auto_flush:
                self._ser.flush()
            if not wait:
                return None
            while True:
                raw = self._ser.readline()
                if not raw:
                    raise FirmwareError(f"Timeout waiting for response to {cmd!r}")
                line = raw.decode(errors="replace").strip()
                if expect and line == expect.name:
                    return line
                if expect_pattern and expect_pattern.match(line):
                    return line

    # ---------------- Context / repr ---------------
    def __enter__(self) -> "PanTilt":
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

    def __repr__(self) -> str:
        state = "open" if self.is_open() else "closed"
        return f"<PanTilt port={self._cfg.port!r} ({state})>"
