# main.py
"""
Entry-point for the face-tracking system.

Live-tuning
-----------
While the program is running you can edit ``runtime_params.json`` and the new
values (lead-time, noise terms, controller gains, etc.) will take effect on
the very next frame.  See ``face_tracking/live_tuning.py`` for details.

Extra (Windows / WSL2 only)
---------------------------
When executed inside WSL2 the script ensures that the required USB devices
(bus-ids listed in ``_REQUIRED_BUSIDS``) are attached via *usbipd* **before**
the vision stack starts.  Harmless on non-Windows hosts.
"""
from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
import time
from typing import Iterable, Set

from face_tracking.config import (
    CameraConfig,
    DetectorConfig,
    PanTiltConfig,
    SearchConfig,
    TrackerConfig,
)
from face_tracking.processor import TargetingProcessor

# ────────────────────────────────────────────────────────────────────────────
#   USBIPD  HELPER (only relevant on Windows/WSL2)
# ────────────────────────────────────────────────────────────────────────────
_REQUIRED_BUSIDS = ("6-1", "1-9")  # Update if you add/remove hardware


def _ensure_usb_devices(busids: Iterable[str]) -> None:
    """Detach + attach each bus-id so it shows up inside WSL."""
    usbipd_exe = shutil.which("usbipd.exe") or shutil.which("usbipd")
    if not usbipd_exe:  # not on Windows or usbipd not installed
        return

    try:
        cp = subprocess.run(
            [usbipd_exe, "list"],
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
    except subprocess.SubprocessError as exc:
        print(f"[USBIPD] Could not run `usbipd list`: {exc}", file=sys.stderr)
        return

    shared: Set[str] = set()
    for line in cp.stdout.splitlines():
        m = re.match(r"^\s*(\d+-\d+)\s+.*\s+(Attached)\s*$", line)
        if m:
            shared.add(m.group(1))

    for bid in busids:
        if bid in shared:
            print(f"[USBIPD] Detaching bus {bid} from WSL …")
            try:
                subprocess.run([usbipd_exe, "detach", "-b", bid], check=True)
                time.sleep(1)
            except subprocess.SubprocessError as exc:
                print(f"[USBIPD]   ! failed to detach {bid}: {exc}", file=sys.stderr)

        print(f"[USBIPD] Attaching bus {bid} to WSL …")
        try:
            subprocess.run([usbipd_exe, "attach", "-b", bid, "--wsl"], check=True)
            time.sleep(2)
        except subprocess.SubprocessError as exc:
            print(f"[USBIPD]   ! failed to attach {bid}: {exc}", file=sys.stderr)


def detach_usb_devices(busids: Iterable[str]) -> None:
    """Detach the given USB bus-ids from WSL on shutdown (best-effort)."""
    usbipd_exe = shutil.which("usbipd.exe") or shutil.which("usbipd")
    if not usbipd_exe:
        return

    try:
        cp = subprocess.run(
            [usbipd_exe, "list"],
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
    except subprocess.SubprocessError as exc:
        print(f"[USBIPD] Could not run `usbipd list`: {exc}", file=sys.stderr)
        return

    attached: Set[str] = set()
    for line in cp.stdout.splitlines():
        m = re.match(r"^\s*(\d+-\d+)\s+.*\s+(Attached)\s*$", line)
        if m:
            attached.add(m.group(1))

    for bid in busids:
        if bid not in attached:
            continue
        print(f"[USBIPD] Detaching bus {bid} from WSL …")
        try:
            subprocess.run([usbipd_exe, "detach", "-b", bid], check=True)
            time.sleep(1)
        except subprocess.SubprocessError as exc:
            print(f"[USBIPD]   ! failed to detach {bid}: {exc}", file=sys.stderr)


# ────────────────────────────────────────────────────────────────────────────
#   M A I N
# ────────────────────────────────────────────────────────────────────────────
def main() -> None:
    _ensure_usb_devices(_REQUIRED_BUSIDS)

    print("Initializing Face-Tracking System…")
    print("Hint: edit 'runtime_params.json' at any time to tweak parameters.\n")

    # -------------------- Config blobs --------------------
    cam_cfg  = CameraConfig()
    det_cfg  = DetectorConfig()
    trk_cfg  = TrackerConfig()
    pt_cfg   = PanTiltConfig(port="/dev/ttyACM0")  # set to None to disable turret
    srch_cfg = SearchConfig(
        enabled=True,
        pan_angles_deg=[45.0, 0.0, -40.0, 0.0],
        dwell_time_s=2.5,
        no_detection_threshold_s=3.5,
    )

    # ------------------------ Banner ----------------------
    print(
        f"Camera: idx={cam_cfg.device_index}, "
        f"{cam_cfg.width}x{cam_cfg.height}@{cam_cfg.fps_request} FPS, "
        f"HFOV={cam_cfg.horizontal_fov_deg}°"
    )
    print(
        f"Detector: conf={det_cfg.min_detection_confidence}, "
        f"min_bbox={det_cfg.min_bbox_size_for_tracking}px"
    )
    if pt_cfg.port:
        print(
            f"PanTilt: port={pt_cfg.port}, "
            f"Kp=({pt_cfg.kp_pan},{pt_cfg.kp_tilt}), "
            f"deadzone={pt_cfg.error_deadzone_px}px, "
            f"rate={pt_cfg.max_cmd_rate_hz} Hz"
        )
        print(
            f"Search: enabled={srch_cfg.enabled}, "
            f"angles={srch_cfg.pan_angles_deg}, "
            f"dwell={srch_cfg.dwell_time_s}s, "
            f"threshold={srch_cfg.no_detection_threshold_s}s"
        )
    else:
        print("PanTilt: DISABLED")

    # ------------------------ Run -------------------------
    TargetingProcessor(cam_cfg, det_cfg, trk_cfg, pt_cfg, srch_cfg).run()
    detach_usb_devices(_REQUIRED_BUSIDS)
    print("Main program finished.")


if __name__ == "__main__":
    main()
