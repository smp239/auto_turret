# main.py
"""
Entry-point for the face-tracking system.

Extra: on Windows/WSL we automatically make sure the required USB devices
(busid 6-1 and 6-2) are attached to WSL with usbipd *before* the vision stack
starts.  If they're already in the “Shared / Attached” state nothing is done.

Requirements:
* usbipd-win 4.0+ installed on the host
* This script running inside WSL2 (so it can invoke usbipd.exe)
"""
from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
from typing import Iterable, Set

from face_tracking.config import (
    CameraConfig,
    DetectorConfig,
    PanTiltConfig,
    SearchConfig,
    TrackerConfig,
)
from face_tracking.processor import TargetingProcessor

# ---------------------------------------------------------------------------
#                        USBIPD  helper
# ---------------------------------------------------------------------------
_REQUIRED_BUSIDS = ("6-1", "6-2")  # Update if you add/remove hardware


def _ensure_usb_devices(busids: Iterable[str]) -> None:
    """
    Call `usbipd.exe attach -b <busid> --wsl` for each bus ID that is *not*
    already in the Shared/Attached state.

    Harmless on non-Windows hosts: if usbipd.exe is missing we just skip.
    """
    usbipd_exe = shutil.which("usbipd.exe") or shutil.which("usbipd")
    if not usbipd_exe:  # Not on Windows or usbipd not installed
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
        m = re.match(r"^\s*(\d+-\d+)\s+.*\s+(Shared|Attached)\s*$", line)
        if m:
            shared.add(m.group(1))

    for bid in busids:
        if bid not in shared:
            print(f"[USBIPD] Attaching bus {bid} to WSL …")
            try:
                subprocess.run([usbipd_exe, "attach", "-b", bid, "--wsl"], check=True)
            except subprocess.SubprocessError as exc:
                print(f"[USBIPD]   ! failed to attach {bid}: {exc}", file=sys.stderr)


# ---------------------------------------------------------------------------
#                        Main  program
# ---------------------------------------------------------------------------
def main() -> None:
    # Make sure our Arduino + Arducam are mapped into WSL first
    _ensure_usb_devices(_REQUIRED_BUSIDS)

    print("Initializing Face-Tracking System…")

    # -------------------- Config --------------------
    cam_cfg = CameraConfig()
    det_cfg = DetectorConfig()
    trk_cfg = TrackerConfig()
    pt_cfg = PanTiltConfig(port="/dev/ttyACM0")  # set to None to disable turret
    srch_cfg = SearchConfig(
        enabled=True,
        pan_angles_deg=[45.0, 0.0, -40.0, 0.0],
        dwell_time_s=2.5,
        no_detection_threshold_s=3.5,
    )

    # -------------------- Banner --------------------
    print(
        f"Camera: idx={cam_cfg.device_index}, {cam_cfg.width}x{cam_cfg.height}@"
        f"{cam_cfg.fps_request} FPS, HFOV={cam_cfg.horizontal_fov_deg}°"
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
            f"Search: enabled={srch_cfg.enabled}, angles={srch_cfg.pan_angles_deg}, "
            f"dwell={srch_cfg.dwell_time_s}s, threshold={srch_cfg.no_detection_threshold_s}s"
        )
    else:
        print("PanTilt: DISABLED")

    # ------------------- Run! -----------------------
    TargetingProcessor(cam_cfg, det_cfg, trk_cfg, pt_cfg, srch_cfg).run()
    print("Main program finished.")


if __name__ == "__main__":
    main()
