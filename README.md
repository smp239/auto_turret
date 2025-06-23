````markdown
# 🎯 GPU-Powered Paintball Turret — Roadmap & Implementation Guide

Welcome to the official **hardware-static** roadmap for turning your paintball turret into a GPU-accelerated, sub-150 fps, range-aware splat-machine.  
This document merges the best ideas from prior drafts, locks out extra hardware, and makes **YOLOv11n + ByteTrack** the one-true detector backbone.

---

## ✈️ Table of Contents

1. [Phased Roadmap](#phased-roadmap)
2. [Risk & Effort Profile](#risk--effort-profile)
3. [Key Implementation Hints](#key-implementation-hints)
4. [What **Not** to Do (for now)](#what-not-to-do-for-now)
5. [Milestone Chart](#milestone-chart)
6. [Why this Plan Works](#why-this-plan-works)

---

## Phased Roadmap

| Phase | Definition of Done | Key Tasks & Milestones | Objective Test |
|-------|-------------------|------------------------|----------------|
| **0 GPU foundation** | CUDA/ROCm visible to OpenCV ≥ 4.10; PyTorch sees the GPU | * Install/verify drivers<br>* Build/pip OpenCV-CUDA wheel<br>* One-liner benchmark `torch.hub.load("ultralytics/yolo11", "yolo11n")` on 640 × 480 tensor<br>* Add runtime *GPU-presence guard* in `main.py` | `mean(det_latency) ≤ 7 ms` (≈ 140 fps) with GPU util < 40 % |
| **1 YOLOv11n + ByteTrack backbone** | Per-frame list of `(bbox, id, conf)` for **all** persons replaces Mediapipe | * Wrap Ultralytics YOLO11n<br>* Enable built-in ByteTrack (Ultralytics ≥ 11.0)<br>* Target-selection: closest to image centre & highest conf<br>* Expose live conf threshold | 90 fps; ID stable while two players cross once (10 s clip) |
| **2 Hybrid MOSSE/KCF latency shaver** | Only every **N**-th frame (start N = 5) runs YOLO; correlation filter fills the gaps | * Spawn MOSSE per active ID (conf > τ)<br>* Health-check via PSNR/IoU<br>* Auto-tune `N` to keep GPU < 50 %, CPU < 35 % | Tracker frames < 5 ms; YOLO frames < 15 ms; 1-min stress: < 2 % ID swap |
| **3 6-state KF + measured latency** | KF compensates both motion lag & camera delay | * State `(x,y,vx,vy,ax,ay)`; rebuild F,Q<br>* Measure **Δt_cam** & **Δt_motor** once<br>* `predict_lead_time = Δt_cam + Δt_motor` | 10 m static target: 9 / 10 paintballs land inside 0.5° cone (~9 cm) |
| **4 Turret PD + Feed-Forward; pose cache** | < 0.2 s settle; overshoot < 0.2°; serial traffic ↓ 50 % | * Add D-term (`kd·ė`) and FF (`kff·e_future`)<br>* Cache last pose; poll encoder every 250 ms or on error | ±45° square-wave: rise < 0.2 s; overshoot < 0.2°; serial bytes ↓ ≥ 40 % |
| **5 Monocular range + ballistic drop** | Vertical error from gravity ≤ 5 cm at 8 m | * `range_m = 1.7 m · focal_px / bbox_h_px` (bbox ≥ 40 px)<br>* Lookup/fit flight-time for v₀ ≈ 75 m/s<br>* Tilt correction `Δθ = arctan(0.5·g·t² / range)` | Tripod @ 8 m: 50-shot group fits 0.3 m circle  ≥ 85 % |

> **All phases are independently shippable** and revertible; hardware stays frozen throughout.

---

## Risk & Effort Profile

| Phase | Effort (dev / test) | Technical Risk | Roll-back Plan |
|-------|--------------------|----------------|----------------|
| 0 | 0.5 – 1 day | Low – driver setup | Keep CPU path; skip GPU guard |
| 1 | 1 – 2 days | Medium – API swap | Re-enable Mediapipe branch |
| 2 | 2 days | Medium – drift / re-ID | Set `N = 1` (YOLO every frame) |
| 3 | 2 – 3 days | Medium – maths bugs | Revert to 4-state CV KF |
| 4 | 1 day | Low | Drop D & FF terms |
| 5 | 2 days | Medium – range noise | Gate by bbox size; add disable flag |

---

## Key Implementation Hints

* **Ultralytics ≥ 11.0** already couples YOLOv11n *and* ByteTrack:  
  ```python
  from ultralytics import YOLO
  model = YOLO("yolo11n.pt", tracker="bytetrack")
````

* **Correlation templates**: keep last 8×8 grayscale patches per track; refresh at each YOLO key-frame to fight drift.
* **Derivative term**: low-pass `ė` with a 1-pole IIR (`α = 0.15`) or KF jitter will bite you.
* **Test harness**: dump `TargetReport` + commanded pan/tilt to CSV; a tiny Jupyter notebook plots error vs time after each phase.
* **Runtime params**: extend the JSON schema *once* in Phase 0; later phases just add keys—no breaking changes during live tuning.

---

## What **Not** to Do (for now)

* **Stereo / ToF depth**, **closed-loop steppers/encoders** – forbidden by the hardware freeze.
* **Early ballistic compensation** – wait until Phase 5; controller must be stable first.

---

## Milestone Chart

| Week  | 1         | 2         | 3       | 4         | 5           | 6          | 7       |
| ----- | --------- | --------- | ------- | --------- | ----------- | ---------- | ------- |
| Phase | 0         | **0 → 1** | 1       | **1 → 2** | 2           | **2 → 3**  | 3       |
| —     |           | GPU bench | YOLO+BT | Hybrid    | Hybrid tune | 6-state KF | KF tune |
| Week  | 8         | 9         | 10      | 11        | 12          | 13         | 14      |
| Phase | **3 → 4** | 4         | 4 tune  | **4 → 5** | 5           | 5 test     | buffer  |
| —     | latency   | PD/FF     | cache   | range     | ballistics  | field day  | —       |

*(Slide right if blockers appear; every phase is reversible.)*

---

## Why this Plan Works

* **No hidden deps** – Phase 0 makes GPU capability an explicit, testable requirement.
* **One backbone to rule them all** – YOLOv11n + ByteTrack is locked in from Phase 1 onward.
* **Measured, not guessed** – latency is characterised, range is computed, every claim has a pass/fail test.
* **Safe to merge anytime** – each phase leaves `main.py` runnable; a single flag reverts to the previous behaviour.

Follow the phases in order, keep the CSV logger rolling, and you’ll end up with a paintball turret whose miss distance is dominated by the paintball, not the software. Happy splatting! 🟠

```
```
