# detector.py
"""MediaPipe face-detection adapter."""
from typing import List, Tuple, Optional

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.framework.formats import location_data_pb2

from face_tracking.config import DetectorConfig


class MediaPipeFaceDetector:
    def __init__(self, config: DetectorConfig):
        self.config = config
        self.detector = mp.solutions.face_detection.FaceDetection(
            model_selection=config.model_selection,
            min_detection_confidence=config.min_detection_confidence,
        )

    def detect(self, frame_bgr: np.ndarray) -> List[Tuple[Tuple[float, float, float, float], float]]:
        """
        Returns a list of ((x, y, w, h), confidence), with floats and clamped to image.
        """
        out: List[Tuple[Tuple[float, float, float, float], float]] = []
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        results = self.detector.process(rgb)
        ih, iw = rgb.shape[:2]

        if results.detections:
            for det in results.detections:
                ld = det.location_data
                if not ld or ld.format != location_data_pb2.LocationData.RELATIVE_BOUNDING_BOX:
                    continue
                bb = ld.relative_bounding_box
                x = bb.xmin * iw
                y = bb.ymin * ih
                w = bb.width * iw
                h = bb.height * ih
                # Clamp to valid region
                x = max(0.0, min(x, iw - w))
                y = max(0.0, min(y, ih - h))
                # Size threshold
                if w < self.config.min_bbox_size_for_tracking or h < self.config.min_bbox_size_for_tracking:
                    continue
                conf = float(det.score[0]) if det.score else 0.0
                out.append(((x, y, w, h), conf))

        out.sort(key=lambda t: t[1], reverse=True)
        return out

    def close(self) -> None:
        self.detector.close()
