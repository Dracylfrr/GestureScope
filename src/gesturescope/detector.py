from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import cv2
import mediapipe as mp

from .gesture_rules import GestureResult, classify_gesture
from .tracker import HandTracker
from .utils import landmark_center, normalized_landmark_list, normalized_to_pixel


@dataclass
class HandDetection:
    hand_id: int
    handedness: str
    handedness_score: float
    gesture: str
    gesture_confidence: float
    center: tuple[float, float]
    bbox: tuple[int, int, int, int]
    landmarks: list[tuple[float, float, float]]
    finger_state: dict[str, bool]


class MultiHandGestureDetector:
    """MediaPipe-based multi-hand detector with a simple gesture classifier."""

    def __init__(
        self,
        max_hands: int = 6,
        min_detection_confidence: float = 0.55,
        min_tracking_confidence: float = 0.55,
        tracker: Optional[HandTracker] = None,
    ) -> None:
        self.max_hands = max(1, int(max_hands))
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_styles = mp.solutions.drawing_styles
        self.tracker = tracker or HandTracker()
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=self.max_hands,
            model_complexity=1,
            min_detection_confidence=float(min_detection_confidence),
            min_tracking_confidence=float(min_tracking_confidence),
        )

    def close(self) -> None:
        self.hands.close()

    def process_frame(self, frame_bgr, frame_index: int) -> list[HandDetection]:
        height, width = frame_bgr.shape[:2]
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        frame_rgb.flags.writeable = False
        results = self.hands.process(frame_rgb)

        if not results.multi_hand_landmarks:
            self.tracker.update([], frame_index)
            return []

        plain_landmarks = [normalized_landmark_list(hand_lms.landmark) for hand_lms in results.multi_hand_landmarks]
        centers = [landmark_center(lms) for lms in plain_landmarks]
        track_ids = self.tracker.update(centers, frame_index)

        detections: list[HandDetection] = []
        handedness_results = results.multi_handedness or []

        for idx, landmarks in enumerate(plain_landmarks):
            if idx >= len(track_ids):
                continue

            handedness_label = "Unknown"
            handedness_score = 0.0
            if idx < len(handedness_results) and handedness_results[idx].classification:
                cls = handedness_results[idx].classification[0]
                handedness_label = str(cls.label)
                handedness_score = float(cls.score)

            gesture: GestureResult = classify_gesture(landmarks, handedness_label)

            xs = [p[0] for p in landmarks]
            ys = [p[1] for p in landmarks]
            x1, y1 = normalized_to_pixel((min(xs), min(ys)), width, height)
            x2, y2 = normalized_to_pixel((max(xs), max(ys)), width, height)
            pad = 12
            x1 = max(0, x1 - pad)
            y1 = max(0, y1 - pad)
            x2 = min(width - 1, x2 + pad)
            y2 = min(height - 1, y2 + pad)

            detections.append(
                HandDetection(
                    hand_id=track_ids[idx],
                    handedness=handedness_label,
                    handedness_score=handedness_score,
                    gesture=gesture.label,
                    gesture_confidence=gesture.confidence,
                    center=centers[idx],
                    bbox=(x1, y1, x2, y2),
                    landmarks=landmarks,
                    finger_state=gesture.finger_state,
                )
            )

        return detections

    def draw_detections(self, frame_bgr, detections: list[HandDetection], raw_mediapipe_results: Any = None):
        """Draw boxes and labels. Landmark skeletons are drawn manually from stored landmarks."""
        height, width = frame_bgr.shape[:2]
        connections = self.mp_hands.HAND_CONNECTIONS

        for det in detections:
            # Draw landmark connections.
            pixel_points = {
                idx: normalized_to_pixel((lm[0], lm[1]), width, height)
                for idx, lm in enumerate(det.landmarks)
            }
            for a, b in connections:
                if a in pixel_points and b in pixel_points:
                    cv2.line(frame_bgr, pixel_points[a], pixel_points[b], (255, 255, 255), 1)
            for point in pixel_points.values():
                cv2.circle(frame_bgr, point, 3, (255, 255, 255), -1)

            x1, y1, x2, y2 = det.bbox
            cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), (255, 255, 255), 2)
            label = f"Hand_{det.hand_id} {det.handedness} {det.gesture} {det.gesture_confidence:.2f}"
            cv2.putText(
                frame_bgr,
                label,
                (x1, max(24, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

        return frame_bgr
