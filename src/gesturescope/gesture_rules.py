from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

from .utils import euclidean

# MediaPipe Hands landmark indices.
WRIST = 0
THUMB_CMC = 1
THUMB_MCP = 2
THUMB_IP = 3
THUMB_TIP = 4
INDEX_MCP = 5
INDEX_PIP = 6
INDEX_DIP = 7
INDEX_TIP = 8
MIDDLE_MCP = 9
MIDDLE_PIP = 10
MIDDLE_DIP = 11
MIDDLE_TIP = 12
RING_MCP = 13
RING_PIP = 14
RING_DIP = 15
RING_TIP = 16
PINKY_MCP = 17
PINKY_PIP = 18
PINKY_DIP = 19
PINKY_TIP = 20

FINGER_JOINTS = {
    "index": (INDEX_MCP, INDEX_PIP, INDEX_DIP, INDEX_TIP),
    "middle": (MIDDLE_MCP, MIDDLE_PIP, MIDDLE_DIP, MIDDLE_TIP),
    "ring": (RING_MCP, RING_PIP, RING_DIP, RING_TIP),
    "pinky": (PINKY_MCP, PINKY_PIP, PINKY_DIP, PINKY_TIP),
}


@dataclass
class GestureResult:
    label: str
    confidence: float
    finger_state: Dict[str, bool]


def _xy(landmarks, idx: int) -> Tuple[float, float]:
    return (float(landmarks[idx][0]), float(landmarks[idx][1]))


def _finger_extended(landmarks, name: str) -> bool:
    """Estimate extension using wrist-to-tip distance compared with wrist-to-pip distance.

    This works reasonably well for upright classroom gestures and avoids complicated
    hand-orientation logic. It is intentionally simple for an MVP.
    """
    _, pip, _, tip = FINGER_JOINTS[name]
    wrist = _xy(landmarks, WRIST)
    return euclidean(wrist, _xy(landmarks, tip)) > euclidean(wrist, _xy(landmarks, pip)) * 1.15


def _thumb_extended(landmarks, handedness: str) -> bool:
    wrist = _xy(landmarks, WRIST)
    tip = _xy(landmarks, THUMB_TIP)
    ip = _xy(landmarks, THUMB_IP)
    # Distance-based thumb extension is more stable across mirrored cameras than x-axis logic.
    return euclidean(wrist, tip) > euclidean(wrist, ip) * 1.18


def _pinch_distance(landmarks) -> float:
    return euclidean(_xy(landmarks, THUMB_TIP), _xy(landmarks, INDEX_TIP))


def classify_gesture(landmarks, handedness: str = "Unknown") -> GestureResult:
    """Rule-based static gesture classifier.

    This is not meant to replace a trained classifier. It gives a working MVP and
    creates labeled data that can later be used to train a better model.
    """
    if not landmarks or len(landmarks) < 21:
        return GestureResult("Unknown", 0.0, {})

    fingers = {
        "thumb": _thumb_extended(landmarks, handedness),
        "index": _finger_extended(landmarks, "index"),
        "middle": _finger_extended(landmarks, "middle"),
        "ring": _finger_extended(landmarks, "ring"),
        "pinky": _finger_extended(landmarks, "pinky"),
    }

    non_thumb_count = sum([fingers["index"], fingers["middle"], fingers["ring"], fingers["pinky"]])
    extended_count = sum(fingers.values())

    thumb_tip = _xy(landmarks, THUMB_TIP)
    index_tip = _xy(landmarks, INDEX_TIP)
    index_mcp = _xy(landmarks, INDEX_MCP)
    wrist = _xy(landmarks, WRIST)
    pinch = _pinch_distance(landmarks)
    palm_scale = max(euclidean(wrist, index_mcp), 1e-6)

    # Pinch: thumb and index tips very close; other fingers may vary.
    if pinch < palm_scale * 0.45:
        return GestureResult("Pinch", 0.82, fingers)

    # All fingers open.
    if extended_count >= 5:
        return GestureResult("Open_Palm", 0.86, fingers)

    # Fist: no non-thumb fingers extended and thumb not strongly extended.
    if extended_count == 0 or (not any([fingers["index"], fingers["middle"], fingers["ring"], fingers["pinky"]]) and not fingers["thumb"]):
        return GestureResult("Closed_Fist", 0.84, fingers)

    # Pointing up: index only.
    if fingers["index"] and not any([fingers["middle"], fingers["ring"], fingers["pinky"]]):
        return GestureResult("Pointing_Up", 0.78, fingers)

    # Victory: index and middle up, ring and pinky down.
    if fingers["index"] and fingers["middle"] and not fingers["ring"] and not fingers["pinky"]:
        return GestureResult("Victory", 0.80, fingers)

    # Thumbs up/down: thumb extended while other fingers mostly closed.
    if fingers["thumb"] and non_thumb_count <= 1:
        # Image y-axis points downward. Smaller y means visually higher.
        if thumb_tip[1] < wrist[1] - palm_scale * 0.25:
            return GestureResult("Thumbs_Up", 0.75, fingers)
        if thumb_tip[1] > wrist[1] + palm_scale * 0.25:
            return GestureResult("Thumbs_Down", 0.75, fingers)

    # Fallback label.
    confidence = 0.45 + min(0.25, extended_count * 0.04)
    return GestureResult("Unknown", confidence, fingers)
