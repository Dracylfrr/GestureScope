import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from gesturescope.gesture_rules import classify_gesture


def make_open_palm():
    # Very rough normalized landmarks shaped like an open hand.
    pts = [(0.5, 0.8, 0.0)] * 21
    pts[0] = (0.5, 0.8, 0.0)
    pts[1] = (0.42, 0.70, 0.0)
    pts[2] = (0.36, 0.64, 0.0)
    pts[3] = (0.30, 0.58, 0.0)
    pts[4] = (0.24, 0.52, 0.0)
    pts[5] = (0.44, 0.58, 0.0)
    pts[6] = (0.42, 0.46, 0.0)
    pts[7] = (0.41, 0.36, 0.0)
    pts[8] = (0.40, 0.24, 0.0)
    pts[9] = (0.50, 0.56, 0.0)
    pts[10] = (0.50, 0.42, 0.0)
    pts[11] = (0.50, 0.31, 0.0)
    pts[12] = (0.50, 0.18, 0.0)
    pts[13] = (0.56, 0.58, 0.0)
    pts[14] = (0.58, 0.46, 0.0)
    pts[15] = (0.59, 0.36, 0.0)
    pts[16] = (0.60, 0.25, 0.0)
    pts[17] = (0.62, 0.62, 0.0)
    pts[18] = (0.66, 0.52, 0.0)
    pts[19] = (0.68, 0.42, 0.0)
    pts[20] = (0.70, 0.32, 0.0)
    return pts


def test_open_palm_classifies():
    result = classify_gesture(make_open_palm(), "Right")
    assert result.label in {"Open_Palm", "Unknown"}
    assert 0 <= result.confidence <= 1
