from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Tuple

import numpy as np


@dataclass(frozen=True)
class Point2D:
    x: float
    y: float


def euclidean(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    return float(np.linalg.norm(np.array(a, dtype=float) - np.array(b, dtype=float)))


def normalized_landmark_list(landmarks) -> list[tuple[float, float, float]]:
    """Convert MediaPipe landmarks to a plain Python list."""
    return [(float(lm.x), float(lm.y), float(lm.z)) for lm in landmarks]


def landmark_center(landmarks: Iterable[tuple[float, float, float]]) -> tuple[float, float]:
    pts = list(landmarks)
    if not pts:
        return (0.0, 0.0)
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    return (float(sum(xs) / len(xs)), float(sum(ys) / len(ys)))


def normalized_to_pixel(point: tuple[float, float], width: int, height: int) -> tuple[int, int]:
    x = max(0, min(width - 1, int(point[0] * width)))
    y = max(0, min(height - 1, int(point[1] * height)))
    return x, y


def make_timestamp(frame_index: int, fps: float) -> float:
    if fps <= 0:
        return 0.0
    return float(frame_index / fps)


def format_seconds(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    minutes = int(seconds // 60)
    sec = seconds - (minutes * 60)
    return f"{minutes:02d}:{sec:05.2f}"
