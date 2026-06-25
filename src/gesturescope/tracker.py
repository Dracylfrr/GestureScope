from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, Optional

from .utils import euclidean


@dataclass
class Track:
    track_id: int
    center: tuple[float, float]
    last_seen_frame: int
    age: int = 0
    missing_frames: int = 0
    metadata: dict = field(default_factory=dict)


class HandTracker:
    """Tiny greedy tracker for assigning stable IDs to detected hands.

    It matches new hand centers to previous centers by nearest distance.
    This is enough for a classroom MVP, but it can be replaced later with SORT,
    DeepSORT, ByteTrack, or landmark embedding matching.
    """

    def __init__(self, max_distance: float = 0.18, max_missing_frames: int = 15):
        self.max_distance = max_distance
        self.max_missing_frames = max_missing_frames
        self._tracks: Dict[int, Track] = {}
        self._next_id = 1

    @property
    def tracks(self) -> Dict[int, Track]:
        return dict(self._tracks)

    def update(self, centers: Iterable[tuple[float, float]], frame_index: int) -> list[int]:
        centers = list(centers)
        assigned_ids: list[Optional[int]] = [None] * len(centers)
        used_tracks: set[int] = set()

        # Greedy nearest-neighbor matching.
        candidates: list[tuple[float, int, int]] = []
        for det_idx, center in enumerate(centers):
            for track_id, track in self._tracks.items():
                if track_id in used_tracks:
                    continue
                dist = euclidean(center, track.center)
                if dist <= self.max_distance:
                    candidates.append((dist, det_idx, track_id))

        candidates.sort(key=lambda x: x[0])
        for _, det_idx, track_id in candidates:
            if assigned_ids[det_idx] is not None or track_id in used_tracks:
                continue
            assigned_ids[det_idx] = track_id
            used_tracks.add(track_id)

        # Update matched tracks and create new tracks.
        for det_idx, center in enumerate(centers):
            track_id = assigned_ids[det_idx]
            if track_id is None:
                track_id = self._next_id
                self._next_id += 1
                self._tracks[track_id] = Track(
                    track_id=track_id,
                    center=center,
                    last_seen_frame=frame_index,
                    age=1,
                    missing_frames=0,
                )
                assigned_ids[det_idx] = track_id
            else:
                track = self._tracks[track_id]
                track.center = center
                track.last_seen_frame = frame_index
                track.age += 1
                track.missing_frames = 0

        # Age out missing tracks.
        current_ids = {tid for tid in assigned_ids if tid is not None}
        for track_id in list(self._tracks.keys()):
            if track_id not in current_ids:
                self._tracks[track_id].missing_frames += 1
                if self._tracks[track_id].missing_frames > self.max_missing_frames:
                    del self._tracks[track_id]

        return [int(tid) for tid in assigned_ids if tid is not None]

    def reset(self) -> None:
        self._tracks.clear()
        self._next_id = 1
