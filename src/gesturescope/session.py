from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Iterable, TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from .detector import HandDetection
from .utils import format_seconds


@dataclass
class GestureEvent:
    session_id: str
    frame_index: int
    timestamp_sec: float
    hand_id: int
    handedness: str
    handedness_score: float
    gesture: str
    gesture_confidence: float
    center_x: float
    center_y: float
    bbox_x1: int
    bbox_y1: int
    bbox_x2: int
    bbox_y2: int
    finger_state: dict
    landmarks: list


@dataclass
class GestureSession:
    fps: float = 30.0
    source: str = "webcam"
    session_id: str = field(default_factory=lambda: datetime.now().strftime("%Y%m%d_%H%M%S"))
    events: list[GestureEvent] = field(default_factory=list)

    def add_detections(self, frame_index: int, timestamp_sec: float, detections: Iterable["HandDetection"]) -> None:
        for det in detections:
            self.events.append(
                GestureEvent(
                    session_id=self.session_id,
                    frame_index=int(frame_index),
                    timestamp_sec=float(timestamp_sec),
                    hand_id=int(det.hand_id),
                    handedness=det.handedness,
                    handedness_score=float(det.handedness_score),
                    gesture=det.gesture,
                    gesture_confidence=float(det.gesture_confidence),
                    center_x=float(det.center[0]),
                    center_y=float(det.center[1]),
                    bbox_x1=int(det.bbox[0]),
                    bbox_y1=int(det.bbox[1]),
                    bbox_x2=int(det.bbox[2]),
                    bbox_y2=int(det.bbox[3]),
                    finger_state=dict(det.finger_state),
                    landmarks=list(det.landmarks),
                )
            )

    def to_dataframe(self) -> pd.DataFrame:
        if not self.events:
            return pd.DataFrame()
        return pd.DataFrame([asdict(e) for e in self.events])

    def summarize(self) -> dict:
        df = self.to_dataframe()
        if df.empty:
            return {
                "session_id": self.session_id,
                "source": self.source,
                "total_events": 0,
                "unique_hands": 0,
                "gesture_summary": [],
                "hand_summary": [],
            }

        frame_step = 1.0 / self.fps if self.fps > 0 else 0.0
        gesture_group = (
            df.groupby("gesture")
            .agg(
                observations=("gesture", "count"),
                avg_confidence=("gesture_confidence", "mean"),
                unique_hands=("hand_id", "nunique"),
                first_seen=("timestamp_sec", "min"),
                last_seen=("timestamp_sec", "max"),
            )
            .reset_index()
        )
        gesture_group["approx_duration_sec"] = gesture_group["observations"] * frame_step
        gesture_group = gesture_group.sort_values(["observations", "avg_confidence"], ascending=False)

        hand_group = (
            df.groupby("hand_id")
            .agg(
                observations=("hand_id", "count"),
                most_common_gesture=("gesture", lambda x: x.value_counts().index[0]),
                avg_confidence=("gesture_confidence", "mean"),
                first_seen=("timestamp_sec", "min"),
                last_seen=("timestamp_sec", "max"),
            )
            .reset_index()
            .sort_values("hand_id")
        )
        hand_group["approx_visible_duration_sec"] = hand_group["observations"] * frame_step

        return {
            "session_id": self.session_id,
            "source": self.source,
            "total_events": int(len(df)),
            "unique_hands": int(df["hand_id"].nunique()),
            "start_time_sec": float(df["timestamp_sec"].min()),
            "end_time_sec": float(df["timestamp_sec"].max()),
            "gesture_summary": gesture_group.to_dict(orient="records"),
            "hand_summary": hand_group.to_dict(orient="records"),
        }

    def export(self, export_dir: str | Path) -> dict[str, Path]:
        export_dir = Path(export_dir)
        export_dir.mkdir(parents=True, exist_ok=True)
        session_dir = export_dir / f"session_{self.session_id}"
        session_dir.mkdir(parents=True, exist_ok=True)

        df = self.to_dataframe()
        summary = self.summarize()

        csv_path = session_dir / "session_events.csv"
        json_path = session_dir / "session_events.json"
        summary_json_path = session_dir / "session_summary.json"
        md_path = session_dir / "gesture_report.md"
        html_path = session_dir / "gesture_report.html"

        if not df.empty:
            df.to_csv(csv_path, index=False)
            # JSON lines would be more scalable, but pretty JSON is easier for class demos.
            json_path.write_text(json.dumps([asdict(e) for e in self.events], indent=2), encoding="utf-8")
        else:
            csv_path.write_text("No events recorded.\n", encoding="utf-8")
            json_path.write_text("[]\n", encoding="utf-8")

        summary_json_path.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
        md_path.write_text(self._make_markdown_report(summary), encoding="utf-8")
        html_path.write_text(self._make_html_report(summary), encoding="utf-8")

        return {
            "session_dir": session_dir,
            "csv": csv_path,
            "json": json_path,
            "summary_json": summary_json_path,
            "markdown": md_path,
            "html": html_path,
        }

    def _make_markdown_report(self, summary: dict) -> str:
        lines = []
        lines.append(f"# GestureScope Report: {summary['session_id']}")
        lines.append("")
        lines.append(f"- Source: `{summary.get('source', self.source)}`")
        lines.append(f"- Total observations: **{summary.get('total_events', 0)}**")
        lines.append(f"- Unique tracked hands: **{summary.get('unique_hands', 0)}**")
        if summary.get("total_events", 0):
            lines.append(f"- Timeline: **{format_seconds(summary['start_time_sec'])} → {format_seconds(summary['end_time_sec'])}**")
        lines.append("")
        lines.append("## Gesture Summary")
        lines.append("")
        lines.append("| Gesture | Observations | Approx. Duration | Avg. Confidence | Unique Hands | First Seen | Last Seen |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|")
        for row in summary.get("gesture_summary", []):
            lines.append(
                f"| {row['gesture']} | {int(row['observations'])} | "
                f"{row['approx_duration_sec']:.2f}s | {row['avg_confidence']:.2f} | "
                f"{int(row['unique_hands'])} | {format_seconds(row['first_seen'])} | {format_seconds(row['last_seen'])} |"
            )
        lines.append("")
        lines.append("## Per-Hand Summary")
        lines.append("")
        lines.append("| Hand ID | Observations | Approx. Visible Duration | Most Common Gesture | Avg. Confidence | First Seen | Last Seen |")
        lines.append("|---:|---:|---:|---|---:|---:|---:|")
        for row in summary.get("hand_summary", []):
            lines.append(
                f"| Hand_{int(row['hand_id'])} | {int(row['observations'])} | "
                f"{row['approx_visible_duration_sec']:.2f}s | {row['most_common_gesture']} | "
                f"{row['avg_confidence']:.2f} | {format_seconds(row['first_seen'])} | {format_seconds(row['last_seen'])} |"
            )
        lines.append("")
        lines.append("## Notes")
        lines.append("")
        lines.append("This report is generated from landmark-based hand tracking. Counts are observations, not medically or legally verified measurements.")
        return "\n".join(lines) + "\n"

    def _make_html_report(self, summary: dict) -> str:
        md = self._make_markdown_report(summary)
        body = "\n".join(f"<p>{line}</p>" if line else "" for line in md.splitlines())
        return f"""<!doctype html>
<html lang=\"en\">
<head>
  <meta charset=\"utf-8\" />
  <title>GestureScope Report {summary['session_id']}</title>
  <style>
    body {{ font-family: Arial, sans-serif; max-width: 960px; margin: 40px auto; line-height: 1.5; }}
    p {{ margin: 0.25rem 0; }}
    code {{ background: #eee; padding: 2px 4px; }}
  </style>
</head>
<body>
<pre>{md}</pre>
</body>
</html>
"""
