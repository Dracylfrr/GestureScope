from __future__ import annotations

import argparse
from pathlib import Path

import cv2

from .detector import MultiHandGestureDetector
from .session import GestureSession
from .utils import make_timestamp


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Process a video file with GestureScope")
    parser.add_argument("video_path", type=str, help="Path to a video file such as .mp4, .mov, or .avi")
    parser.add_argument("--max-hands", type=int, default=6, help="Maximum number of hands to detect.")
    parser.add_argument("--min-detection-confidence", type=float, default=0.55)
    parser.add_argument("--min-tracking-confidence", type=float, default=0.55)
    parser.add_argument("--export-dir", type=str, default="outputs", help="Folder for reports.")
    parser.add_argument("--preview", action="store_true", help="Show annotated preview while processing.")
    parser.add_argument("--save-annotated", action="store_true", help="Save an annotated output video.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    video_path = Path(args.video_path)
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    export_dir = Path(args.export_dir)
    export_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 1:
        fps = 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    detector = MultiHandGestureDetector(
        max_hands=args.max_hands,
        min_detection_confidence=args.min_detection_confidence,
        min_tracking_confidence=args.min_tracking_confidence,
    )
    session = GestureSession(fps=fps, source=str(video_path))

    writer = None
    if args.save_annotated:
        annotated_dir = export_dir / "annotated_videos"
        annotated_dir.mkdir(parents=True, exist_ok=True)
        out_path = annotated_dir / f"{video_path.stem}_annotated.mp4"
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(out_path), fourcc, fps, (width, height))
        print(f"Saving annotated video to: {out_path}")

    frame_index = 0
    print(f"Processing video: {video_path}")

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            detections = detector.process_frame(frame, frame_index)
            timestamp_sec = make_timestamp(frame_index, fps)
            session.add_detections(frame_index, timestamp_sec, detections)
            detector.draw_detections(frame, detections)

            if writer is not None:
                writer.write(frame)

            if args.preview:
                cv2.imshow("GestureScope Video Processing", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            if frame_index % int(max(1, fps * 5)) == 0:
                print(f"Processed frame {frame_index}...")

            frame_index += 1

    finally:
        paths = session.export(export_dir)
        print(f"Report exported to: {paths['session_dir']}")
        detector.close()
        cap.release()
        if writer is not None:
            writer.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
