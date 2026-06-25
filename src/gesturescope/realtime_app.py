from __future__ import annotations

import argparse
from pathlib import Path

import cv2

from .detector import MultiHandGestureDetector
from .session import GestureSession
from .utils import make_timestamp


def draw_hud(frame, recording: bool, frame_index: int, detections_count: int, export_dir: str):
    status = "RECORDING" if recording else "PAUSED"
    text_lines = [
        f"GestureScope MVP | {status}",
        f"Frame: {frame_index} | Hands: {detections_count}",
        "Keys: r=start/stop  e=export  s=screenshot  q=quit",
        f"Export dir: {export_dir}",
    ]
    y = 26
    for line in text_lines:
        cv2.putText(frame, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 255, 255), 2, cv2.LINE_AA)
        y += 26


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="GestureScope real-time multi-hand gesture MVP")
    parser.add_argument("--camera", type=int, default=0, help="Webcam index. Usually 0 for the default camera.")
    parser.add_argument("--max-hands", type=int, default=6, help="Maximum number of hands to detect.")
    parser.add_argument("--min-detection-confidence", type=float, default=0.55)
    parser.add_argument("--min-tracking-confidence", type=float, default=0.55)
    parser.add_argument("--export-dir", type=str, default="outputs", help="Folder for reports and screenshots.")
    parser.add_argument("--width", type=int, default=1280, help="Requested camera width.")
    parser.add_argument("--height", type=int, default=720, help="Requested camera height.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    export_dir = Path(args.export_dir)
    export_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera index {args.camera}. Try --camera 1 or check camera permissions.")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 1:
        fps = 30.0

    detector = MultiHandGestureDetector(
        max_hands=args.max_hands,
        min_detection_confidence=args.min_detection_confidence,
        min_tracking_confidence=args.min_tracking_confidence,
    )
    session = GestureSession(fps=fps, source=f"webcam:{args.camera}")

    recording = False
    frame_index = 0
    last_export_paths = None

    print("GestureScope real-time demo started.")
    print("Press r to start/stop recording, e to export, s for screenshot, q to quit.")

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Camera frame could not be read. Exiting.")
                break

            frame = cv2.flip(frame, 1)  # Mirror view feels natural for classroom demos.
            detections = detector.process_frame(frame, frame_index)

            if recording:
                timestamp_sec = make_timestamp(frame_index, fps)
                session.add_detections(frame_index, timestamp_sec, detections)

            detector.draw_detections(frame, detections)
            draw_hud(frame, recording, frame_index, len(detections), str(export_dir))

            cv2.imshow("GestureScope MVP - Multi-Hand Gesture Logger", frame)
            key = cv2.waitKey(1) & 0xFF

            if key == ord("r"):
                recording = not recording
                print("Recording started." if recording else "Recording paused.")

            elif key == ord("e"):
                last_export_paths = session.export(export_dir)
                print(f"Exported report to: {last_export_paths['session_dir']}")

            elif key == ord("s"):
                screenshot_path = export_dir / f"screenshot_{session.session_id}_{frame_index}.jpg"
                cv2.imwrite(str(screenshot_path), frame)
                print(f"Saved screenshot: {screenshot_path}")

            elif key == ord("q"):
                break

            frame_index += 1

    finally:
        if session.events:
            last_export_paths = session.export(export_dir)
            print(f"Final report exported to: {last_export_paths['session_dir']}")
        detector.close()
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
