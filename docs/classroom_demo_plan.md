# Classroom Demo Plan

## Goal

Demonstrate a working multi-hand gesture recognition MVP that can detect several visible hands at once, label simple gestures, track hands over time, and export a document summarizing the session.

## Setup

1. Place a laptop/webcam facing the front of the classroom.
2. Use bright lighting and avoid strong backlighting.
3. Ask 2-4 students to stand where their hands are clearly visible.
4. Start with `--max-hands 4` or `--max-hands 6`.
5. Increase `--max-hands` only if performance remains smooth.

## Demo flow

1. Launch the app:

```bash
python run_realtime.py --max-hands 6 --export-dir outputs/classroom_demo
```

2. Press `r` to start recording.
3. Ask participants to perform simple gestures:
   - open palm
   - closed fist
   - pointing up
   - victory sign
   - thumbs up
   - pinch
4. Press `r` again to pause recording.
5. Press `e` to export the report.
6. Open the generated `gesture_report.md` or `gesture_report.html`.

## What to say during the demo

GestureScope uses computer vision to locate hand landmarks, then applies a simple gesture classification layer. Each hand is assigned a temporary tracking ID so that the report can summarize which gestures appeared over time.

## Limitations to mention honestly

- It is not a sign-language interpreter.
- It works best when hands are not overlapping.
- IDs can switch when hands leave the frame or cross over each other.
- The built-in gesture rules are simple and should later be replaced or improved with a trained classifier.

## Strong future extension

Use the exported landmark JSON as training data for a custom classifier. This turns the MVP into a data-collection tool and makes the project stronger for research or GitHub.
