# src/pipeline.py — CLI: video/webcam in -> anonymised video out
import argparse
import time

import cv2
import numpy as np

from anonymize import anonymize
from detect import FaceDetector
from track import IoUTracker

MODES = {
    "fast":     dict(scales=[640], thr=0.5),
    "balanced": dict(scales=[1024], thr=0.3),
    "max":      dict(scales=[640, 1600], thr=0.3),
}

p = argparse.ArgumentParser(description="Detect, track and anonymise faces in video.")
p.add_argument("--source", default="0", help="0 = webcam, or path to a video file")
p.add_argument("--output", default="", help="optional output .mp4 path")
p.add_argument("--model", default="models/face_detection_yunet_2026may.onnx")
p.add_argument("--mode", default="fast", choices=list(MODES), help="speed / privacy trade-off")
p.add_argument("--method", default="blur", choices=["blur", "pixelate", "solid"])
p.add_argument("--no-track", action="store_true", help="detection only (for comparison)")
p.add_argument("--debug", action="store_true", help="draw boxes: green = detected, orange = predicted")
p.add_argument("--show", action="store_true")
args = p.parse_args()

cfg = MODES[args.mode]
cap = cv2.VideoCapture(int(args.source) if args.source.isdigit() else args.source)
fps_in = cap.get(cv2.CAP_PROP_FPS) or 30
det, tracker = FaceDetector(args.model, score_thr=cfg["thr"], scales=cfg["scales"]), IoUTracker()
writer, times, n_pred_frames, n_frames = None, [], 0, 0

while True:
    ok, frame = cap.read()
    if not ok:
        break
    t0 = time.perf_counter()

    faces = det.detect(frame)
    if args.no_track:
        items = [(np.array([x, y, x + w, y + h]), -1, False) for x, y, w, h, _ in faces]
    else:
        items = tracker.update(faces)

    anonymize(frame, [b for b, _, _ in items], method=args.method)
    times.append(time.perf_counter() - t0)

    n_frames += 1
    n_pred_frames += any(pred for _, _, pred in items)
    if args.debug:
        for b, tid, pred in items:
            c = (0, 165, 255) if pred else (0, 255, 0)
            cv2.rectangle(frame, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), c, 2)
            cv2.putText(frame, f"id {tid}", (int(b[0]), int(b[1]) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, c, 1)
        cv2.putText(frame, f"{args.mode} | {times[-1]*1000:.1f} ms | track {'off' if args.no_track else 'on'}",
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    if args.output:
        if writer is None:
            h, w = frame.shape[:2]
            writer = cv2.VideoWriter(args.output, cv2.VideoWriter_fourcc(*"mp4v"), fps_in, (w, h))
        writer.write(frame)
    if args.show:
        cv2.imshow("Anonymizer (press q to quit)", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

cap.release()
if writer:
    writer.release()
cv2.destroyAllWindows()
if times:
    med = np.median(times) * 1000
    print(f"mode {args.mode} | frames {n_frames} | median {med:.1f} ms/frame (~{1000/med:.0f} FPS) | "
          f"p95 {np.percentile(times, 95)*1000:.1f} ms | frames with tracker-filled boxes: {n_pred_frames}")