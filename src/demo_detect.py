# src/demo_detect.py — baseline test: draw YuNet detections + measure detection time
import argparse
import time

import cv2
import numpy as np

from detect import FaceDetector

p = argparse.ArgumentParser()
p.add_argument("--source", default="0", help="0 = webcam, or path to a video file")
p.add_argument("--model", default="models/face_detection_yunet_2026may.onnx")
args = p.parse_args()

cap = cv2.VideoCapture(int(args.source) if args.source.isdigit() else args.source)
det = FaceDetector(args.model)
times, size = [], None

while True:
    ok, frame = cap.read()
    if not ok:
        break
    size = f"{frame.shape[1]}x{frame.shape[0]}"

    t0 = time.perf_counter()
    faces = det.detect(frame)
    times.append(time.perf_counter() - t0)

    for x, y, w, h, s in faces:
        cv2.rectangle(frame, (int(x), int(y)), (int(x + w), int(y + h)), (0, 255, 0), 2)
        cv2.putText(frame, f"{s:.2f}", (int(x), int(y) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    cv2.putText(frame, f"{len(faces)} faces | detect {times[-1]*1000:.1f} ms", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    cv2.imshow("YuNet baseline (press q to quit)", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
if times:
    print(f"frames {len(times)} | frame size {size} | median detect {np.median(times)*1000:.1f} ms "
          f"| p95 {np.percentile(times, 95)*1000:.1f} ms")