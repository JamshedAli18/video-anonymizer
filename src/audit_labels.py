# src/audit_labels.py — list likely unlabelled faces (detections >= MIN size with no matching label)
import sys

import cv2
import numpy as np

from detect import FaceDetector
from eval_video import EVAL_FRAMES, MIN_FACE_H, MODEL, load_mot
from track import iou_matrix

clip = sys.argv[1] if len(sys.argv) > 1 else "A"
gt = load_mot(f"data/videos/labels/{clip}.zip")
cap = cv2.VideoCapture(f"data/videos/test/{clip}.mp4")
det = FaceDetector(MODEL, score_thr=0.5, scales=[640, 1600])
n = EVAL_FRAMES.get(clip, 180)

print(f"Clip {clip}: checking every 10th frame of 0..{n-1} for unlabelled faces >= {MIN_FACE_H}px")
total = 0
for f in range(0, n, 10):
    cap.set(cv2.CAP_PROP_POS_FRAMES, f)
    ok, frame = cap.read()
    if not ok:
        break
    d = det.detect(frame)
    boxes = d[:, :4].copy(); boxes[:, 2:] += boxes[:, :2]
    boxes = boxes[(boxes[:, 3] - boxes[:, 1]) >= MIN_FACE_H]
    G = np.array([x[1:] for x in gt.get(f, [])], np.float32).reshape(-1, 4)
    near = iou_matrix(boxes, G).max(1) if len(G) else np.zeros(len(boxes))
    missing = boxes[near < 0.1]
    if len(missing):
        total += len(missing)
        print(f"  frame {f:>3}: {len(missing)} possible unlabelled face(s) at (x1,y1,x2,y2) "
              f"{missing.round().astype(int).tolist()}")
        vis = frame.copy()
        for x1, y1, x2, y2 in G.astype(int):
            cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 0, 255), 2)
        for x1, y1, x2, y2 in missing.astype(int):
            cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 255), 2)
        cv2.imwrite(f"results/audit_{clip}_{f:03d}.jpg", vis)
print(f"\nTotal possible unlabelled faces: {total}  (yellow = check these, red = your labels)")