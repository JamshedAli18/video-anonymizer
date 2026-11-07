# src/debug_labels.py — inspect CVAT labels vs detections on a few frames
import sys
import zipfile

import cv2
import numpy as np

from detect import FaceDetector
from eval_video import MODEL, blur_mask, face_coverage, load_mot

clip = sys.argv[1] if len(sys.argv) > 1 else "C"
zp, vp = f"data/videos/labels/{clip}.zip", f"data/videos/test/{clip}.mp4"

with zipfile.ZipFile(zp) as z:
    print("Zip contents:", z.namelist())
    gt_name = [n for n in z.namelist() if n.endswith("gt.txt")][0]
    print("First 3 lines of gt.txt:")
    for line in z.read(gt_name).decode().splitlines()[:3]:
        print("   ", line)

gt = load_mot(zp)
cap = cv2.VideoCapture(vp)
W, H = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(f"Video size: {W}x{H} | labelled frames: {min(gt)}..{max(gt)} ({len(gt)} frames)")

det = FaceDetector(MODEL, score_thr=0.5, scales=[640])
for target in [0, 100, 200]:
    cap.set(cv2.CAP_PROP_POS_FRAMES, target)
    ok, frame = cap.read()
    if not ok:
        continue
    d = det.detect(frame)
    boxes = d[:, :4].copy(); boxes[:, 2:] += boxes[:, :2]
    G = np.array([x[1:] for x in gt.get(target, [])], np.float32).reshape(-1, 4)
    m = blur_mask(boxes, H, W)
    cov = face_coverage(G, m)

    print(f"\nFrame {target}")
    print("  GT boxes   (x1,y1,x2,y2):", G.round().astype(int).tolist())
    print("  Pred boxes (x1,y1,x2,y2):", boxes.round().astype(int).tolist())
    print("  Coverage per GT face    :", [None if c is None else round(c, 3) for c in cov])

    vis = frame.copy()
    vis[m > 0] = (0.5 * vis[m > 0] + [0, 127, 0]).astype(np.uint8)      # green = blurred area
    for x1, y1, x2, y2 in G.astype(int):
        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 0, 255), 2)             # red = label
    for x1, y1, x2, y2 in boxes.astype(int):
        cv2.rectangle(vis, (x1, y1), (x2, y2), (255, 0, 0), 2)             # blue = detection
    cv2.imwrite(f"results/debug_{clip}_{target}.jpg", vis)

print("\nSaved results/debug_C_*.jpg  (red = your label, blue = detection, green = blurred area)")