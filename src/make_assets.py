# src/make_assets.py — website assets: anonymised demo videos (H.264), before/after stills, results data
import csv
import json
import os
import shutil
import subprocess

import cv2
import imageio_ffmpeg
import numpy as np

from anonymize import anonymize
from detect import FaceDetector
from track import IoUTracker

MODEL = "models/face_detection_yunet_2026may.onnx"
OUT = "web_assets"
CLIPS = ["A", "C"]
STILL_FRAMES = {"A": [60, 150], "C": [60, 200]}
FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()

for sub in ["videos", "images", "data"]:
    os.makedirs(f"{OUT}/{sub}", exist_ok=True)


def to_h264(src, dst):
    subprocess.run([FFMPEG, "-y", "-loglevel", "error", "-i", src, "-c:v", "libx264", "-pix_fmt", "yuv420p",
                    "-crf", "26", "-preset", "veryfast", "-movflags", "+faststart", "-an", dst], check=True)
    os.remove(src)


def label(img, text):
    cv2.rectangle(img, (0, 0), (len(text) * 11 + 16, 30), (0, 0, 0), -1)
    cv2.putText(img, text, (8, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    return img


for clip in CLIPS:
    cap = cv2.VideoCapture(f"data/videos/test/{clip}.mp4")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    W, H = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    det, tracker = FaceDetector(MODEL, score_thr=0.5, scales=[640]), IoUTracker()   # "fast" mode

    tmp_anon, tmp_side = f"{OUT}/videos/_{clip}_anon.mp4", f"{OUT}/videos/_{clip}_side.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    w_anon = cv2.VideoWriter(tmp_anon, fourcc, fps, (W, H))
    w_side = cv2.VideoWriter(tmp_side, fourcc, fps, (W, H // 2))

    f = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        orig = frame.copy()
        items = tracker.update(det.detect(frame))
        anonymize(frame, [b for b, _, _ in items], method="blur")
        w_anon.write(frame)

        left = label(cv2.resize(orig, (W // 2, H // 2)), "Original")
        right = label(cv2.resize(frame, (W // 2, H // 2)), "Anonymised (fast mode)")
        w_side.write(np.hstack([left, right]))

        if f in STILL_FRAMES[clip]:
            q = [cv2.IMWRITE_JPEG_QUALITY, 85]
            cv2.imwrite(f"{OUT}/images/{clip}_{f:03d}_before.jpg", orig, q)
            cv2.imwrite(f"{OUT}/images/{clip}_{f:03d}_after.jpg", frame, q)
            boxes = frame.copy()
            for b, tid, pred in items:
                c = (0, 165, 255) if pred else (0, 255, 0)
                cv2.rectangle(boxes, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), c, 2)
            cv2.imwrite(f"{OUT}/images/{clip}_{f:03d}_boxes.jpg", boxes, q)
        f += 1

    cap.release(); w_anon.release(); w_side.release()
    to_h264(tmp_anon, f"{OUT}/videos/{clip}_anonymised.mp4")
    to_h264(tmp_side, f"{OUT}/videos/{clip}_side_by_side.mp4")
    print(f"{clip}: {f} frames -> videos + {len(STILL_FRAMES[clip])} before/after stills")

# Results data for website charts
for name in ["widerface_eval.json", "widerface_eval_multiscale.json", "video_eval.json"]:
    if os.path.exists(f"results/{name}"):
        shutil.copy(f"results/{name}", f"{OUT}/data/{name}")

# Video credits (only clips actually used)
with open("data/videos/sources.csv") as fh:
    credits = [r for r in csv.DictReader(fh) if r["clip"] in CLIPS]
with open(f"{OUT}/data/credits.json", "w") as fh:
    json.dump(credits, fh, indent=1)

print("\nAssets:")
for root, _, files in os.walk(OUT):
    for fn in sorted(files):
        p = os.path.join(root, fn)
        print(f"  {p:<50} {os.path.getsize(p)/1e6:6.2f} MB")